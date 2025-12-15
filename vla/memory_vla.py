from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
import torch.nn.functional as F
from transformers import LlamaTokenizerFast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from action_model.action_model import ActionModel
from action_model.models import DiT

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    时间步嵌入器，将标量时间步转换为向量表示
    
    该模块主要用于为序列数据中的时间信息生成嵌入表示，
    采用正弦位置编码方式生成频率特征，然后通过MLP映射到目标维度。
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        初始化时间步嵌入器
        Args:
            hidden_size (int): 输出嵌入向量的维度大小
            frequency_embedding_size (int, optional): 频率嵌入的维度大小，默认为256
        """
        super().__init__()
        # 多层感知机，用于将频率嵌入映射到隐藏层维度
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),  # 使用SiLU激活函数
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建正弦时间步嵌入
        Args:
            t (Tensor): 形状为(N,)的一维张量，包含N个时间步索引，可以是小数
            dim (int): 输出嵌入的维度
            max_period (int, optional): 控制嵌入的最小频率，默认为10000  
        Returns:
            Tensor: 形状为(N, D)的正弦位置嵌入张量 
        References:
            https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        half = dim // 2  # 计算一半维度用于cos和sin
        # 计算不同频率的指数衰减因子
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        # 计算角度参数：t * freqs
        args = t[:, None].float() * freqs[None]
        # 拼接cos和sin得到位置嵌入
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果维度是奇数，添加零填充
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        前向传播，将时间步转换为嵌入向量
        Args:
            t (Tensor): 输入时间步张量  
        Returns:
            Tensor: 时间步嵌入向量
        """
        # 将输入移动到MLP所在的设备
        t = t.to(next(self.mlp.parameters()).device)
        # 生成时间步的频率嵌入并转换数据类型
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        # 通过MLP映射到最终的嵌入向量
        t_emb = self.mlp(t_freq)
        return t_emb


class CrossTransformerBlock(nn.Module):
    """
    跨注意力变换块，用于实现查询序列与键值序列之间的注意力交互
    
    该模块通过交叉注意力机制实现两个不同序列间的特征交互，
    常用于将当前工作记忆与历史记忆进行融合处理。
    """
    def __init__(self, feature_dim: int):
        """
        初始化交叉注意力变换块
        
        Args:
            feature_dim (int): 特征维度，所有输入输出张量的特征维度大小
        """
        super().__init__()
        # 查询、键、值的线性投影层，将输入映射到特征空间
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        # 注意力输出后的层归一化
        self.attn_norm = nn.LayerNorm(feature_dim)

        # 前馈神经网络（Feed-Forward Network）
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),  # 特征维度扩展4倍
            nn.GELU(),                                # GELU激活函数
            nn.Linear(feature_dim * 4, feature_dim)   # 投影回原始特征维度
        )
        # 前馈网络输出后的层归一化
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self,
                query: torch.Tensor, # (B, N, D) - 查询序列
                k: torch.Tensor,     # (B, M, D) - 键序列
                v: torch.Tensor      # (B, M, D) - 值序列
                ) -> torch.Tensor:
        """
        前向传播过程，执行交叉注意力计算和前馈网络处理
        
        Args:
            query (torch.Tensor): 查询序列，形状为(B, N, D)
                B: 批次大小
                N: 查询序列长度
                D: 特征维度
            k (torch.Tensor): 键序列，形状为(B, M, D)
                M: 键序列长度
                D: 特征维度（与查询序列一致）
            v (torch.Tensor): 值序列，形状为(B, M, D)
                M: 值序列长度（通常与键序列长度相同）
                D: 特征维度（与键序列一致）
                
        Returns:
            torch.Tensor: 经过交叉注意力和前馈网络处理后的特征，形状为(B, N, D)
        """
        # 对查询、键、值进行线性投影
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 执行缩放点积注意力计算
        # 使用查询、键、值计算注意力权重，并应用于值序列
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        # 残差连接 + 层归一化
        # 将注意力输出与原始查询进行残差连接，并进行层归一化
        x = self.attn_norm(query + attn_out)

        # 前馈神经网络处理
        ffn_out = self.ffn(x)
        
        # 残差连接 + 层归一化
        # 将前馈网络输出与输入进行残差连接，并进行层归一化
        return self.ffn_norm(x + ffn_out)


class BottleneckSE(nn.Module):
    # 构建per-token压缩模块，用于降低视觉特征维度
    def __init__(self, C_in, C_mid, C_out):
        """
        初始化函数，构建一个包含降维、激励和升维操作的神经网络模块
        参数:
            C_in (int): 输入通道数
            C_mid (int): 中间通道数
            C_out (int): 输出通道数
        """
        super().__init__()
        self.C_in = C_in
        self.C_mid = C_mid
        self.C_out = C_out

        # 1x1卷积层用于降低输入特征图的通道维度
        self.reduce = nn.Conv2d(C_in, C_mid, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        # 注意力激励模块：通过全局平均池化和全连接层计算通道注意力权重
        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_mid, C_mid//16, 1),
            nn.ReLU(),
            nn.Conv2d(C_mid//16, C_mid, 1),
            nn.Sigmoid()
        )
        # 1x1卷积层用于将特征图扩展到目标输出通道数
        self.expand = nn.Conv2d(C_mid, C_out, 1, bias=False)

    def forward(self, x):
        """
        前向传播函数，处理输入特征并应用空间注意力机制
        参数:
            x: 输入张量，形状为(B, N, C)，其中B是批次大小，N是特征数量，C是通道数 
        返回:
            处理后的张量，形状为(B, N, C_out)，其中C_out是输出通道数
        """
        _b, _n, _c = x.shape
        _h = _w = int(math.sqrt(_n))
        assert _h * _h == _n, "Input feature has no spatial structure"
        # 重塑输入张量为4D格式以便进行卷积操作
        x = x.reshape(_b, _h, _w, _c).permute(0, 3, 1, 2)  # (B, C_in, H, W)
        z = self.act(self.reduce(x))
        w = self.excite(z)
        # 应用注意力权重并扩展到目标通道数
        final = self.expand(z * w)
        final = final.reshape(_b, self.C_out, _n).permute(0, 2, 1)
        return final


class GateFusion(nn.Module):
    """
    门控融合模块，用于自适应地融合两个特征张量
    
    该模块通过学习一个门控权重来动态调整两个输入特征的融合比例，
    实现更灵活的特征融合效果。
    """
    def __init__(self, dim: int):
        """
        初始化门控融合模块
        Args:
            dim (int): 输入特征的维度，两个输入张量应具有相同的维度
        """
        super().__init__()
        # 创建线性投影层，将2*dim的拼接特征映射到dim维度
        self.proj = nn.Linear(dim * 2, dim)
        # 对权重进行小方差的正态分布初始化
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        # 对偏置进行小方差的正态分布初始化
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        前向传播，执行门控特征融合
        Args:
            x1 (torch.Tensor): 第一个输入特征张量，形状为[..., dim]
            x2 (torch.Tensor): 第二个输入特征张量，形状为[..., dim] 
        Returns:
            torch.Tensor: 融合后的特征张量，形状与输入相同
        """
        # 将两个输入张量在最后一个维度上拼接
        # 通过线性变换和sigmoid激活函数计算门控权重
        scale = torch.sigmoid(
            self.proj(
                torch.cat([x1, x2],
                dim=-1)
            )
        )
        # 使用门控权重对两个输入进行加权融合
        # scale * x1 + (1 - scale) * x2 实现了自适应加权平均
        fused = scale * x1 + (1 - scale) * x2
        return fused


class CogMemBank(nn.Module):
    """
    认知记忆库（CogMemBank）
    """
    def __init__(self,
                 dataloader_type: str,
                 group_size: int,
                 token_size: int,
                 mem_length: int = 16,
                 retrieval_layers: int = 2,
                 use_timestep_pe: bool = True,
                 fusion_type: str = 'gate',
                 consolidate_type: str = 'tome',
                 update_fused: bool = False,
                 ):
        """
        初始化模型组件和配置参数
        Args:
            dataloader_type (str): 数据加载器类型，必须是 'stream' 或 'group'
            group_size (int): 分组大小，用于数据分组处理
            token_size (int): token维度大小，用于表示每个token的特征维度
            mem_length (int, optional): 内存长度，默认为16，控制记忆序列的长度
            retrieval_layers (int, optional): 检索层的数量，默认为2，用于跨注意力计算
            use_timestep_pe (bool, optional): 是否使用时间步位置编码，默认为True
            fusion_type (str, optional): 融合类型，必须是 'gate' 或 'add'，控制特征融合方式
            consolidate_type (str, optional): 合并类型，必须是 'fifo' 或 'tome'，控制序列合并策略
            update_fused (bool, optional): 是否更新融合特征，默认为False
        Returns:
            None
        """
        super().__init__()
        # 参数有效性验证
        assert dataloader_type in ('stream', 'group')
        assert fusion_type in ('gate', 'add')
        assert consolidate_type in ('fifo', 'tome')
        # 存储配置参数
        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused
        # 初始化检索模块，包含多个交叉注意力块
        self.retrieval_blocks = nn.ModuleList([
            CrossTransformerBlock(self.token_size)
            for _ in range(self.retrieval_layers)
        ])
        # 根据融合类型初始化门控融合模块
        if self.fusion_type == 'gate':
            self.gate_fusion_blocks = GateFusion(self.token_size)
        # 初始化时间步编码器，用于添加时间信息
        if self.use_timestep_pe:
            self.timestep_encoder = TimestepEmbedder(
                self.token_size,
                frequency_embedding_size=self.token_size // 4)
        else:
            self.timestep_encoder = None
        # 重置模型状态
        self.reset()

    def reset(self):
        # bank[episode_id] = [(timestep, feat[N,D]), ...]
        self.bank = {}
        self.eid_stream = None

    def clear_episode(self, episode_id):
        self.bank.pop(episode_id, None)

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id):
        """
        通过特征融合来合并最相似的相邻时间步的特征向量
        该函数计算相邻时间步特征之间的余弦相似度，找到最相似的一对相邻特征，
        然后将它们融合成一个特征向量，从而减少特征库中的条目数量。
        参数:
            episode_id: 用于索引特征库的标识符 
        返回值:
            无返回值，直接修改实例的bank属性
        """
        bank = self.bank.get(episode_id, [])
        T = len(bank)
        if T < 2:
            return
        feats = [feat for (_, feat) in bank]
        # 计算相邻时间步特征之间的余弦相似度
        sims = []
        for i in range(T - 1):
            f1 = feats[i].flatten(1) if feats[i].dim() > 1 else feats[i].unsqueeze(0)
            f2 = feats[i+1].flatten(1) if feats[i+1].dim() > 1 else feats[i+1].unsqueeze(0)
            sims.append(F.cosine_similarity(f1, f2, dim=1).mean().item())
        # 找到相似度最高的相邻特征对
        idx_max = int(torch.tensor(sims).argmax().item())
        # 融合最相似的相邻特征对
        timestep_i, feat_i = bank[idx_max]
        timestep_j, feat_j = bank[idx_max + 1]
        fused_feat = 0.5 * (feat_i + feat_j)
        # 用融合后的特征替换原来的特征，并删除被合并的特征
        bank[idx_max] = (timestep_i, fused_feat.detach().clone())
        bank.pop(idx_max + 1)

    @torch.no_grad()
    def _memory_consolidate(
            self,
            episode_id,
            feat: torch.Tensor,
            timestep: Optional[torch.Tensor]):
        """
        巩固记忆库中的特征表示，根据指定的策略维护每个episode的记忆长度。
        当前episode的记忆超过预设长度时，根据巩固策略进行裁剪或合并：
        - FIFO策略：保留最新的记忆项
        - ToMe策略：使用token合并方法进行记忆压缩
        参数:
            episode_id: episode的唯一标识符，用于索引对应的记忆库
            feat: 当前时间步的特征张量，需要存储到记忆库中
            timestep: 可选的时间步信息，用于记录特征对应的时间戳 
        返回值:
            无返回值
        """
        # 如果当前episode_id不在记忆库中，则初始化一个空列表
        if episode_id not in self.bank:
            self.bank[episode_id] = []
        # 将当前时间步的特征（脱离计算图的副本）添加到对应episode的记忆库中
        self.bank[episode_id].append((timestep, feat.detach().clone()))
        # 当记忆库长度超过预设的最大长度时，执行记忆巩固策略
        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == 'fifo':
                # FIFO策略：只保留最新的mem_length个记忆项
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length:]
            elif self.consolidate_type == "tome":
                # ToMe策略：调用token合并方法进行记忆压缩
                self._consolidate_with_token_merge(episode_id)
            else:
                raise NotImplementedError

    def process_batch(
        self,
        tokens: torch.Tensor, # [B, N, D_role]
        episode_ids: np.array,
        timesteps: np.array,
    ) -> torch.Tensor:
        """
        FIXME：记忆处理
        处理一个批次的输入tokens，结合历史记忆进行增强，并返回增强后的特征。
        加入长期记忆(history)
        从历史记忆中检索(cross attention)
        融合检索结果(gate fusion)
        参数:
            tokens (torch.Tensor): 输入的 token 特征，形状为 [B, N, D_role]。
            episode_ids (np.array): 每个样本所属的 episode ID，用于管理记忆。
            timesteps (np.array): 每个样本对应的时间步，用于时间编码。
        返回:
            torch.Tensor: 经过记忆增强后的特征，形状为 [B, N, D_role]。
        """
        assert episode_ids is not None, "episode_ids must be provided during training"

        if self.use_timestep_pe:
            assert timesteps is not None, "timesteps must be provided during training"

        B, N, D = tokens.shape
        outputs = []
        # 训练阶段进行 episode 管理逻辑
        if self.training:
            if self.dataloader_type == 'group':
                self.bank.clear()
                self.eid_stream = None
            elif self.dataloader_type == 'stream':
                first_eid = episode_ids[0]
                if self.eid_stream is not None and self.eid_stream != first_eid:
                    self.clear_episode(self.eid_stream)
                self.eid_stream = first_eid

        for i in range(B):
            # FIXME：1) episode 管理逻辑
            eid = episode_ids[i]
            if self.training:
                if self.dataloader_type == 'group':
                    if i > 0 and i % self.group_size == 0:
                        prev_group_eid = episode_ids[i - self.group_size]
                        self.clear_episode(prev_group_eid)
                if self.dataloader_type == 'stream':
                    if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                        self.clear_episode(episode_ids[i - 1])
                        self.eid_stream = episode_ids[i]

            # FIXME：2) 从记忆库中检索历史信息并进行注意力交互
            # 1. 获取当前样本和历史记忆----
            working_mem = tokens[i].unsqueeze(0)  # (1, N, D)
            hist = self.bank.get(eid, [])
            # ----
            if len(hist) > 0:
                hist_feats = [feat for _, feat in hist]
                episode_mem = torch.stack(hist_feats, dim=0).reshape(-1, D).unsqueeze(0)  # (1, T*N, D)

                if self.use_timestep_pe:
                    hist_timesteps = [t for t, _ in hist]
                    hist_timesteps = torch.tensor(hist_timesteps).to(working_mem.device)
                    pe = self.timestep_encoder(hist_timesteps).unsqueeze(0)  # (1, T, D)
                    pe = pe.repeat_interleave(N, dim=1) # (1, T*N, D)
                else:
                    pe = torch.zeros_like(episode_mem)

                query = working_mem
                for block in self.retrieval_blocks:
                    query = block(query, episode_mem + pe, episode_mem)
                retrieved_episode_mem = query
            else:
                # 没有历史信息时，使用当前工作记忆作为 episode 记忆
                retrieved_episode_mem = working_mem  # (1, N, D)

            # FIXME：3) 记忆融合：将当前工作记忆与检索到的记忆进行融合
            if self.fusion_type == 'add':
                fused_feats = (working_mem + retrieved_episode_mem) * 0.5
            elif self.fusion_type == 'gate':
                fused_feats = self.gate_fusion_blocks(working_mem, retrieved_episode_mem)

            outputs.append(fused_feats)

            # FIXME：4) 将融合后的特征或原始特征存入记忆库
            timestep_i = timesteps[i] if self.use_timestep_pe else None
            if self.update_fused:
                self._memory_consolidate(eid, fused_feats.squeeze(0), timestep_i)
            else:
                self._memory_consolidate(eid, tokens[i], timestep_i)
        return torch.cat(outputs, dim=0)  # [B, N, D_role]


class PerMemBank(CogMemBank):
    """
    感知记忆库(PerMemBank)
    
    感知记忆库用于存储和处理感知相关的特征信息，继承自认知记忆库(CogMemBank)。
    该类主要负责处理视觉感知特征的记忆管理，包括特征的存储、检索和融合。
    与认知记忆库使用相同的架构和参数配置，但专门用于处理感知特征。
    """
    def __init__(self,
                 dataloader_type: str,           # 数据加载器类型，支持 'stream' 或 'group'
                 group_size: int,                # 分组大小，用于数据分组处理
                 token_size: int,                # token维度大小，表示每个token的特征维度
                 mem_length: int = 16,           # 内存长度，控制记忆序列的最大长度，默认为16
                 retrieval_layers: int = 2,      # 检索层数量，用于跨注意力计算，默认为2
                 use_timestep_pe: bool = True,   # 是否使用时间步位置编码，默认为True
                 fusion_type: str = 'gate',      # 融合类型，支持 'gate' 或 'add'，控制特征融合方式，默认为'gate'
                 consolidate_type: str = 'tome', # 合并类型，支持 'fifo' 或 'tome'，控制序列合并策略，默认为'tome'
                 update_fused: bool = False,     # 是否更新融合特征，默认为False
                 ):
        """
        初始化感知记忆库
        
        参数说明:
            dataloader_type (str): 数据加载器类型，必须是 'stream' 或 'group'
            group_size (int): 分组大小，用于数据分组处理
            token_size (int): token维度大小，用于表示每个token的特征维度
            mem_length (int, optional): 内存长度，默认为16，控制记忆序列的长度
            retrieval_layers (int, optional): 检索层的数量，默认为2，用于跨注意力计算
            use_timestep_pe (bool, optional): 是否使用时间步位置编码，默认为True
            fusion_type (str, optional): 融合类型，必须是 'gate' 或 'add'，控制特征融合方式
            consolidate_type (str, optional): 合并类型，必须是 'fifo' 或 'tome'，控制序列合并策略
            update_fused (bool, optional): 是否更新融合特征，默认为False
        """
        super().__init__(
            dataloader_type=dataloader_type,
            group_size=group_size,
            token_size=token_size,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            use_timestep_pe=use_timestep_pe,
            fusion_type=fusion_type,
            consolidate_type=consolidate_type,
            update_fused=update_fused,
        )


class MemoryVLA(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_model_type: str = 'DiT-L',
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        dataloader_type: str = "group",
        group_size: int = 16,
        per_token_size: int = 256,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = 'gate',
        consolidate_type: str = 'tome',
        update_fused: bool = False,
        **kwargs,
    ) -> None:
        """
        初始化模型组件，包括视觉语言模型、动作预测模型以及记忆模块。

        参数:
            vlm (PrismaticVLM): 预训练的视觉-语言模型。
            action_model_type (str): 动作模型的类型，默认为 'DiT-L'。
            token_size (int): 用于动作建模的token大小，默认为4096。
            action_dim (int): 动作空间的维度，默认为7。
            future_action_window_size (int): 预测未来动作的时间窗口大小，默认为15。
            use_ema (bool): 是否使用指数移动平均模型，默认为False。
            norm_stats (Dict): 归一化统计信息，结构复杂，默认为None。
            dataloader_type (str): 数据加载方式，默认为"group"。
            group_size (int): 分组大小，默认为16。
            per_token_size (int): 每个token的特征维度，默认为256。
            mem_length (int): 记忆长度，默认为16。
            retrieval_layers (int): 用于检索的层数，默认为2。
            use_timestep_pe (bool): 是否在时间步中使用位置编码，默认为True。
            fusion_type (str): 特征融合类型，默认为'gate'。
            consolidate_type (str): token压缩策略，默认为'tome'。
            update_fused (bool): 是否更新融合后的特征，默认为False。
            **kwargs: 其他未使用的参数。

        返回值:
            None
        """
        super().__init__()
        # 保存传入的参数到实例变量
        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.use_ema = use_ema
        self.norm_stats = norm_stats

        self.cog_token_size = token_size

        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.per_token_size = per_token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused

        self.cur_timestep = 0
        # 计算视觉特征维度：DINO和SigLIP特征提取器输出通道之和
        self.vision_dim = self.vlm.vision_backbone.dino_featurizer.patch_embed.proj.weight.shape[0] + \
                 self.vlm.vision_backbone.siglip_featurizer.patch_embed.proj.weight.shape[0]

        # 构建per-token压缩模块，用于降低视觉特征维度
        self.per_compr = BottleneckSE(
            C_in=self.vision_dim,
            C_mid=self.per_token_size * 2,
            C_out=self.per_token_size,
        )
        # 初始化认知记忆库（CogMemBank）和感知记忆库（PerMemBank）
        self.cog_mem_bank = CogMemBank(
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            token_size=self.cog_token_size,
            mem_length=self.mem_length,
            retrieval_layers=self.retrieval_layers,
            use_timestep_pe=self.use_timestep_pe,
            fusion_type=self.fusion_type,
            consolidate_type=self.consolidate_type,
            update_fused=self.update_fused,
        )
        self.per_mem_bank = PerMemBank(
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            token_size=self.per_token_size,
            mem_length=self.mem_length,
            retrieval_layers=self.retrieval_layers,
            use_timestep_pe=self.use_timestep_pe,
            fusion_type=self.fusion_type,
            consolidate_type=self.consolidate_type,
            update_fused=self.update_fused,
        )
        # 初始化动作预测模型
        self.action_model = ActionModel(
            model_type=action_model_type,
            token_size=token_size,
            in_channels=action_dim,
            future_action_window_size=future_action_window_size,
            use_per_attn=True,
            per_token_size=per_token_size,
        )
        # 初始化模块键列表
        self.all_module_keys = []
        self._trainable_module_keys = []
        # 如果启用EMA，则复制动作模型并冻结梯度
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys.append('ema_diffusion')
        # 添加VLM模块中的所有子模块键
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)
        # 遍历当前模型的所有子模块，记录可训练模块
        for name, module in self.named_children():
            if name != "vlm" and any(p.requires_grad for p in module.parameters()):
                self.all_module_keys.append(name)
                self._trainable_module_keys.append(name)

    @property
    def trainable_module_keys(self) -> List[str]:
        """
        获取所有可训练模块的键名列表
        该属性将视觉语言模型(VLM)中的可训练模块键名与当前对象的可训练模块键名进行合并，
        返回完整的可训练模块键名列表。其中VLM模块的键名会添加"vlm."前缀以区分来源。
        Returns:
            List[str]: 包含所有可训练模块键名的列表，其中VLM模块键名带有"vlm."前缀
        """
        keys = []
        # 收集VLM模型中的可训练模块键名，并添加"vlm."前缀
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        # 合并当前对象自身的可训练模块键名
        keys += self._trainable_module_keys
        return keys
    
    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone
    
    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        action_masks: Optional[torch.FloatTensor] = None,
        timesteps: np.array = None,
        episode_ids: np.array = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
    ) -> Tuple:
        """
        执行一次前向传播过程，通过视觉-语言模型（VLM）处理输入，并结合认知与感知记忆模块以及动作预测模型计算最终损失。
        参数：
            input_ids (torch.LongTensor, optional): 输入文本的 token ID 序列。
            attention_mask (Optional[torch.Tensor], optional): 指示哪些位置是有效输入的注意力掩码。
            pixel_values (Optional[torch.FloatTensor], optional): 图像像素值，用于视觉编码。
            labels (Optional[torch.LongTensor], optional): 用于语言建模任务的目标标签。
            actions (Optional[torch.FloatTensor], optional): 动作序列数据。
            action_masks (Optional[torch.FloatTensor], optional): 对动作序列进行掩码操作的数据。
            timesteps (np.array, optional): 时间步信息，用于记忆模块的时间管理。
            episode_ids (np.array, optional): 回合计数标识，用于区分不同 episode 的样本。
            inputs_embeds (Optional[torch.FloatTensor], optional): 已经嵌入后的输入表示。
            past_key_values (Optional[List[torch.FloatTensor]], optional): 缓存的历史键值对，用于加速解码。
            use_cache (Optional[bool], optional): 是否使用缓存机制。
            output_attentions (Optional[bool], optional): 是否输出注意力权重。
            output_hidden_states (Optional[bool], optional): 是否输出所有隐藏层状态。
            return_dict (Optional[bool], optional): 是否以字典形式返回结果。
            repeated_diffusion_steps (int, optional): 在扩散过程中重复动作预测的次数，默认为 4。

        返回：
            Tuple: 包含两个元素：
                - loss (torch.Tensor): 计算得到的动作预测损失。
                - output (CausalLMOutputWithPast): 来自 VLM 前向传播的结果对象，包含语言建模相关的输出及损失等信息。
        """
        output = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取图像中 patch 的数量，根据不同的视觉主干网络结构选择对应属性
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        # 获取最后一层隐藏状态并去除视觉 token 部分
        last_hidden_state = output.hidden_states[-1]
        last_hidden_state = last_hidden_state[:, num_patch :] # last_hidden_state[0] 总的VLM输出，last_hidden_state[1] 删除了前num_patch个视觉tokens

        # 根据 attention mask 定位每条序列最后一个有效 token 的索引，并提取其对应的特征作为认知 token
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden_state.size(-1))

        cog_tokens = last_hidden_state.gather(
            1, expanded_indices.unsqueeze(1))  # [B, 1, D]

        # 获取视觉特征并通过压缩模块处理
        vision_feats = self.vlm.vision_feats
        per_tokens = self.per_compr(vision_feats)

        # 使用认知和感知记忆库分别更新当前 batch 中的 token 表征
        cog_tokens = self.cog_mem_bank.process_batch(
            tokens=cog_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )
        per_tokens = self.per_mem_bank.process_batch(
            tokens=per_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )

        # 将动作序列在时间维度上截断后复制多次，适配后续扩散步骤的需求
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)

        # 同样地将认知和感知 token 复制多份以匹配动作张量形状
        cog_tokens_repeated = cog_tokens.repeat(
            repeated_diffusion_steps, 1, 1)
        per_tokens_repeated = per_tokens.repeat(
            repeated_diffusion_steps, 1, 1)

        # 调用动作模型计算损失
        loss = self.action_model.loss(
            actions_repeated,
            cog_tokens_repeated,
            per_tokens_repeated,
        )
        return loss, output

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        为模型的不同组件返回一个FSDP包装策略的组合策略。
        该方法创建一个联合策略，将视觉主干、语言模型主干和Prismatic特定模块的FSDP包装策略组合在一起。
        使用_or_policy确保每个模块只会被一个最适合的策略包装，避免重复包装。
        Returns:
            Callable: 一个部分应用的_or_policy函数，包含所有子组件的包装策略。
                     任何未被特定策略覆盖的模块将自动被归入根VLM FSDP实例。
        """
        # 获取视觉主干网络的FSDP包装策略
        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        # 获取语言模型主干网络的FSDP包装策略
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()
        # 创建Prismatic特定模块的包装策略
        # 包括各种投影器(LinearProjector, MLPProjector, FusedMLPProjector)和DiT模块
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
        )
        # 返回组合策略：使用_or_policy将所有策略合并
        # 注意：没有后备策略；任何未被上述策略覆盖的模块将自动归入根VLM FSDP实例
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        action_model_type: str = 'DiT-L',
        use_ema: bool = False,
        norm_stats = None,
        **kwargs,
    ) -> MemoryVLA:
        """
        从预训练检查点加载一个 MemoryVLA 模型实例。

        参数：
            pretrained_checkpoint (Path): 预训练模型权重文件路径。
            model_id (str): 模型标识符，用于初始化视觉-语言模型结构。
            vision_backbone (VisionBackbone): 视觉主干网络配置对象。
            llm_backbone (LLMBackbone): 大语言模型主干网络配置对象。
            enable_mixed_precision_training (bool, optional): 是否启用混合精度训练。默认为 True。
            arch_specifier (str, optional): 架构指定字符串（如激活函数、MLP 类型等）。默认为 "gelu-mlp"。
            freeze_weights (bool, optional): 是否冻结所有模型参数的梯度。默认为 True。
            action_dim (int, optional): 动作空间维度。默认为 7。
            future_action_window_size (int, optional): 预测未来动作的时间窗口大小。默认为 15。
            action_model_type (str, optional): 动作预测模块使用的模型类型。默认为 'DiT-L'。
            use_ema (bool, optional): 是否使用指数移动平均(EMA)权重进行推理。默认为 False。
            norm_stats (optional): 归一化统计信息（例如均值和标准差），用于数据标准化。
            **kwargs: 其他传递给 PrismaticVLM 和 MemoryVLA 初始化的额外关键字参数。

        返回：
            MemoryVLA: 加载完成并初始化好的 MemoryVLA 实例。
        """

        # 加载 VLM 主干网络，复用 PrismaticVLM 的实现
        vlm = PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # 从检查点加载模型权重（自定义格式 --> 应同时包含 projector 和 llm 权重）
        model_state_dict = torch.load(
            pretrained_checkpoint,
            # map_location="cpu",
            map_location="cuda",
        )["model"]
        # 确保检查点包含 projector 和 llm_backbone 的权重
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM [from_pretrained](file://d:\doctor-code\code\MemoryVLA-openvla-codebase\MemoryVLA-openvla-codebase\vla\memory_vla.py#L795-L876) expects checkpoint with keys for [projector](file://d:\doctor-code\code\MemoryVLA-openvla-codebase\MemoryVLA-openvla-codebase\prismatic\util\nn_utils.py#L0-L0) AND [llm_backbone](file://d:\doctor-code\code\MemoryVLA-openvla-codebase\MemoryVLA-openvla-codebase\vla\memory_vla.py#L640-L641)!"

        # 加载 projector 和 llm_backbone 的权重
        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        # 如果存在 vision_backbone 权重，则也进行加载
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # 冻结权重（如果需要）
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # 初始化 MemoryVLA 实例
        memory_vla = MemoryVLA(vlm,
                        token_size = vlm.llm_backbone.llm.lm_head.in_features,
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        action_model_type = action_model_type,
                        use_ema = use_ema,
                        norm_stats = norm_stats,
                        **kwargs,
                        )

        # 从检查点加载 ActionModel 权重
        if "action_model" in model_state_dict:
            memory_vla.action_model.load_state_dict(model_state_dict["action_model"], strict=False)
            # 如果使用了EMA且检查点中有ema_diffusion，则加载EMA权重
            assert use_ema is False, "Does not support using EMA weights from pretrained checkpoint."
            if "ema_diffusion" in model_state_dict and use_ema:
                memory_vla.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
            elif use_ema:
                memory_vla.ema_diffusion.load_state_dict(model_state_dict["action_model"])
        else:
            # 如果检查点中没有ActionModel，则发出警告并初始化一个新的
            overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")

        # 加载其他模块的权重
        for key, sub_state in model_state_dict.items():
            if key not in {"projector", "llm_backbone", "vision_backbone",
                           "action_model", "ema_diffusion"}:
                module = getattr(memory_vla, key, None)
                module.load_state_dict(sub_state, strict=True)

        # 清理内存
        del model_state_dict
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return memory_vla

    @torch.inference_mode()
    def predict_action(
        self, 
        image: Image,                   # 输入的图像，PIL.Image类型
        instruction: str,               # 任务指令字符串
        unnorm_key: Optional[str] = None,  # 用于获取反归一化统计信息的数据集名称
        cfg_scale: float = 1.5,         # 分类器自由引导(CFG)的缩放因子
        use_ddim: bool = False,         # 是否使用DDIM采样替代DDPM采样
        num_ddim_steps: int = 10,       # DDIM采样的步数
        episode_first_frame: str = 'False',  # 是否为回合的第一帧
        **kwargs: str                   # 其他传递给生成函数的关键字参数
    ) -> np.ndarray:
        """
        VLA推理的核心函数；将输入图像和任务指令映射到连续动作。
        
        参数:
            image: PIL图像，格式为[高度, 宽度, 3]
            instruction: 任务指令字符串
            unnorm_key: 可选的数据集名称，用于获取反归一化统计信息；
                       如果为None，则检查模型是否仅在一个数据集上训练，并获取相应统计信息
            cfg_scale: 分类器自由引导(CFG)的缩放因子；如果等于1.0，则禁用CFG
            use_ddim: 使用DDIM采样替代DDPM采样
            num_ddim_steps: 采样时使用的DDIM步数
            episode_first_frame: 是否为回合第一帧的标志
            
        返回:
            未归一化的(连续)动作向量 --> 末端执行器增量
        """
        # 获取图像预处理函数和分词器
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer

        # 构建VLA提示
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # 对提示文本进行分词处理
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # 为Llama分词器添加特殊token
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
            )
        else:
            raise ValueError(f"Unsupported [tokenizer](file://d:\doctor-code\code\MemoryVLA-openvla-codebase\MemoryVLA-openvla-codebase\prismatic\models\backbones\llm\base_llm.py#L0-L0) type = {type(tokenizer)}")

        model_dtype = next(self.parameters()).dtype

        # 预处理图像
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device, dtype=model_dtype)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vlm.device, dtype=model_dtype) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # 设置自动混合精度类型
        autocast_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float32

        # 使用自动混合精度进行推理
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(autocast_dtype == torch.bfloat16)):
            # 生成模型输出
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # 形状: [1, seq]
                pixel_values=pixel_values,                      # 形状: [1, 3, res, res] 或 Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                **kwargs,
            )

        # 获取模型数据类型
        model_dtype = next(self.action_model.net.parameters()).dtype
        # 提取最后一个时间步的认知token
        cog_tokens = output.hidden_states[-1][-1][:,-1,:]
        assert (cog_tokens.shape[0], cog_tokens.shape[1]) == (1,4096), "批量大小必须为1以进行动作预测"

        # 调整认知token的形状和数据类型
        cog_tokens = cog_tokens.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # 获取视觉特征并进行压缩处理
        vision_feats = self.vlm.vision_feats
        per_tokens = self.per_compr(vision_feats)

        # 检查episode_first_frame参数的有效性
        assert episode_first_frame in ['True', 'False'], "episode_first_frame必须是'True'或'False'"
        if episode_first_frame == 'True':
            # 如果是回合第一帧，重置记忆库
            print(" ** 重置记忆 ** ")
            self.cog_mem_bank.reset()
            self.per_mem_bank.reset()
            self.cur_timestep = 0

        # 设置当前时间步和回合ID
        episode_ids = [0]
        timesteps = [torch.tensor(self.cur_timestep, device=cog_tokens.device)]
        self.cur_timestep += 1

        # 使用认知和感知记忆库处理token
        cog_tokens = self.cog_mem_bank.process_batch(
            tokens=cog_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )

        per_tokens = self.per_mem_bank.process_batch(
            tokens=per_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )

        # 采样随机噪声
        B = cog_tokens.shape[0]
        noise = torch.randn(
            B, 
            self.future_action_window_size+1, 
            self.action_model.in_channels, 
            device=cog_tokens.device
        ).to(model_dtype)  # [B, T, D]
    
        # 设置分类器自由引导:
        using_cfg = cfg_scale > 1.0
        if using_cfg:
            # 如果使用CFG，复制噪声和条件
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  # [k, D]
            uncondition = uncondition.expand(B, *uncondition.shape[1:])  # [B, k, D]
            z = torch.cat([cog_tokens, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
            # 为无条件和有条件样本重复感知token
            model_kwargs.update({'per_token': per_tokens.repeat(2, 1, 1)})
        else:
            # 不使用CFG的情况
            model_kwargs = dict(z=cog_tokens)
            sample_fn = self.action_model.net.forward
            model_kwargs.update({'per_token': per_tokens})

        # DDIM采样
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                sample_fn, 
                noise.shape, 
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=cog_tokens.device,
                eta=0.0
            )
        else:
            # DDPM采样
            samples = self.action_model.diffusion.p_sample_loop(
                sample_fn, 
                noise.shape, 
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=cog_tokens.device
            )
        
        # 如果使用CFG，移除无类别样本
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)
        normalized_actions = samples[0].cpu().numpy()

        # 反归一化动作
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        # 裁剪归一化动作到[-1, 1]范围
        normalized_actions = np.clip(normalized_actions, -1, 1)
        # 处理夹爪开合动作（第7维）
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        # 根据mask进行反归一化
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions


    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        检查并验证用于动作反归一化的数据集键名
        
        当模型在多个数据集上训练时，需要指定特定的数据集统计信息用于动作反归一化。
        此方法确保提供了有效的键名，或者在只有一个数据集时自动选择。
        
        Args:
            norm_stats (dict): 包含各数据集归一化统计信息的字典
            unnorm_key (str, optional): 指定用于反归一化的数据集键名
            
        Returns:
            str: 验证后的数据集键名
            
        Raises:
            AssertionError: 当存在多个数据集但未指定键名，或指定的键名不存在时抛出
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """
        获取策略动作空间的维度
        
        根据指定数据集的归一化统计信息，返回动作空间的维度大小。
        
        Args:
            unnorm_key (str, optional): 指定用于获取动作维度的数据集键名
            
        Returns:
            int: 动作空间的维度大小
        """
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """
        获取策略动作空间的统计信息
        
        根据指定数据集的归一化统计信息，返回动作空间的完整统计信息。
        
        Args:
            unnorm_key (str, optional): 指定用于获取动作统计信息的数据集键名
            
        Returns:
            dict: 动作空间的统计信息字典
        """
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
