import torch
from torch import nn

from action_model.models import DiT
from action_model import create_diffusion
from . import gaussian_diffusion as gd

# 创建不同尺寸的DiT模型工厂函数
def DiT_S(**kwargs):
    """
    创建小型DiT模型
    参数:
        **kwargs: 传递给DiT构造函数的其他参数
    返回:
        DiT: 深度为6，隐藏层大小为384，头数为4的DiT模型实例
    """
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)

def DiT_B(**kwargs):
    """
    创建基础型DiT模型
    参数:
        **kwargs: 传递给DiT构造函数的其他参数
    返回:
        DiT: 深度为12，隐藏层大小为768，头数为12的DiT模型实例
    """
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_L(**kwargs):
    """
    创建大型DiT模型
    参数:
        **kwargs: 传递给DiT构造函数的其他参数
    返回:
        DiT: 深度为24，隐藏层大小为1024，头数为16的DiT模型实例
    """
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

# DiT模型类型字典映射
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L}

class ActionModel(nn.Module):
    """
    动作模型类，基于扩散模型的动作预测器
    
    该类实现了一个基于DiT(Diffusion Transformer)的动作预测模型，使用扩散过程来生成连续的动作序列。
    主要功能包括噪声调度、前向扩散采样、反向去噪预测以及损失计算。
    """
    def __init__(self, 
                 token_size,                    # 条件token的维度大小
                 model_type,                    # 模型类型，如'DiT-S', 'DiT-B', 'DiT-L'
                 in_channels,                   # 输入动作的通道数/维度
                 future_action_window_size,     # 预测未来动作的时间窗口大小
                 diffusion_steps=100,           # 扩散过程的总时间步数，默认100
                 noise_schedule='squaredcos_cap_v2',  # 噪声调度策略，默认使用平方余弦
                 use_per_attn=False,            # 是否使用感知注意力机制
                 per_token_size=None,           # 感知token的维度大小
                 ):
        """
        初始化动作模型
        
        参数:
            token_size (int): 条件token的维度大小，通常来自语言或认知特征
            model_type (str): 模型类型，可选'DiT-S', 'DiT-B', 'DiT-L'
            in_channels (int): 输入动作的维度，例如7维机器人关节动作
            future_action_window_size (int): 需要预测的未来动作序列长度
            diffusion_steps (int): 扩散过程的总时间步数
            noise_schedule (str): 噪声调度策略名称
            use_per_attn (bool): 是否启用感知注意力机制
            per_token_size (int): 感知token的维度大小，当use_per_attn为True时必需
        """
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion提供了前向和反向函数q_sample和p_sample
        self.diffusion_steps = diffusion_steps
        # 创建标准扩散过程对象
        self.diffusion = create_diffusion(timestep_respacing="", 
                                        noise_schedule=noise_schedule, 
                                        diffusion_steps=self.diffusion_steps, 
                                        sigma_small=True, 
                                        learn_sigma=False)
        self.ddim_diffusion = None  # DDIM采样器，初始为空
        
        # 根据模型方差类型决定是否学习sigma参数
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.future_action_window_size = future_action_window_size

        # 根据模型类型创建相应的DiT网络
        self.net = DiT_models[model_type](
            token_size=token_size,
            in_channels=in_channels,
            class_dropout_prob=0.1,         # 类别条件dropout概率
            learn_sigma=learn_sigma,        # 是否学习sigma参数
            future_action_window_size=future_action_window_size,
            use_per_attn=use_per_attn,      # 是否使用感知注意力
            per_token_size=per_token_size,  # 感知token维度
        )

    def loss(self, x, z, per_token):
        """
        计算给定条件z和真实动作x的损失
        参数:
            x (torch.Tensor): actions_repeated生成目标（被加噪的动作序列），形状为[B, T, C]
            z (torch.Tensor): 全局条件（cognitive token），形状为[B, 1, D]， 任务级、语义级、策略级信息
            per_token (torch.Tensor): 局部条件（perception token），形状为[B, N, D_per]  ，当前传感器/视觉的状态 
        返回:
            torch.Tensor: 平均L2损失值 
            最终模型学会从随机噪声中“反向生成动作序列”
        """
        # 采样随机噪声和时间步
        noise = torch.randn_like(x)  # [B, T, C] 与真实动作同形状的高斯噪声
        # 在批次维度上为每个样本生成随机时间步
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)
        # 从真实动作x和噪声采样得到x_t（前向扩散过程）
        x_t = self.diffusion.q_sample(x, timestep, noise)
        # 通过网络预测噪声
        noise_pred = self.net(x_t, timestep, z, per_token=per_token)
        # 确保预测噪声、真实噪声和输入动作形状一致
        assert noise_pred.shape == noise.shape == x.shape
        # 计算L2损失（均方误差）
        loss = ((noise_pred - noise) ** 2).mean()
        # 可选：增加变分下界损失 loss += loss_vlb
        return loss

    def create_ddim(self, ddim_step=10):
        """
        创建DDIM采样器
        
        DDIM(Denoising Diffusion Implicit Models)是一种加速采样的方法，
        相比于标准的DDPM采样，可以用更少的步骤获得相似的质量。
        参数:
            ddim_step (int): DDIM采样的步数，默认为10  
        返回:
            gaussian_diffusion.Diffusion: 创建的DDIM扩散对象
        """
        self.ddim_diffusion = create_diffusion(timestep_respacing="ddim"+str(ddim_step), 
                                               noise_schedule=self.noise_schedule,
                                               diffusion_steps=self.diffusion_steps, 
                                               sigma_small=True, 
                                               learn_sigma=False)
        return self.ddim_diffusion