"""
nn_utils.py

实用函数和 PyTorch 子模块定义。
"""

import torch
import torch.nn as nn


# === 各种投影模块的定义，签名格式 :: [..., in_dim] --> [..., out_dim] ===

class LinearProjector(nn.Module):
    """
    线性投影器类，用于将视觉特征映射到语言模型维度空间。
    """
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        """
        初始化线性投影器。

        Args:
            vision_dim: 视觉特征的维度
            llm_dim: 语言模型的维度
        """
        super().__init__()
        # 定义线性投影层，带偏置项
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，执行线性投影操作。

        Args:
            img_patches: 图像块特征张量，形状为 [..., vision_dim]

        Returns:
            投影后的特征张量，形状为 [..., llm_dim]
        """
        return self.projector(img_patches)


class MLPProjector(nn.Module):
    """
    多层感知机(MPL)投影器类，使用GELU激活函数。
    """
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        """
        初始化MLP投影器。

        Args:
            vision_dim: 视觉特征的维度
            llm_dim: 语言模型的维度
            mlp_type: MLP类型，默认为"gelu-mlp"
        """
        super().__init__()
        if mlp_type == "gelu-mlp":
            # 定义两层MLP结构：线性层->GELU激活->线性层
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),  # 第一层线性变换
                nn.GELU(),                                  # GELU激活函数
                nn.Linear(llm_dim, llm_dim, bias=True),     # 第二层线性变换
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，执行MLP投影操作。

        Args:
            img_patches: 图像块特征张量，形状为 [..., vision_dim]

        Returns:
            投影后的特征张量，形状为 [..., llm_dim]
        """
        return self.projector(img_patches)


class FusedMLPProjector(nn.Module):
    """
    融合MLP投影器类，具有更复杂的网络结构。
    """
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp") -> None:
        """
        初始化融合MLP投影器。

        Args:
            fused_vision_dim: 融合视觉特征的维度
            llm_dim: 语言模型的维度
            mlp_type: MLP类型，默认为"fused-gelu-mlp"
        """
        super().__init__()
        # 设置初始投影维度为视觉维度的4倍
        self.initial_projection_dim = fused_vision_dim * 4
        
        if mlp_type == "fused-gelu-mlp":
            # 定义三层MLP结构：线性层->GELU->线性层->GELU->线性层
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),  # 第一层扩展维度
                nn.GELU(),                                                            # GELU激活函数
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),           # 第二层降维到LLM维度
                nn.GELU(),                                                            # GELU激活函数
                nn.Linear(llm_dim, llm_dim, bias=True),                               # 第三层保持维度
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，执行融合MLP投影操作。

        Args:
            fused_img_patches: 融合图像块特征张量，形状为 [..., fused_vision_dim]

        Returns:
            投影后的特征张量，形状为 [..., llm_dim]
        """
        return self.projector(fused_img_patches)