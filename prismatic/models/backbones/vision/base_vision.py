"""
base_vision.py

Abstract class definition of a Vision Backbone (Visual Featurizer), with full annotations of class methods, utility
functions, and initialization logic.

We also define the generic TimmViTBackbone class here, providing a default interface for loading any TIMM Vision
Transformer model for feature extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from PIL.Image import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# === Interface for an Image Transform  图像变换的接口===
class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...


# === Custom Torchvision Image Transforms 图像变换的接口 ===
@dataclass
class LetterboxPad:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")

 
# === Abstract Base Class for arbitrary Vision Backbones 任意视觉主干网络的抽象基类 ===
class VisionBackbone(nn.Module, ABC):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__()
        self.identifier: str = vision_backbone_id
        self.image_resize_strategy: str = image_resize_strategy
        self.default_image_size: int = default_image_size

        # Instance attributes for a Vision Backbone
        self.featurizer: nn.Module = None
        self.image_transform: ImageTransform = None

    def get_image_transform(self) -> ImageTransform:
        return self.image_transform

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the featurizer given a set of processed images, returning patch/grid features."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_image_resolution(self) -> Tuple[int, int, int]: ...

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_patches(self) -> int: ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...


# === Abstract Base Class for Arbitrary TIMM Vision Transformer Backbones ===
class TimmViTBackbone(VisionBackbone, ABC):
    def __init__(
        self,
        vision_backbone_id: str,
        timm_path_or_url: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        override_act_layer: Optional[str] = None,
    ) -> None:
        """
        初始化 VisionBackbone 类的一个实例，该类基于 TIMM 库中的 Vision Transformer 模型。

        参数:
            vision_backbone_id (str): 视觉主干网络的唯一标识符。
            timm_path_or_url (str): TIMM 模型名称或模型文件路径/URL，用于加载预训练模型。
            image_resize_strategy (str): 图像缩放策略，支持 "resize-naive", "resize-crop", "letterbox"。
            default_image_size (int, optional): 默认输入图像尺寸（宽高），默认为 224。
            override_act_layer (Optional[str], optional): 可选激活层类型，若提供则覆盖模型默认激活函数，默认为 None。

        返回:
            None
        """
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.timm_path_or_url = timm_path_or_url
        self.override_act_layer = override_act_layer
        self.dtype = torch.bfloat16

        # 初始化特征提取器（Vision Transformer），如果需要从 HF 或 TIMM Hub 下载模型
        if self.override_act_layer is None:
            self.featurizer: VisionTransformer = timm.create_model(
                self.timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
            )
        else:
            self.featurizer: VisionTransformer = timm.create_model(
                self.timm_path_or_url,
                pretrained=True,
                num_classes=0,
                img_size=self.default_image_size,
                act_layer=self.override_act_layer,
            )
        self.featurizer.eval()

        # 对特征提取器的 forward 方法进行猴子补丁以确保与 FSDP 兼容
        # 注意：默认设置 get_intermediate_layers 返回倒数第二层的 patch 特征
        # TODO(siddk): 在 PyTorch 相关问题解决后移除此逻辑
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )

        # 验证当前实现仅支持 TIMM Vision Transformer 类型模型
        assert isinstance(self.featurizer, VisionTransformer), (
            "Featurizer is not a TIMM VisionTransformer; if you would like to support a new visual representation, "
            "file an issue or implement the requisite logic (see `prismatic/models/backbones/vision/base_vision.py`)!"
        )

        # 获取并修正模型数据配置信息，默认图像大小需与设定一致
        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # 构建基础图像变换流程
        default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        # 处理 SigLIP 和 IN1K 模型默认变换中图像被放大后再裁剪的问题
        if "siglip" in self.timm_path_or_url or "in1k" in self.timm_path_or_url:
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)
            default_image_transform = Compose(
                [
                    Resize(self.default_image_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        # 根据指定的图像缩放策略构建最终图像变换管道
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose(
                [
                    Resize(target_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = default_image_transform

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert "mean" in self.data_cfg, "TIMM [data_cfg](file://d:\doctor-code\code\MemoryVLA-openvla-codebase\MemoryVLA-openvla-codebase\prismatic\models\backbones\vision\base_vision.py#L0-L0) missing image normalization mean!"

            # 计算填充颜色值（根据归一化均值反推）
            fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])

            # 构造新的图像变换流程，先进行 letterbox 填充再应用其他变换
            self.image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then the _entire_ featurizer."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Runs transformed image/pixel tensor through vision backbone, returning _all_ patch features."""
        return self.featurizer(pixel_values)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
