"""
materialize.py

Factory class for initializing Vision Backbones, LLM Backbones, and VLMs from a set registry; provides and exports
individual functions for clear control flow.
"""

from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm import LLaMa2LLMBackbone, LLMBackbone, MistralLLMBackbone, PhiLLMBackbone
from prismatic.models.backbones.vision import (
    CLIPViTBackbone,
    DinoCLIPViTBackbone,
    DinoSigLIPViTBackbone,
    DinoV2ViTBackbone,
    ImageTransform,
    IN1KViTBackbone,
    SigLIPViTBackbone,
    VisionBackbone,
)
from prismatic.models.vlms import PrismaticVLM

# === Registries =>> Maps ID --> {cls(), kwargs} :: Different Registries for Vision Backbones, LLM Backbones, VLMs ===
# fmt: off

# === Vision Backbone Registry ===
VISION_BACKBONES = {
    # === 224px Backbones ===
    "clip-vit-l": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-so400m": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinov2-vit-l": {"cls": DinoV2ViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1k-vit-l": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinosiglip-vit-so-224px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 224}},

    # === Assorted CLIP Backbones ===
    "clip-vit-b": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "clip-vit-l-336px": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 336}},

    # === Assorted SigLIP Backbones ===
    "siglip-vit-b16-224px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-256px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 256}},
    "siglip-vit-b16-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
    "siglip-vit-so400m-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},

    # === Fused Backbones ===
    "dinoclip-vit-l-336px": {"cls": DinoCLIPViTBackbone, "kwargs": {"default_image_size": 336}},
    "dinosiglip-vit-so-384px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
}


# === Language Model Registry ===
LLM_BACKBONES = {
    # === LLaMa-2 Pure (Non-Chat) Backbones ===
    "llama2-7b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === LLaMa-2 Chat Backbones ===
    "llama2-7b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Vicuna-v1.5 Backbones ===
    "vicuna-v15-7b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "vicuna-v15-13b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Mistral v0.1 Backbones ===
    "mistral-v0.1-7b-pure": {"cls": MistralLLMBackbone, "kwargs": {}},
    "mistral-v0.1-7b-instruct": {"cls": MistralLLMBackbone, "kwargs": {}},

    # === Phi-2 Backbone ===
    "phi-2-3b": {"cls": PhiLLMBackbone, "kwargs": {}},
}

# fmt: on


def get_vision_backbone_and_transform(
    vision_backbone_id: str, 
    image_resize_strategy: str,
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform.
    Args:
        vision_backbone_id: 视觉主干网络的标识符字符串
        image_resize_strategy: 图像调整大小策略字符串
    Returns:
        包含视觉主干网络实例和图像变换实例的元组
    Raises:
        ValueError: 当提供的vision_backbone_id不被支持时抛出异常
    """
    # 检查请求的视觉主干网络是否在支持的配置中
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg["cls"](
            vision_backbone_id, image_resize_strategy, **vision_cfg["kwargs"]
        )
        image_transform = vision_backbone.get_image_transform()
        return vision_backbone, image_transform

    else:
        raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    """
    获取指定ID的LLM骨干网络和对应的分词器
    Args:
        llm_backbone_id (str): LLM骨干网络的标识符
        llm_max_length (int, optional): LLM的最大序列长度，默认为2048
        hf_token (Optional[str], optional): HuggingFace访问令牌，用于访问私有模型，默认为None
        inference_mode (bool, optional): 是否处于推理模式，默认为False
    Returns:
        Tuple[LLMBackbone, PreTrainedTokenizerBase]: 返回LLM骨干网络实例和预训练分词器的元组
    Raises:
        ValueError: 当提供的llm_backbone_id不被支持时抛出异常
    """
    # 检查请求的LLM骨干网络是否在支持列表中
    if llm_backbone_id in LLM_BACKBONES:
        llm_cfg = LLM_BACKBONES[llm_backbone_id]
        # 实例化对应的LLM骨干网络
        llm_backbone: LLMBackbone = llm_cfg["cls"](
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            **llm_cfg["kwargs"],
        )
        # 获取与骨干网络配套的分词器
        tokenizer = llm_backbone.get_tokenizer()
        return llm_backbone, tokenizer

    else:
        raise ValueError(f"LLM Backbone `{llm_backbone_id}` is not supported!")
    
def get_vlm(
    model_id: str,
    arch_specifier: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    enable_mixed_precision_training: bool = True,
) -> PrismaticVLM:
    """
    轻量级包装函数，用于初始化视觉-语言模型(VLM)，主要用于未来扩展性考虑。
    参数:
        model_id: 模型标识符字符串，用于唯一标识模型实例
        arch_specifier: 架构描述符，指定模型的具体架构配置
        vision_backbone: 视觉骨干网络对象，处理图像输入
        llm_backbone: 语言模型骨干网络对象，处理文本输入
        enable_mixed_precision_training: 是否启用混合精度训练，默认为True
    返回:
        PrismaticVLM: 初始化完成的视觉-语言模型实例
    """
    return PrismaticVLM(
        model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
    )
