# 导入标准库
import json
import os
from pathlib import Path
from typing import List, Optional, Union
from huggingface_hub import HfFileSystem, hf_hub_download

# 导入项目依赖
from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch

# 导入自定义模块
from vla import MemoryVLA

# 初始化日志记录器
overwatch = initialize_overwatch(__name__)

# 定义HuggingFace仓库地址
HF_HUB_REPO = "TRI-ML/prismatic-vlms"

# === 可用模型相关函数 ===

def available_models() -> List[str]:
    """获取所有注册的模型ID列表"""
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    """获取所有全局注册的模型名称列表"""
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    """获取指定模型的描述信息
    
    Args:
        model_id_or_name: 模型ID或名称
        
    Returns:
        模型描述字符串
        
    Raises:
        ValueError: 当找不到指定模型时抛出异常
    """
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # 打印并返回模型描述
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model 加载预训练模型=== 
def load(
    model_id_or_path: Union[str, Path],   # 加载的模型ID或本地路径siglip-224px+7b
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,    # 是否用于训练模式加载，True 表示模型权重不会被冻结，允许进行梯度更新
) -> PrismaticVLM:
    """从本地磁盘或HuggingFace Hub加载预训练的PrismaticVLM模型"""
    
    # 判断输入是本地目录还是模型ID
    if os.path.isdir(model_id_or_path):
        # 处理本地路径情况
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # 获取配置文件和检查点路径
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        # 处理从HuggingFace Hub下载情况
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
            )

    # 从config.json加载模型配置
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # 显示找到的配置信息
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # 加载视觉骨干网络
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # FIXME：覆盖 hf 模型 ID，改成本地路径（硬编码修复）
    model_cfg["llm_backbone_id"] = "/home/guojiahui/MemoryVLA/models/Llama-2-7b-hf"

    # 加载LLM骨干网络
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,  # 根据是否训练决定推理模式
    )

    # 使用预训练检查点创建VLM实例
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,  # 根据训练标志决定是否冻结权重
    )

    return vlm

# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    **kwargs,
) -> MemoryVLA:
    """从本地磁盘或HuggingFace Hub加载预训练的MemoryVLA模型"""

    # TODO (siddk, moojink) :: 统一与上面load()函数的语义；目前load_vla()假设路径指向检查点.pt文件而非顶层运行目录！
    
    # 判断输入是本地文件还是模型ID
    if os.path.isfile(model_id_or_path):
        # 处理本地检查点文件情况
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # 验证检查点路径格式是否正确
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # 获取配置文件和数据集统计信息路径
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    else:
        # 处理从HuggingFace Hub搜索情况
        overwatch.info(f"Checking HF for `{(hf_path := str(Path(model_id_or_path)))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        # 查找有效的检查点文件
        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/*.pt")
        if (len(valid_ckpts) == 0) or (len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")

        target_ckpt = Path(valid_ckpts[-1]).name
        model_id_or_path = str(model_id_or_path)  # 转换为字符串供HF Hub API使用
        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            # 下载配置文件、数据集统计信息和检查点
            config_json = hf_hub_download(
                repo_id=model_id_or_path, filename=f"{('config.json')!s}", cache_dir=cache_dir
            )
            dataset_statistics_json = hf_hub_download(
                repo_id=model_id_or_path, filename=f"{('dataset_statistics.json')!s}", cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=model_id_or_path, filename=f"{(Path('checkpoints') / target_ckpt)!s}", cache_dir=cache_dir
            )

    # 从config.json加载VLA配置和基础VLM模型配置
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()

    # 加载数据集统计信息用于动作反归一化
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    # 显示找到的配置信息
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # 加载视觉骨干网络
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
    )

    # 加载LLM骨干网络
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )

    # 使用预训练检查点创建VLA实例
    overwatch.info(f"Loading VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint")
    vla = MemoryVLA.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        norm_stats=norm_stats,  # 传递归一化统计信息
        image_resize_strategy=model_cfg.image_resize_strategy,
        **kwargs,
    )

    return vla