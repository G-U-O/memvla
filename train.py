import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
import yaml
import draccus

import torch
import torch.distributed as dist

from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
# 导入训练相关模块
from training import VLAMetrics, get_train_strategy
from conf.vla import VLAConfig, VLARegistry
from vla import load, load_vla
from vla import MemoryVLA
from vla import get_vla_dataset_and_collator
from vla.datasets.rlds.utils.data_utils import save_dataset_statistics

def print_model_memory_report(model):
    """打印模型各模块的参数量和显存占用（静态权重）"""
    print("\n" + "="*80)
    print(f"{'Module Name':<30} | {'Memory (GB)':<12} | {'Params (M)':<12} | {'Trainable (M)':<12}")
    print("-" * 80)
    
    total_bytes = 0
    total_trainable_bytes = 0
    
    # 遍历模型的一级子模块 (例如 global_adapter, paligemma_with_expert)
    for name, child in model.named_children():
        param_count = 0
        param_bytes = 0
        trainable_count = 0
        trainable_bytes = 0
        
        for p in child.parameters():
            size = p.numel() * p.element_size() # 元素数量 * 每个字节大小 (BF16=2, FP32=4)
            param_count += p.numel()
            param_bytes += size
            
            if p.requires_grad:
                trainable_count += p.numel()
                trainable_bytes += size
        
        total_bytes += param_bytes
        total_trainable_bytes += trainable_bytes
        
        print(f"{name:<30} | {param_bytes/1e9:.4f} GB   | {param_count/1e6:.2f} M      | {trainable_count/1e6:.2f} M")
        
        # 特别展开查看巨大的 paligemma_with_expert 内部
        if name == "paligemma_with_expert":
            for sub_name, sub_child in child.named_children():
                sub_bytes = sum(p.numel() * p.element_size() for p in sub_child.parameters())
                print(f"  └─ {sub_name:<25} | {sub_bytes/1e9:.4f} GB")

    print("-" * 80)
    print(f"Total Static Weights: {total_bytes/1e9:.4f} GB")
    
    # 估算优化器状态占用 (AdamW 需要存2份状态，通常是 FP32，所以是参数量的 8 倍)
    # 注意：这只是估算，实际可能更大
    optimizer_overhead = total_trainable_bytes * 4 # 假设优化器状态是 FP32 (4 bytes) * 2 (momentum, variance) = 8 bytes per param. 
    # 但原始权重如果是 BF16 (2 bytes), 优化器状态就是权重的 4 倍。
    
    print(f"Est. Optimizer State (AdamW): {optimizer_overhead/1e9:.4f} GB (如果你没冻结，这会非常大！)")
    print("="*80 + "\n")

# Sane Defaults # 设置环境变量，禁用tokenizer的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger` 初始化日志系统 =>> 封装了 `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    # VLA配置 (`conf/vla.py`); 可通过 --vla.type `VLARegistry.<VLA>.vla_id` 覆盖
    # vla_id: "prism-dinosiglip-224px+oxe+diffusion"
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.EXP_COGACT_OXE_MAGIC_SOUP_PLUS_MINUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path("") # Path to dataset directory
    run_root_dir: Path = Path("runs") # Path to directory to store logs & checkpoints

    # Resume Run Parameters 恢复运行参数
    pretrained_checkpoint: Optional[Union[str, Path]] = None # 检查点的绝对路径
    is_resume: bool = True # 是否继续之前的训练运行，仅在提供预训练检查点时适用
    resume_step: Optional[int] = None # 恢复的全局步骤（应与检查点匹配）
    resume_epoch: Optional[int] = None # 恢复的轮次（应与检查点匹配）

    # 运行参数
    run_id: Optional[str] = None # 用于日志记录、Weights & Biases 的运行ID
    run_id_note: Optional[str] = None # 额外的日志记录说明
    save_interval: int = 2500 # 保存检查点的间隔（以步数计）
    image_aug: bool = True # 是否启用图像增强
    seed: int = 42 # 随机种子（用于可重现性）

    # HF Hub 凭据（用于任何需要授权的模型）
    hf_token: Union[str, Path] = Path(".hf_token") # HF Token的环境变量或路径

    # 跟踪参数
    trackers: Tuple[str, ...] = ("jsonl", "wandb") # 要初始化的跟踪器（如果使用W&B，请添加配置！）
    wandb_project: str = "" # 要记录到的W&B项目的名称（使用默认值！）
    wandb_entity: str = "" # 要记录的实体名称

    # 模型参数
    repeated_diffusion_steps: int = 4 # 训练动作模型（扩散模型）的重复步骤
    load_all_data_for_training: bool = True # 加载所有训练数据
    future_action_window_size: int = 15 # 动作分块，预测未来动作+当前动作
    action_model_type: str = 'DiT-L' # 动作模型类型，可选 ['DiT-S', 'DiT-B', 'DiT-L']
    use_ema: bool = False # 动作模型的EMA版本
    action_dim: int = 7 # 动作空间的维度
    dataloader_type: str = "group" # 数据加载器类型，可选 ['group', 'stream', 'parallel_stream']
    group_size: int = 16 # 'group' 数据加载器的组大小
    per_token_size: int = 256 # 感知压缩的令牌大小
    mem_length: int = 16 # 记忆长度
    retrieval_layers: int = 2 # 记忆检索的层数
    use_timestep_pe: bool = True # 是否使用时间步位置编码
    fusion_type: str = 'gate' # 记忆融合类型，可选 ['gate', 'add']
    consolidate_type: str = 'tome' # 记忆合并类型，可选 ['fifo', 'tome']
    update_fused: bool = False # 是否更新融合的记忆

    def __post_init__(self) -> None:
        """从 `self.vla` 提取优化参数以便使用 =>> 验证 `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`验证 `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("MemoryVLA Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    # 注意 =>> 在 `torchrun` 下，初始化 [overwatch] 会自动设置 `torch.distributed`    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()
    #  --vla.type prism-dinosiglip-224px+oxe+diffusion \
    #  --vla.data_mix ${data_mix} \
    #  --vla.expected_world_size ${n_gpu} \
    #  --vla.per_device_batch_size ${bs} \
    #  --vla.global_batch_size $((n_gpu * bs)) \
    #  --vla.learning_rate 2e-5 \
    #  --vla.max_steps 20000 \
    #  --vla.shuffle_buffer_size ${shuffle_buffer_size} \

    # Configure Unique Run Name & Save Directory 配置唯一的运行名称和保存目录
    vla_id = cfg.vla.vla_id
    # run_id='memvla_libero_goal'
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # 开始 =>> 创建目录并设置随机性
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    # hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)  # 使用海象运算符(:=)创建并赋值 run_dir 变量
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # 保存配置 =>> 额外保存JSON版本以供后续HF集成
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
    
    dist.barrier()
    # 加载VLA检查点（如果从训练中恢复）或基础VLM（从 `cfg.vla.base_vlm` ID或路径）
    #   =>> 注意 :: 验证所有参数都以FP32格式加载！
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:  # 恢复训练模式
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> 注意 :: 我们要求开发者传入 `resume_*` 参数作为额外的健全性检查！
        if cfg.is_resume:  # 提取步数和轮次信息
            step_match = re.search(r"step-(\d+)-", cfg.pretrained_checkpoint)
            epoch_match = re.search(r"epoch-(\d+)-", cfg.pretrained_checkpoint)

            if step_match and epoch_match:
                step = int(step_match.group(1)) # 移除前导零
                epoch = int(epoch_match.group(1))  # 移除前导零
                assert step == cfg.resume_step, f"Mismatch in step: {step} != {cfg.resume_step}"
                assert epoch == cfg.resume_epoch, f"Mismatch in epoch: {epoch} != {cfg.resume_epoch}"
            else:
                raise ValueError(f"Checkpoint filename format incorrect: {cfg.pretrained_checkpoint}")

        overwatch.info("Loading VLA Checkpoint")
        if cfg.use_ema:
            overwatch.info("Loading EMA of Diffusion")
        kwargs = vars(cfg)
        model_id_or_path = kwargs.pop("pretrained_checkpoint")
        vla = load_vla(model_id_or_path=model_id_or_path, load_for_training=True, **kwargs)

    else:  # 从头开始创建模型。
        vlm = load(model_id_or_path=cfg.vla.base_vlm, hf_token=cfg.hf_token, load_for_training=True)
        overwatch.info("Creating VLA from Base VLM")
        if cfg.use_ema:
            overwatch.info("Creating EMA for Diffusion")
        vla = MemoryVLA(vlm=vlm, **vars(cfg))  # 基础VLM包装成完整的VLA（视觉-语言-动作）模型
        del vlm   # 基础VLM引用以节省内存
        overwatch.info("DO NOT specify image_resize_strategy, use resize-naive")

    # [Validate] 模型应该处于全精度状态！
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # 根据冻结与未冻结参数确定训练"阶段" --> 支持不同的微调方案！
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "full-finetune"  # Full fine-tuning 完全微调
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "finetune"  # Frozen vision encoder 冻结视觉编码器
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        stage = "align"  # Fine-tuning projector 微调投影器
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone and cfg.vla.unfreeze_last_llm_layer:
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer 微调视觉编码器、投影器和LLM最后一层
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone and cfg.vla.unfreeze_last_llm_layer:
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only 仅微调LLM最后一层
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )
    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vla.freeze_backbones(stage)

    # Print number of total/trainable model parameters 打印总参数数/可训练参数数
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )
    # TODO:打印模型大小
    print_model_memory_report(vla)
    
    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    # FIXME: 1.数据加载
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        data_root_dir=cfg.data_root_dir,
        data_mix=cfg.vla.data_mix,
        image_transform=vla.vision_backbone.get_image_transform(),
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=cfg.future_action_window_size,
        dataloader_type=cfg.dataloader_type,
        group_size=cfg.group_size,
    )

    # 保存数据集统计信息以便在推理时进行反归一化
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
    
    dist.barrier()
    # 创建训练策略
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vla,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))
    if cfg.pretrained_checkpoint is not None and cfg.is_resume:
        train_strategy.load_optimizer_and_scheduler(cfg.pretrained_checkpoint)

    # 创建指标 =>> 处理实时跟踪，记录到指定的跟踪器（例如，JSONL，Weights & Biases）
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    #  TODO: 运行VLA训练
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        metrics,
        save_interval=cfg.save_interval,
        action_model=True,
        repeated_diffusion_steps=cfg.repeated_diffusion_steps,
    )

    # 最终化
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
