from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling

from training.metrics import Metrics, VLAMetrics
from vla import MemoryVLA

# LLaMA EOS Token
EOS_TOKEN = 2


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: Union[PrismaticVLM, MemoryVLA],
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        repeated_diffusion_steps: int = 4,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype
        self.repeated_diffusion_steps = repeated_diffusion_steps

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def load_optimizer_and_scheduler(self, checkpoint_path: str) -> None: ...
    
    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    # FIXME:=== VLA Training ===
    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
        action_model: bool = True,
        repeated_diffusion_steps: int = 4,
    ) -> None:
        """
        运行VLA（Vision-Language-Action）模型的训练循环。
        参数：
            vla_dataset (IterableDataset): 用于训练的数据集，必须是可迭代数据集。
            collator (PaddedCollatorForActionPrediction): 数据批处理时使用的collator对象，负责将样本组织成批次。
            metrics (VLAMetrics): 用于记录损失、评估指标等信息的对象。
            save_interval (int, 可选): 每隔多少个step保存一次checkpoint，默认为2500。
            save_full_model (bool, 可选): 是否保存完整的模型权重，默认为True。
            action_model (bool, 可选): 标志是否使用动作预测模块，默认为True。
            repeated_diffusion_steps (int, 可选): 扩散过程重复步数，默认为4。

        返回值：
            None
        """
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        #assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # 创建DataLoader，设置num_workers为0，因为RLDS加载器已内置并行机制
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === 开始训练阶段 ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * (len(dataloader) // self.grad_accumulation_steps)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # 清除梯度缓存（预防性措施）
            if self.vlm.use_ema is not None and self.vlm.use_ema == True:
                self.vlm.ema_diffusion.eval()
            self.optimizer.zero_grad()

            # [约定] DataLoader包装了RLDS Loader（通过.as_numpy_iterator()隐式调用.repeat()）
            # 因此遍历DataLoader本质上是无限循环（无需外层epoch循环），PyTorch语义略有不同，
            # 所以下面会根据已完成的梯度更新次数动态计算当前epoch。
            for train_idx, batch in enumerate(dataloader):
                # 注意：batch将在VLM.forward()中解包，并由AMP/FSDP自动处理设备移动与混合精度操作
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    assert action_model is not None
                    loss, output = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        actions=batch["actions"],
                        pixel_values=batch["pixel_values"],
                        action_masks=batch["action_masks"],
                        labels=batch["labels"],
                        timesteps=batch["timesteps"],
                        episode_ids=batch["episode_ids"],
                        output_hidden_states=True,
                        repeated_diffusion_steps=repeated_diffusion_steps,
                    )
                # 提交损失值并执行反向传播
                metrics.commit(loss=loss)
                
                normalized_loss = loss / self.grad_accumulation_steps
                normalized_loss.backward()

                # === 执行梯度更新步骤 ===
                # 当完成梯度累积后才进行优化器更新
                if (train_idx + 1) % self.grad_accumulation_steps == 0:
                    # 裁剪梯度，防止爆炸；该方法因DDP/FSDP策略而异
                    self.clip_grad_norm()

                    # 更新优化器和学习率调度器
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    if self.vlm.use_ema is not None and self.vlm.use_ema == True:
                        update_ema(self.vlm.ema_diffusion, self.vlm.action_model)
                    self.optimizer.zero_grad()
                    
                    # 动态计算当前epoch（基于全局step数）
                    epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                    # 推送统计信息到metrics对象
                    metrics.commit(update_step_time=True, global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                    status = metrics.push()

                    # 判断是否需要保存检查点或终止训练
                    if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                        (metrics.global_step % save_interval) == 0
                    ):
                        self.save_checkpoint(
                            metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                        )
                        dist.barrier()

                    if terminate:
                        return

                # 更新进度条显示状态
                progress.update()
                progress.set_description(status)
