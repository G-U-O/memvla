"""
prismatic.py

PyTorch Module defining a PrismaticVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
PyTorch 模块定义了一个 PrismaticVLM，这是我们在工作中定义各种不同 VM 的通用接口。
注意事项：
- 目前，我们不子类化 “transformers. PretrainedModel”（或 CaisalLM）。
  相反，我们假设一个非常有限的子集允许分派到底层 LLM 的 “生成” 实用程序（馈送输入通过我们的自定义投影垫片）。
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class PrismaticVLM(VLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        **kwargs,
    ) -> None:
        """
        初始化 PrismaticVLM 实例。

        Args:
            model_id: 模型唯一标识符
            vision_backbone: 视觉主干网络，用于处理图像输入
            llm_backbone: 语言模型主干网络，用于处理文本输入
            enable_mixed_precision_training: 是否启用混合精度训练，默认为 True
            arch_specifier: 架构指定符，决定投影层类型，默认为 "gelu-mlp"
            **kwargs: 其他传递给父类的参数
        """
        # 调用父类 VLM 的初始化方法
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # 设置权重初始化种子以确保投影层的一致性
        torch.manual_seed(vision_backbone.embed_dim)

        # 根据架构指定符初始化投影层（适配器）
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            # 线性投影层
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            # 融合 GELU 激活函数的多层感知机投影层
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            # 带 GELU 激活函数的多层感知机投影层
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            # 不支持的架构指定符抛出异常
            raise ValueError(f"PrismaticVLM with `{arch_specifier = }` is not supported!")

        # 追踪视觉主干网络是否需要梯度更新
        self.vision_backbone_requires_grad = False

        # 设置模块键值，用于检查点保存和模型加载
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        self.trainable_module_keys = []

        # === 生成工具 ===
        # 用于计算似然值，获取 "True", "False", "Yes", "No" 等触发词对应的 token 索引
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            # 对触发字符串进行编码，不添加特殊token
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            # 确保每个触发字符串只对应一个token
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    @classmethod   # 装饰器会自动将类本身作为第一个参数传递给方法
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        **kwargs,
    ) -> PrismaticVLM:
        """
        从预训练检查点初始化一个PrismaticVLM模型，冻结所有权重，适用于推理任务。

        Args:
            pretrained_checkpoint: 预训练模型检查点文件的路径
            model_id: 模型的唯一标识符
            vision_backbone: 模型的视觉骨干网络组件
            llm_backbone: 模型的语言模型骨干网络组件
            enable_mixed_precision_training: 是否启用混合精度训练（默认：True）
            arch_specifier: 架构指定字符串（默认："gelu-mlp"）
            freeze_weights: 加载权重后是否冻结所有模型权重（默认：True）
            **kwargs: 传递给构造函数的其他关键字参数

        Returns:
            PrismaticVLM: 初始化完成的模型实例，已加载预训练权重
        """
        # 创建模型实例
        vlm = cls(    # PrismaticVLM()
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"
        # 加载各个组件的权重
        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        根据训练阶段冻结或解冻模型的不同组件。

        支持多个预训练/微调阶段，控制视觉主干（vision backbone）、语言模型主干（LLM backbone）和投影器（projector）
        的可训练性。在不同阶段中，部分模块会被设置为不可更新参数（requires_grad=False），而另一部分则保持可训练状态。

        :param stage: 当前的训练阶段，支持以下取值：
                        - "align": 冻结 vision_backbone 和 llm_backbone，仅训练 projector。
                        - "finetune" 或 "vla-train": 冻结 vision_backbone，训练 projector 和 llm_backbone。
                        - "full-finetune" 或 "vla-full-train": 所有模块均设为可训练。
                        - "last-layer-finetune" 或 "vla-last-layer-train": 仅解冻 LLM 最后一层，其余全部冻结。
                        - "vla-sandwich-train": 解冻 vision_backbone、projector 及 LLM 最后一层，其余冻结。
        :return: None
        """

        if stage == "align":
            # 在 align 阶段：只训练 projector，冻结 vision_backbone 和 llm_backbone
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # 更新可训练模块列表
            self.trainable_module_keys = ["projector"]

            # 更新跟踪标志
            self.vision_backbone_requires_grad = False

            # 明确记录各模块是否被冻结或可训练
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"finetune", "vla-train"}:
            # 在 finetune 阶段：训练 projector 和 llm_backbone，冻结 vision_backbone
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # 更新可训练模块列表
            self.trainable_module_keys = ["projector", "llm_backbone"]

            # 更新跟踪标志
            self.vision_backbone_requires_grad = False

            # 明确记录各模块的状态
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"full-finetune", "vla-full-train"}:
            # 全量微调阶段：所有模块都参与训练，并将 vision_backbone 转换为 float32 类型
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # 更新可训练模块列表
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # 更新跟踪标志
            self.vision_backbone_requires_grad = True

            # 记录当前各模块的训练状态
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"last-layer-finetune", "vla-last-layer-train"}:
            # 仅微调最后一层：冻结所有模块，但解冻 LLM 的最后几层特定模块
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)

            # 解冻 LLM 主干中的最后层模块
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # 更新可训练模块列表
            self.trainable_module_keys = ["llm_backbone"]

            # 更新跟踪标志
            self.vision_backbone_requires_grad = False

            # 输出详细日志信息
            # fmt: off
            overwatch.info(f"[Frozen]                    🥶   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        elif stage in {"vla-sandwich-train"}:
            # sandwich 微调策略：解冻 vision_backbone 和 projector，同时解冻 LLM 最后一层
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.llm_backbone.requires_grad_(False)

            # 解冻 LLM 主干中的最后层模块
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # 更新可训练模块列表
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # 更新跟踪标志
            self.vision_backbone_requires_grad = True

            # 输出详细日志信息
            # fmt: off
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        else:
            # 不支持的阶段抛出异常
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

        # 打印当前网络中所有可训练参数名称用于调试
        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"PrismaticVLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`

    # ruff: noqa: C901
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        proprio_feat: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Run Visual Feature Extraction
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])

        self.vision_feats = patch_features

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)

        if proprio_feat is not None:
            projected_patch_embeddings = torch.cat([projected_patch_embeddings, proprio_feat], dim=1)

        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids) # b,l,c

        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
            )

        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        llm_output = self.llm_backbone(
                input_ids=None,
                attention_mask=fused_attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=fused_embeddings,
                labels=fused_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return llm_output
    # === GenerationMixin Methods ===
    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

    #===生成混合方法===
    #=>注意：以下方法覆盖了'transformers. GenerationMisin'的功能；这些期望
    #        在每个函数签名中收缩，并且还期望我们的“转发”函数大致采取
    #        与底层LLM相同的参数（参见'LlamaModelForCausalLM'作为示例）

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        proprio_feat: Optional[torch.Tensor] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        为文本生成准备输入数据，主要处理缓存机制和多模态数据的整合。
        借鉴自 `LlamaForCausalLM`，主要用于在生成过程中处理缓存逻辑。
        Args:
            input_ids: 输入token的ID序列
            attention_mask: 注意力掩码，指示哪些位置是有效输入
            pixel_values: 图像像素值，用于多模态输入
            inputs_embeds: 输入嵌入向量，可替代input_ids直接提供嵌入
            past_key_values: 过去的key/value缓存，用于加速生成
            use_cache: 是否使用缓存机制
            proprio_feat: 本体感知特征（可选）
            **kwargs: 其他参数
        Returns:
            包含所有必要输入的字典
        """
        # 如果存在过去的key/value缓存，只需要最后一个token的input_ids
        # 这是为了在生成过程中利用缓存，避免重复计算历史token的注意力
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # 如果提供了inputs_embeds且没有past_key_values，使用inputs_embeds作为模型输入
        # 否则使用input_ids作为模型输入
        # 这确保inputs_embeds只在第一代步骤中使用
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 确保pixel_values和其他关键参数在model_inputs中得到保留
        # 这些参数对于多模态生成至关重要
        model_inputs.update(
            {
                "attention_mask": attention_mask,      # 注意力掩码
                "pixel_values": pixel_values,          # 图像像素值
                "past_key_values": past_key_values,    # 缓存的key/value对
                "use_cache": use_cache,                # 是否使用缓存
                "proprio_feat": proprio_feat,          # 本体感知特征
            }
        )

        return model_inputs

    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        """
        批量生成文本，支持图像和文本的多模态输入。
        Args:
            pixel_values: 图像像素值，可以是张量或字典形式
            texts: 输入文本列表
            return_string_probabilities: 可选，需要返回概率的字符串列表
            **kwargs: 其他传递给生成函数的参数
        Returns:
            生成的文本列表，或者当指定了return_string_probabilities时，返回对应的概率列表
        """
        # 获取语言模型的tokenizer
        tokenizer = self.llm_backbone.tokenizer

        # 准备输入数据：对每个文本进行tokenize并转换为tensor
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        
        # 处理图像数据，统一转换为设备上的张量格式
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # 创建输出列表
        gen_texts, gen_probabilities = [], []

        # 获取混合精度训练的数据类型
        autocast_dtype = self.llm_backbone.half_precision_dtype
        
        # 启用自动混合精度进行推理
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # 遍历批次中的每个样本
            for idx, input_ids in enumerate(batch_input_ids):
                # 处理当前样本的图像数据
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                # 根据是否需要返回字符串概率来处理不同的生成逻辑
                if return_string_probabilities is None:
                    # 不需要概率信息，直接生成文本
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    # 提取生成的token IDs（去除输入部分）
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]
                    # 解码生成的token IDs为文本，并去除特殊token和首尾空格
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    # 需要返回特定字符串的概率
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_scores=True,              # 输出分数用于计算概率
                        return_dict_in_generate=True,    # 返回生成过程的详细信息
                        **kwargs,
                    )

                    # 提取生成的token IDs（去除输入部分）
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # 解码生成的token IDs为文本，并去除特殊token和首尾空格
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # 计算所有token的概率分布（对logits进行softmax）
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # 获取指定字符串的归一化概率
                    # 根据字符串获取对应的token索引
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    # 提取这些token的概率值
                    string_probs_unnormalized = token_probs[slice_idxs]
                    # 归一化概率分布
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    # 转换为numpy数组并添加到结果列表
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        # 根据是否需要概率信息返回相应结果
        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
        """
        根据输入图像和提示文本生成响应文本。
        Args:
            image: 输入的PIL图像对象
            prompt_text: 提示文本字符串
            **kwargs: 其他传递给生成函数的参数
        Returns:
            生成的文本字符串
        """
        # 目前为了简化实现，只支持batch size为1的生成
        # 获取图像变换函数和语言模型的tokenizer
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # 准备输入数据
        # 对提示文本进行tokenization处理并转换为tensor
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        # 对图像进行预处理变换
        pixel_values = image_transform(image)
        # 将图像数据增加一个批次维度并移至指定设备
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # 调用父类的generate方法进行文本生成 --> 利用`GenerationMixin`的功能，实际会调用本类的forward方法
        # 获取语言模型的半精度数据类型用于混合精度推理
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # 输入token IDs，形状: [1, seq]
                pixel_values=pixel_values,      # 图像像素值，形状: [1, 3, res, res] 或 Dict[str, Shape[1, 3, res, res]]
                **kwargs
            )
            # fmt: on

        # 解码生成的token IDs为文本字符串
        # 仅解码生成的部分（去除输入部分），跳过特殊token并去除首尾空格
        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text
