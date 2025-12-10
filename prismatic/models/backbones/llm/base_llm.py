"""
base_llm.py

Abstract class definition of a large (autoregressive) language model backbone (LLM), with full annotations of class
methods, utility functions, and initialization logic.

We also define the generic HFLLMBackbone class here, providing a default interface for loading any HF
AutoModelForCausalLM (e.g., LLamaForCausalLM). In general, we make the assumption that any given LLM backbone implements
the AutoModelForCausalLM API (though we may add Seq2Seq models in the future).

We make this assumption to keep the LLM handling in this codebase relatively lightweight, and to inherit all the nice HF
utilities around different types of decoding/generation strategies.
"""

import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Optional, Sequence, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.overwatch import initialize_overwatch

# Suppress HF Deprecation Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for arbitrary HF LLM Backbones ===
class LLMBackbone(nn.Module, ABC):
    def __init__(self, llm_backbone_id: str) -> None:
        super().__init__()
        self.identifier = llm_backbone_id

        # Instance attributes for an LLM Backbone
        self.llm: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizerBase = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None: ...

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the LLM given targets (labels), returning the scalar Cross-Entropy Loss"""
        raise NotImplementedError

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def prompt_builder_fn(self) -> Type[PromptBuilder]: ...

    @property
    @abstractmethod
    def transformer_layer_cls(self) -> Type[nn.Module]: ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]: ...

    @property
    def embed_dim(self) -> int:
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id


# === 抽象基类：任意HF因果语言模型 ===
class HFCausalLLMBackbone(LLMBackbone, ABC):
    """HuggingFace因果语言模型的通用实现基类"""
    
    def __init__(
        self,
        llm_backbone_id: str,          # LLM骨干网络标识符
        llm_family: str,               # LLM家族名称（如Llama、Mistral等）
        llm_cls: Type[PreTrainedModel], # 实际的模型类（如LlamaForCausalLM）
        hf_hub_path: str,              # HF Hub上的模型路径
        llm_max_length: int = 2048,    # 最大序列长度
        hf_token: Optional[str] = None, # HF访问令牌
        inference_mode: bool = False,  # 是否为推理模式
        use_flash_attention_2: bool = False, # 是否使用Flash Attention 2
    ) -> None:
        # 调用父类初始化
        super().__init__(llm_backbone_id)
        self.llm_family = llm_family
        self.llm_max_length = llm_max_length
        self.inference_mode = inference_mode

        # 初始化LLM（必要时从HF Hub下载）
        # 注意：我们避免使用AutoModel API以便更明确地处理LLM特定细节
        if not self.inference_mode:
            # 训练模式：加载预训练权重
            overwatch.info(f"Loading [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]", ctx_level=1)
            self.llm = llm_cls.from_pretrained(
                hf_hub_path,
                token=hf_token,
                use_flash_attention_2=use_flash_attention_2 if not self.inference_mode else False,
                # 设置以下参数防止HF产生警告；我们需要贪婪解码
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        # [约定] inference_mode表示我们从预训练检查点加载；无需加载基础权重
        else:
            # 推理模式：仅构建模型结构，不加载权重
            overwatch.info(f"Building empty [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]", ctx_level=1)
            llm_config = AutoConfig.from_pretrained(hf_hub_path, token=hf_token)
            self.llm = llm_cls._from_config(llm_config)

        # 轻量级处理：设置一些LLM参数
        # => 设置 decoder.use_cache = False --> 与梯度检查点不兼容（一般训练时）
        # 参考：https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
        self.llm.config.use_cache = False if not self.inference_mode else True

        # => 当启用梯度检查点且底层LLM没有"可训练"参数（requires_grad为False）时，
        #    反向传播会失败；设置enable_input_requires_grads()注册一个新的前向钩子来解决此问题
        #    对于"全微调"设置也完全安全
        if not self.inference_mode:
            self.llm.enable_input_require_grads()

        # 加载（快速）分词器
        overwatch.info(f"Loading [bold]{llm_family}[/] (Fast) Tokenizer via the AutoTokenizer API", ctx_level=1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_hub_path, model_max_length=self.llm_max_length, token=hf_token, padding_side="right"
        )

        # 验证 => 我们的VLM逻辑假设新输入的标记化以<BOS>标记开始，除非add_special_tokens=False；
        #         对于这些模型，我们经验性地发现，在BOS之后添加图像块效果更好。
        # 因此我们显式验证分词器符合预期行为；如果您读到这里，可能是因为您正在添加一个具有不同分词器行为的新LLM。
        # 如果是这样，请随意覆盖下面的SPECIAL_CASES集合，但请确保在datasets.py和VLM的forward()逻辑中做出相应更改！
        
        # 特殊情况处理
        SPECIAL_CASES = {
            # Phi-2分词器默认不添加任何BOS标记，并将BOS == EOS == ""
            # => 我们会在第一个输入前加上BOS（与图像标记插入逻辑配合良好；已验证对基础LLM生成有效）
            # => 类似Llama-2分词器 -- 我们会添加特殊的PAD标记用于训练目的
            "phi-2-3b",
        }
        if self.identifier in SPECIAL_CASES:
            return

        # 注意 => 这个断言应该适用于所有Llama衍生的分词器（LlamaTokenizerFast ==> 包括Mistral！）
        assert (self.tokenizer("Test 123", add_special_tokens=True).input_ids[0] == self.tokenizer.bos_token_id) and (
            self.tokenizer("Test 123", add_special_tokens=False).input_ids[0] != self.tokenizer.bos_token_id
        ), (
            f"Default Tokenizer of type `{type(self.tokenizer)}` does not automatically prefix inputs with BOS token!\n"
            "Please read the comment in [base_llm.py](file://d:\doctor-code\code\MemoryVLA-openvla-codebase\MemoryVLA-openvla-codebase\prismatic\models\backbones\llm\base_llm.py) for more information!"
        )

    def get_fsdp_wrapping_policy(self) -> Callable:
        """返回一个transformer_auto_wrap_policy，其中包装每个self.transformer_layer_cls实例"""
        transformer_block_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls}
        )

        return transformer_block_policy

    def enable_gradient_checkpointing(self) -> None:
        """调度到底层LLM实例的gradient_checkpointing_enable；为所有PretrainedModel定义"""
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """将输入ID嵌入为向量表示"""
        return self.llm.get_input_embeddings()(input_ids)

    # [约定] 应该匹配底层llm实例的forward调用！
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """执行前向传播"""
        output: CausalLMOutputWithPast = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return output