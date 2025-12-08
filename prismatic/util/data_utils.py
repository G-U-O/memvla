"""
data_utils.py

通用工具和类，用于促进数据加载和批处理。
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple, List

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# HuggingFace 默认 / LLaMa-2 IGNORE_INDEX (用于标签)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """
    对嵌套字典中的每个值应用函数。
    
    Args:
        fn：要应用的函数
        tree：嵌套字典
        
    Returns:
        应用函数后的新字典
    """
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """
    对嵌套字典中的每个值应用函数，并传递键路径作为参数。
    
    Args:
        fn：要应用的函数，接收键路径和值作为参数
        tree：嵌套字典
        keys：当前键路径
        
    Returns:
        应用函数后的新字典
    """
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) 
        for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    """
    用于语言模型训练的数据批处理器，支持填充和多模态数据处理。
    """
    model_max_length: int                          # 模型最大序列长度
    pad_token_id: int                              # 填充token的ID
    default_image_resolution: Tuple[int, int, int] # 默认图像分辨率
    padding_side: str = "right"                    # 填充方向，默认右侧填充
    pixel_values_dtype: torch.dtype = torch.float32 # 像素值数据类型

    def __post_init__(self) -> None:
        """初始化后设置虚拟像素值"""
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        对一批数据实例进行批处理和填充。

        Args:
            instances：数据实例列表，每个实例包含input_ids、labels和pixel_values
            
        Returns:
            批处理后的数据字典
        """
        # 提取input_ids和labels
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # 目前只支持右侧填充的Tokenizer
        # 使用RNN工具进行填充
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # 截断超出最大长度的部分
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # 通过检查pad_token_id生成注意力掩码
        attention_mask = input_ids.ne(self.pad_token_id)

        # === 处理"单模态"(仅语言) vs. "多模态"数据 ===

        # 有些样本是"仅语言"的 --> 构建一个multimodal_indices张量，便于切片操作
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # 根据类型(torch.Tensor或Dict[str, torch.Tensor])和None的存在情况堆叠所有pixel_values
        if len(multimodal_indices) == 0:
            # 没有多模态数据，全部使用虚拟像素值
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            # 像素值是张量类型
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            # 像素值是字典类型
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    """
    用于动作预测的数据批处理器，专门用于VLA(视觉-语言-动作)训练。
    """
    model_max_length: int                     # 模型最大序列长度
    pad_token_id: int                         # 填充token的ID
    padding_side: str = "right"               # 填充方向，默认右侧填充
    pixel_values_dtype: torch.dtype = torch.float32 # 像素值数据类型

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        对一批VLA数据实例进行批处理
       
        Args:
            instances： VLA数据实例列表
        Returns:
            批处理后的数据字典
        """
        # 提取基本数据
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        
        # 提取可选数据
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        if "episode_ids" in instances[0]:
            episode_ids = np.concatenate([
                instance["episode_ids"] for instance in instances
            ], axis=0)
        else:
            episode_ids = None

        if "timesteps" in instances[0]:
            timesteps = np.concatenate([
                instance["timesteps"] for instance in instances
            ], axis=0)
        else:
            timesteps = None

        # 目前训练期间只支持右侧填充的Tokenizers
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # 截断超出最大长度的部分
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # 通过检查pad_token_id生成注意力掩码
        attention_mask = input_ids.ne(self.pad_token_id)

        # [约定] VLA训练不允许"单模态"数据!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # 根据类型堆叠所有pixel_values (torch.Tensor或Dict[str, torch.Tensor])
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # 添加连续动作和批处理
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions)
        action_masks = [instance["action_masks"] for instance in instances]
        action_masks = torch.stack(action_masks)

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            action_masks=action_masks,
            dataset_names=dataset_names,
            episode_ids=episode_ids,
            timesteps=timesteps,
        )

        return output