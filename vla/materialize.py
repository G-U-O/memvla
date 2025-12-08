"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type, Union

from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset, GroupRLDSDataset, StreamRLDSDataset
from vla.action_tokenizer import ActionTokenizer


def get_vla_dataset_and_collator(
    data_root_dir: Path,                    # 数据根目录路径
    data_mix: str,                          # 数据混合配置名称
    image_transform: ImageTransform,        # 图像预处理变换函数
    tokenizer: PreTrainedTokenizerBase,     # 文本分词器
    prompt_builder_fn: Type[PromptBuilder], # 提示构建器类型
    default_image_resolution: Tuple[int, int, int],  # 默认图像分辨率(通道数, 高度, 宽度)
    padding_side: str = "right",            # 填充方向，默认右侧填充
    predict_stop_token: bool = True,        # 是否预测停止token
    shuffle_buffer_size: int = 100_000,     # 洗牌缓冲区大小
    train: bool = True,                     # 是否为训练模式
    image_aug: bool = False,                # 是否进行图像增强
    future_action_window_size: int = 15,    # 未来动作窗口大小
    load_all_data_for_training: bool = True,  # 是否加载所有数据用于训练
    dataloader_type: str = "group",         # 数据加载器类型
    group_size: int = 16,                   # 分组大小
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """
    初始化RLDS数据集（封装TFDS），动作分词器，并初始化变换/整理函数。
    
    数据处理流程：
    1. 使用ActionTokenizer对动作进行分词编码
    2. 使用RLDSBatchTransform对数据批次进行预处理
    3. 使用PaddedCollatorForActionPrediction对批次数据进行填充整理
    4. 根据dataloader_type选择不同的数据集类型进行数据加载
    
    处理后的数据包含：
    - 图像数据：经过预处理和可能的增强的图像
    - 文本数据：经过分词器处理的文本序列
    - 动作数据：编码后的动作序列
    - 注意力掩码：用于指示有效位置的掩码
    - 时间步信息：用于记忆模块的时间步数据
    - 回合ID：用于区分不同回合的数据
    """

    # 初始化动作分词器，用于将连续动作转换为离散token
    action_tokenizer = ActionTokenizer(tokenizer)
    
    # FIXME:2.初始化批次变换函数，用于对RLDS数据进行预处理
    batch_transform = RLDSBatchTransform(
        action_tokenizer,           # 动作分词器
        tokenizer,                  # 文本分词器
        image_transform,            # 图像变换函数
        prompt_builder_fn,          # 提示构建器函数
        predict_stop_token=predict_stop_token,  # 是否预测停止token
    )

    # 初始化批次整理器，用于对变长序列进行填充
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,     # 模型最大序列长度
        tokenizer.pad_token_id,         # 填充token ID
        padding_side=padding_side,      # 填充方向
    )

    # 构建RLDS可迭代数据集，根据不同的数据加载器类型选择相应的数据集类
    if dataloader_type == "normal":
        # 普通数据集：基本的RLDS数据集实现
        dataset = RLDSDataset(
            data_root_dir,                          # 数据根目录
            data_mix,                               # 数据混合配置
            batch_transform,                        # 批次变换函数
            resize_resolution=default_image_resolution[1:],  # 调整图像分辨率(高度, 宽度)
            shuffle_buffer_size=shuffle_buffer_size,         # 洗牌缓冲区大小
            train=train,                            # 是否为训练模式
            future_action_window_size=future_action_window_size,  # 未来动作窗口大小
            image_aug=image_aug,                    # 是否进行图像增强
            load_all_data_for_training=load_all_data_for_training,  # 是否加载所有数据
        )
    elif dataloader_type == "group":
        # 分组数据集：将相关样本分组处理
        assert group_size > 1, "分组大小必须大于1才能使用分组数据集"
        dataset = GroupRLDSDataset(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            future_action_window_size=future_action_window_size,
            image_aug=image_aug,
            load_all_data_for_training=load_all_data_for_training,
            group_size=group_size,                  # 分组大小
        )
    elif dataloader_type == "stream":
        # 流式数据集：流式处理数据，适合长序列
        dataset = StreamRLDSDataset(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            future_action_window_size=future_action_window_size,
            image_aug=image_aug,
            load_all_data_for_training=load_all_data_for_training,
        )

    else:
        raise NotImplementedError(f"数据集类型 {dataloader_type} 未实现。")

    # 返回数据集、动作分词器和批次整理器
    return dataset, action_tokenizer, collator
