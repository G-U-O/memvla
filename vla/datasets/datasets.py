"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
import tensorflow as tf

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from vla.action_tokenizer import ActionTokenizer
from vla.datasets.rlds import make_interleaved_dataset, make_single_dataset, \
    make_interleaved_episodic_dataset
from vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
# FIXME: 2.RLDS数据
@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        
        # For future action predictions
        if rlds_batch["action"].shape[0] > 1:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        else:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": ""},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # Add future actions to batch
        if rlds_batch["action"].shape[0] > 1:
            action = torch.tensor(action, dtype=torch.float32)
            action_mask = None
            if "action_mask" in rlds_batch:
                action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        # Mask prompt tokens
        labels[: int(torch.where(input_ids==2)[0][0])] = IGNORE_INDEX

        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        timesteps = rlds_batch['observation']['timestep']

        return dict(pixel_values=pixel_values,
                    input_ids=input_ids,
                    labels=labels,
                    dataset_name=dataset_name,
                    actions=action,
                    action_masks=action_mask,
                    timesteps=timesteps,
                    episode_ids=None,
                    )


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,                     # 数据根目录路径
        data_mix: str,                           # 数据混合名称或单个数据集名称
        batch_transform: RLDSBatchTransform,     # 批处理转换器
        resize_resolution: Tuple[int, int],      # 图像调整大小的目标分辨率
        shuffle_buffer_size: int = 256_000,      # 洗牌缓冲区大小，默认为256,000
        future_action_window_size: int = 0,      # 未来动作窗口大小，默认为0（不预测未来动作）
        train: bool = True,                      # 是否为训练模式，默认为True
        image_aug: bool = False,                 # 是否启用图像增强，默认为False
        load_all_data_for_training: bool = True, # 是否加载所有数据用于训练，默认为True
        load_depth=False,                        # 是否加载深度信息，默认为False
        load_proprio=False,                      # 是否加载本体感受信息，默认为False
    ) -> None:
        """轻量级封装RLDS TFDS管道，用于PyTorch/OpenVLA数据加载器"""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # 配置RLDS数据集
        if self.data_mix in OXE_NAMED_MIXTURES:
            # 如果指定的数据混合名称存在于预定义混合中，则使用该混合配置
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # 否则假设传入的是单个数据集名称，创建只有一个数据集的"混合"
            mixture_spec = [(self.data_mix, 1.0)]

        # 获取OXE数据集参数和权重
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),          # 加载主视角相机视图
            load_depth=load_depth,                   # 是否加载深度信息
            load_proprio=load_proprio,               # 是否加载本体感受信息
            load_language=True,                      # 加载语言指令
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,  # 动作归一化类型
        )
        
        # RLDS配置字典
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                    # 窗口大小，目前只处理单步
                future_action_window_size=future_action_window_size,  # 未来动作窗口大小，用于动作分块
                skip_unlabeled=True,                              # 跳过没有语言标签的轨迹
                #goal_relabeling_strategy="uniform",              # 目标重标记策略，当前未使用
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,       # 调整图像大小
                num_parallel_calls=16,               # 并行调用数，用于CPU密集型操作（解码、调整大小等）
            ),
            dataset_kwargs_list=per_dataset_kwargs,  # 数据集参数列表
            shuffle_buffer_size=shuffle_buffer_size, # 洗牌缓冲区大小
            sample_weights=weights,                  # 采样权重
            balance_weights=True,                    # 是否平衡权重
            traj_transform_threads=len(mixture_spec),# 轨迹转换线程数
            traj_read_threads=len(mixture_spec),     # 轨迹读取线程数
            train=train,                             # 是否为训练模式
            load_all_data_for_training=load_all_data_for_training,  # 是否加载所有数据用于训练
        )

        # 如果启用了图像增强，则添加图像增强参数
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),  # 随机调整裁剪
                random_brightness=[0.2],               # 随机亮度
                random_contrast=[0.8, 1.2],            # 随机对比度
                random_saturation=[0.8, 1.2],          # 随机饱和度
                random_hue=[0.05],                     # 随机色调
                augment_order=[                        # 增强应用顺序
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),

        # 初始化RLDS数据集
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
            # load_all_data_for_training=rlds_config["load_all_data_for_training"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class GroupRLDSDataset(RLDSDataset):
    def __init__(self, *args,
                 group_size: int = 16,
                 **kwargs):
        self.group_size = group_size
        super().__init__(*args, **kwargs)

    def make_dataset(self, rlds_config):
        return make_interleaved_episodic_dataset(
            **rlds_config,
            group_size=self.group_size,
            use_optim_group_sample=True,
        )

    def __iter__(self) -> Dict[str, Any]:
        episode_id = -1
        for rlds_batch in self.dataset.as_numpy_iterator():
            episode_id += 1
            indices = range(rlds_batch["action"].shape[0])
            for i in indices:
                frame = self.batch_transform(tree_map(lambda x: x[i], rlds_batch))
                frame["episode_ids"] = np.array([episode_id])
                yield frame


class StreamRLDSDataset(RLDSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_dataset(self, rlds_config):
        return make_interleaved_episodic_dataset(
            **rlds_config,
            use_optim_group_sample=False,
        )

    def __iter__(self) -> Dict[str, Any]:
        episode_id = -1
        for rlds_batch in self.dataset.as_numpy_iterator():
            episode_id += 1
            T = rlds_batch["action"].shape[0]
            for i in range(T):
                frame = self.batch_transform(
                    tree_map(lambda x: x[i], rlds_batch)
                )
                frame["episode_ids"] = np.array([episode_id])
                yield frame


def _frame_generator(batch_dict, batch_transform):
    length = batch_dict["action"].shape[0]
    for i in range(length):
        yield batch_transform(tree_map(lambda x: x[i], batch_dict))
