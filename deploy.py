# 导入所需库
import numpy as np
from PIL import Image
from typing import Optional
import os
import argparse
import yaml
from argparse import Namespace
import math
from flask import Flask, request, jsonify
import tempfile

import torch

# 导入自定义模块
from vla import load_vla
from evaluation.simpler_env.adaptive_ensemble import AdaptiveEnsembler

# 初始化Flask应用
app = Flask(__name__)

# 定义MemVLA服务类
class MemVLAService:
    def __init__(
        self,
        saved_model_path: str = "",           # 模型保存路径
        unnorm_key: str = None,               # 反归一化键值
        image_size: list[int] = [224, 224],   # 图像尺寸
        cfg_scale: float = 1.5,               # 分类器引导比例
        num_ddim_steps: int = 10,             # DDIM步骤数
        use_ddim: bool = True,                # 是否使用DDIM
        use_bf16: bool = False,               # 是否使用bfloat16精度
        action_ensemble: bool = True,         # 是否启用动作集成
        adaptive_ensemble_alpha: float = 0.1, # 自适应集成参数alpha
        action_ensemble_horizon: int = 2,     # 动作集成窗口大小
        action_chunking: bool = False,        # 是否启用动作分块
        action_chunking_window: Optional[int] = None, # 动作分块窗口大小
        args=None,
    ) -> None:
        # 设置环境变量禁用tokenizers并行处理
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # 确保动作分块和动作集成不能同时启用
        assert not (action_chunking and action_ensemble), "Now 'action_chunking' and 'action_ensemble' cannot both be True."

        self.unnorm_key = unnorm_key
        print(f"*** unnorm_key: {unnorm_key} ***")

        # 处理参数
        kwargs = vars(args).copy()
        for k in [
            "model_id_or_path", "saved_model_path", "pretrained_checkpoint",
        ]:
            kwargs.pop(k, None)

        # 加载VLA模型
        self.vla = load_vla(
          model_id_or_path=saved_model_path,
          load_for_training=False,
          **kwargs,
        )
        # 将模型移至GPU并设置为评估模式
        self.vla = self.vla.to("cuda").eval()
        
        # 根据配置选择精度模式
        if use_bf16:
            print("Using bfloat16 inference mode (auto-conversion for all modules).")
            self.vla = self.vla.to(torch.bfloat16)
        else:
            print("Using standard float32 inference mode.")
            self.vla = self.vla.to(torch.float32)

        self.cfg_scale = cfg_scale
        self.image_size = image_size
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.action_chunking = action_chunking
        self.action_chunking_window = action_chunking_window
        
        # 初始化动作集成器
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None

        self.args = args
        self.reset()

    def reset(self) -> None:
        # 重置动作集成器状态
        if self.action_ensemble:
            self.action_ensembler.reset()

    def step(
        self,
        image: str,                          # 图像文件路径
        task_description: str = None,        # 任务描述
        episode_first_frame: str = 'False',  # 是否为新episode的第一帧
        *args, **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        输入:
            image: 图像文件路径
            task_description: 任务描述（可选）
            episode_first_frame: 'True' 或 'False'，表示当前帧是否为episode的第一帧
        输出:
            action: 包含末端执行器和夹爪的7自由度动作列表
        """

        # 验证episode_first_frame参数
        assert episode_first_frame in ['True', 'False']

        # 如果是新episode则重置状态
        if episode_first_frame == 'True':
            self.reset()

        # 打开图像
        image: Image.Image = Image.open(image)

        # [重要!]：请在此处以与微调期间完全相同的方式处理输入图像，
        # 确保推理与训练之间的一致性。尽可能保证夹爪在处理后的图像中可见。
        resized_image = resize_image(image, size=self.image_size)

        # 保存调整大小的图像用于调试
        resized_image.save("resized_image.png")
        
        # 使用VLA模型预测动作
        unnormed_actions, normalized_actions = self.vla.predict_action(
            image=resized_image, 
            instruction=task_description,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale, 
            use_ddim=self.use_ddim, 
            num_ddim_steps=self.num_ddim_steps,
            episode_first_frame=episode_first_frame,
        )

        # 根据不同模式处理动作输出
        if self.action_ensemble:
            # 动作集成模式
            unnormed_actions = self.action_ensembler.ensemble_action(unnormed_actions)
            # 将夹爪开合状态转换为0或1
            unnormed_actions[6] = unnormed_actions[6] > 0.5
            action = unnormed_actions.tolist()
        elif self.action_chunking:
            # 动作分块模式
            if self.action_chunking_window is not None:
                chunked_actions = []
                for i in range(0, self.action_chunking_window):
                    chunked_actions.append(unnormed_actions[i].tolist())
                action = chunked_actions
            else:
                raise ValueError("Please specify the 'action_chunking_window' when using action chunking.")
        else:
            # 单动作模式
            unnormed_actions = unnormed_actions[0]
            action = unnormed_actions.tolist()

        print(f"Instruction: {task_description}")
        return action


# [重要!]：请修改此处的图像处理代码，确保输入图像的处理方式与微调阶段完全一致。
# 尽可能保证夹爪在处理后的图像中可见。
def resize_image(image: Image, size=(224, 224), shift_to_left=0):
    w, h = image.size
    # 计算左边距以进行中心裁剪
    left_margin = (w - h) // 2 - shift_to_left
    left_margin = min(max(left_margin, 0), w - h)
    # 裁剪正方形区域
    image = image.crop((left_margin, 0, left_margin + h, h))
    # 调整图像大小
    image = image.resize(size, resample=Image.LANCZOS)
    # 进一步缩放和调整大小
    image = scale_and_resize(image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5)
    return image

# 由于微调过程中使用了随机裁剪数据增强，这里先进行中心裁剪然后调整回原始大小
def scale_and_resize(image : Image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5):
    w, h = image.size
    # 计算新的宽高
    new_w = int(w * math.sqrt(scale))
    new_h = int(h * math.sqrt(scale))
    # 计算边距
    margin_w_max = w - new_w
    margin_h_max = h - new_h
    margin_w = int(margin_w_max * margin_w_ratio)
    margin_h = int(margin_h_max * margin_h_ratio)
    # 裁剪并调整大小
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    image = image.resize(target_size, resample=Image.LANCZOS)
    return image


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_path", type=str, default="")
parser.add_argument("--unnorm_key", type=str, default='custom_finetuning')
parser.add_argument("--image_size", type=list[int], default=[224, 224])
parser.add_argument("--cfg_scale", type=float, default=1.5)
parser.add_argument("--port", type=int, default=2345)
parser.add_argument("--use_bf16", action="store_true")
parser.add_argument("--action_ensemble", action="store_true")
parser.add_argument("--action_ensemble_horizon", type=int, default=2)
parser.add_argument("--adaptive_ensemble_alpha", type=float, default=0.1)
parser.add_argument("--action_chunking", action="store_true")
parser.add_argument("--action_chunking_window", type=int, default=None)

args = parser.parse_args()

# 从配置文件加载参数并与命令行参数合并
with open(os.path.join(os.path.dirname(os.path.dirname(args.saved_model_path)), "config.yaml"), "r") as f:
    yaml_args = yaml.safe_load(f) or {}

def deep_update(base: dict, updates: dict):
    """递归合并两个字典，更新优先但保留基础键"""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

cli_args = vars(args)
merged_args = deep_update(yaml_args.copy(), cli_args)

args = Namespace(**merged_args)

# 创建推理服务实例
inferencer = MemVLAService(
    saved_model_path=args.saved_model_path,
    unnorm_key=args.unnorm_key,
    image_size=args.image_size,
    cfg_scale=args.cfg_scale,
    use_bf16=args.use_bf16,
    action_ensemble=args.action_ensemble,
    adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    action_ensemble_horizon=args.action_ensemble_horizon,
    action_chunking=args.action_chunking,
    action_chunking_window=args.action_chunking_window,
    args=args,
)


# 定义API端点
@app.route('/process_frame', methods=['POST'])
def inference():
    # 检查是否提供了图像
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image = request.files['image']

    # 检查是否提供了文本指令
    if 'text' not in request.form:
        return jsonify({'error': 'No text provided'}), 400
    query = request.form['text']

    # 检查是否提供了episode_first_frame标志
    if 'episode_first_frame' not in request.form:
        return jsonify({'error': 'No episode_first_frame provided'}), 400
    episode_first_frame = request.form['episode_first_frame']

    # 将图像保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        image.save(temp_image.name)
        temp_image_path = temp_image.name

    # 构建输入查询
    input_query = {
        'task_description': query,
        'episode_first_frame': episode_first_frame,
    }

    # 执行推理
    answer = inferencer.step(temp_image_path, **input_query)
    print(answer)

    # 根据不同模式将动作数组转换为字符串
    if inferencer.action_ensemble:
        # 动作集成模式直接转换动作列表
        action_str = ' '.join([str(x) for x in answer])
    elif inferencer.action_chunking:
        # 动作分块模式转换分块的动作
        action_str = ';'.join([' '.join([str(x) for x in chunk]) for chunk in answer])
    else:
        # 单动作模式
        action_str = ' '.join([str(x) for x in answer])

    return jsonify({'response': action_str})

# 启动Flask应用
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=args.port)