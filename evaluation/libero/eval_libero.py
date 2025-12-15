#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIBERO环境评估脚本
该脚本用于在LIBERO基准测试环境中评估训练好的模型性能
"""

import os
from dataclasses import dataclass
from typing import List, Union
import draccus
import numpy as np
import tqdm

# 设置MUJOCO使用osmesa渲染模式 FIXME：服务器上使用egl模式
os.environ['MUJOCO_GL'] = 'osmesa'
# apt install -y libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev


# 导入LIBERO相关模块
from libero.libero import benchmark

# 导入自定义工具模块
from libero_utils import (
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from robot_utils import DATE_TIME, set_seed_everywhere

# 让tensorflow只看到GPU设备
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


@dataclass
class GenerateConfig:
    """
    评估配置类
    定义评估过程中所需的各种参数
    """
    # fmt: off
    task_suite_name: str = "libero_spatial" # 任务套件名称。可选值: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10 # 在模拟器中等待对象稳定的步数
    num_trials_per_task: int = 50 # 每个任务的运行回合数
    spcial_task_id: Union[List[int], int, None] = None # 要评估的任务ID列表（默认: None，评估套件中的所有任务）
    run_id_note: str = "" # 运行标识备注
    local_log_dir: str = "./logs/eval_libero" # 评估日志的本地目录
    seed: int = 7 # 随机种子（用于可重现性）
    resolution: Union[int, tuple] = 256 # 模型输入的图像分辨率
    port: int = 6800 # 服务端口
    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    """
    LIBERO环境评估主函数
    
    Args:
        cfg: 评估配置参数
    """
    if cfg.spcial_task_id is not None and isinstance(cfg.spcial_task_id, int):
        cfg.spcial_task_id = [cfg.spcial_task_id]

    # 设置随机种子以确保可重现性
    set_seed_everywhere(cfg.seed)

    # 初始化本地日志记录
    run_id = f"{cfg.task_suite_name}-{cfg.num_trials_per_task}trials-seed{cfg.seed}-{cfg.run_id_note}-{DATE_TIME}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # 初始化LIBERO任务套件
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # 获取期望的图像尺寸
    resize_size = cfg.resolution # TODO 需要在配置中完成

    ################################################################
    ### 导入策略模块
    from vla_policy import LLaVAClient
    policy = LLaVAClient(base_url=f'http://localhost:{cfg.port}')
    ################################################################

    # 开始评估
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # 如果指定了特定任务，则跳过未指定的任务
        if cfg.spcial_task_id is not None and task_id not in cfg.spcial_task_id:
            print(f"Skipping task {task_id}...")
            continue

        # 获取任务
        task = task_suite.get_task(task_id)

        # 获取默认的LIBERO初始状态
        initial_states = task_suite.get_task_init_states(task_id)

        # 初始化LIBERO环境和任务描述
        env, task_description = get_libero_env(task, resolution=256)

        # 开始执行回合
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # 重置环境
            env.reset()
            policy.reset()
            episode_first_frame = 'True'

            # 设置初始状态
            obs = env.set_init_state(initial_states[episode_idx])

            # 环境设置
            t = 0
            replay_images = []
            # 根据不同的任务套件设置最大步数
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # 最长的训练演示有193步
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # 最长的训练演示有254步
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # 最长的训练演示有270步
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # 最长的训练演示有505步
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # 最长的训练演示有373步

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # 重要：前几个时间步什么都不做，因为模拟器会掉落物体
                    # 我们需要等待它们稳定下来
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                        t += 1
                        continue

                    # 获取预处理后的图像
                    img = get_libero_image(obs, resize_size)

                    # 保存预处理后的图像用于回放视频
                    replay_images.append(img)

                    # 准备观测字典
                    observation = {
                        "base_cam": img,
                        "states": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    ###############################################################################
                    ### 这里我们需要使用flask从模型获取动作
                    ### 我们需要一个策略来获取动作
                    action = policy.process_frame(text=task_description,
                                                  episode_first_frame=episode_first_frame,
                                                  **observation)

                    # 替换分号为空格
                    if ';' in action:
                        action = action.replace(';', ' ')

                    # 字符串转numpy数组
                    action = action.split(' ')
                    action = [float(x) for x in action]
                    action = np.array(action, dtype=float)

                    episode_first_frame = 'False'
                    action_dim = 7

                    # 根据数据中夹爪的定义调整夹爪动作值
                    for i in range(len(action)):
                        if i % action_dim == action_dim - 1:
                            if action[i] == 1.0:
                                action[i] = -1.0
                            elif action[i] == 0.0:
                                action[i] = 1.0

                    done_flag = False
                    chunk_size = len(action) // action_dim
                    for i in range(chunk_size):
                        action_chunk = action[i * action_dim:(i + 1) * action_dim]

                        # 在环境中执行动作
                        obs, reward, done, info = env.step(action_chunk)
                        if done:
                            task_successes += 1
                            total_successes += 1
                            done_flag = True
                            break
                        t += 1

                    if done_flag:
                        break

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # 保存回合的回放视频
            rollout_dir = os.path.join(cfg.local_log_dir, run_id + "_videos")
            save_rollout_video(
                replay_images, total_episodes,
                success=done, task_description=task_description,
                log_file=log_file,
                rollout_dir=rollout_dir,
            )

            # 记录当前结果
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # 记录最终结果
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    log_file.close()

if __name__ == "__main__":
    eval_libero()