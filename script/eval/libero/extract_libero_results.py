#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
提取和汇总LIBERO评估结果的脚本
该脚本用于解析评估日志文件并生成成功率统计表
"""

import os
import re
from collections import defaultdict

# ===== 用户输入 =====
ckpt_paths = [
    # 在此处添加您的日志根目录
    # 例如: ./log/libero/memvla_libero_spatial--image_aug
]

# ===== 正则表达式定义 =====
# 匹配步骤编号的正则表达式
re_step = re.compile(r"step-(\d+)")
# 匹配已完成回合数的正则表达式
re_episode = re.compile(r"# episodes completed so far:\s*(\d+)")
# 匹配当前总成功率的正则表达式
re_total = re.compile(r"Current total success rate:\s*([\d\.]+)")
# 匹配试验次数的正则表达式
re_trials = re.compile(r"(\d+)trials", re.IGNORECASE)


def is_complete(fname: str, episodes: int) -> bool:
    """
    判断评估是否完成
    
    Args:
        fname: 文件名
        episodes: 已完成的回合数
    Returns:
        bool: 如果评估完成返回True，否则返回False
    """
    fn = fname.lower()
    if any(k in fn for k in ["spatial", "object", "goal", "libero_10"]):
        return episodes == 500
    if "libero_90" in fn:
        m = re_trials.search(fn)
        if m:
            trials = int(m.group(1))
            return episodes >= 90 * trials
        return episodes >= 450
    return False


def get_category(fname: str):
    """
    根据文件名获取类别
    
    Args:
        fname: 文件名
    Returns:
        str: 类别名称
    """
    fn = fname.lower()

    # --- 特殊处理带有试验次数的libero_90 ---
    if "libero_90" in fn:
        m = re.search(r"libero_90-(\d+)trials", fn)
        if m:
            trials = m.group(1)
            return f"90-{trials}trials"
        return "90"  # 如果没有明确的试验次数，则回退到默认值

    # --- 正常情况 ---
    for key in ["spatial", "object", "goal", "libero_10"]:
        if key in fn:
            return key.replace("libero_", "")
    return "unknown"


def norm_basename(path: str) -> str:
    """
    将路径基准名标准化为短横线风格，用于匹配运行目录如'memvla-libero-*.pt'
    
    Args:
        path: 路径字符串
        
    Returns:
        str: 标准化后的基准名
    """
    b = os.path.basename(path)
    return b.replace("_", "-")


def find_eval_root(base: str) -> str | None:
    """
    查找并返回评估根目录
    优先查找 base/eval_libero；如果不存在则回退到 parent/eval_libero
    
    Args:
        base: 基础路径
        
    Returns:
        str | None: 找到的评估根目录路径，未找到则返回None
    """
    direct = os.path.join(base, "eval_libero")
    if os.path.isdir(direct):
        return direct
    parent = os.path.join(os.path.dirname(base), "eval_libero")
    if os.path.isdir(parent):
        return parent
    return None


def candidate_run_dirs(eval_root: str, base: str) -> list[tuple[str, str]]:
    """
    返回候选的运行目录列表 [(name, run_dir_path), ...]
    - 如果eval_root在base内部：包含所有子目录
    - 如果eval_root是父级共享的：只包含名称以标准化base名称开头的子目录
    
    Args:
        eval_root: 评估根目录
        base: 基础路径
        
    Returns:
        list[tuple[str, str]]: 运行目录列表，每个元素为(目录名, 完整路径)的元组
    """
    inside = eval_root.startswith(os.path.abspath(base) + os.sep)
    all_subdirs = [
        d for d in os.listdir(eval_root)
        if os.path.isdir(os.path.join(eval_root, d))
    ]
    if inside:
        return [(d, os.path.join(eval_root, d)) for d in sorted(all_subdirs)]
    # 父级共享的情况：过滤
    key = norm_basename(base)
    filtered = [d for d in all_subdirs if d.startswith(key)]
    return [(d, os.path.join(eval_root, d)) for d in sorted(filtered)]


def collect_txt_paths(run_dir: str) -> list[str]:
    """
    收集运行目录下的*.txt文件（深度1）以及嵌套一级目录下的文件（深度2）
    处理如下结构：
      run_dir/*.txt
      run_dir/<sub>/*.txt
      
    Args:
        run_dir: 运行目录路径
        
    Returns:
        list[str]: txt文件路径列表
    """
    txts = []
    # 深度1
    for name in os.listdir(run_dir):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p) and p.endswith(".txt"):
            txts.append(p)
    # 深度2
    for name in os.listdir(run_dir):
        pdir = os.path.join(run_dir, name)
        if os.path.isdir(pdir):
            for sub in os.listdir(pdir):
                p = os.path.join(pdir, sub)
                if os.path.isfile(p) and p.endswith(".txt"):
                    txts.append(p)
    return sorted(txts)


def name_from_run_dir(run_dir_name: str) -> str:
    """
    从运行目录名称中提取显示名称
    优先使用数字步骤 → '20k'；否则直接使用运行目录名称
    
    Args:
        run_dir_name: 运行目录名称
        
    Returns:
        str: 显示名称
    """
    m = re_step.search(run_dir_name)
    if m:
        step_num = int(m.group(1))
        return f"{step_num // 1000}k"
    return run_dir_name


# ===== 主程序 =====
for base in ckpt_paths:
    eval_root = find_eval_root(base)
    if not eval_root:
        print(f"\n❌ {base} has no eval_libero (neither local nor parent), skipped")
        continue

    runs = candidate_run_dirs(eval_root, base)
    if not runs:
        print(f"\n❌ {base} no matching runs under {eval_root}, skipped")
        continue

    print(f"\n📂 {os.path.basename(base)}")

    # name -> { category -> max_rate }
    results = defaultdict(lambda: defaultdict(float))

    for run_dir_name, run_dir in runs:
        name = name_from_run_dir(run_dir_name)
        cat_to_rates = defaultdict(list)

        for txt_path in collect_txt_paths(run_dir):
            fname = os.path.basename(txt_path)
            category = get_category(fname)

            try:
                text = open(txt_path, "r", encoding="utf-8", errors="ignore").read()
            except:
                continue

            ep_match = None
            total_match = None
            for m in re_episode.finditer(text):
                ep_match = m
            for m in re_total.finditer(text):
                total_match = m
            if not ep_match or not total_match:
                continue

            episodes = int(ep_match.group(1))
            total_rate = float(total_match.group(1))

            if is_complete(fname, episodes):
                cat_to_rates[category].append(total_rate)

        for cat, vals in cat_to_rates.items():
            results[name][cat] = max(vals)

    if not results:
        print("(no complete results)")
        continue

    all_cats = sorted({c for run in results.values() for c in run.keys()})
    print("\n| Name | " + " | ".join(all_cats) + " |")
    print("|" + "|".join("---" for _ in range(len(all_cats) + 1)) + "|")

    def sort_key(n: str):
        return (0, int(n[:-1])) if n.endswith("k") and n[:-1].isdigit() else (1, n)

    for name in sorted(results.keys(), key=sort_key):
        vals = [f"{results[name][c]:.3f}" if c in results[name] else "-" for c in all_cats]
        print("| " + " | ".join([name] + vals) + " |")

    cat_best = defaultdict(lambda: ("", 0.0))
    for name, cats in results.items():
        for c, v in cats.items():
            if v > cat_best[c][1]:
                cat_best[c] = (name, v)

    print()
    for c, (name, v) in cat_best.items():
        print(f"⭐ best({c}): {name} ({v:.3f})")