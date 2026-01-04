#!/bin/bash
set -euo pipefail
# ===== HEADLESS RENDER SETTINGS =====
# export MUJOCO_GL=osmesa
# export EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export EGL_PLATFORM=surfaceless
unset DISPLAY
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so

echo "[info] Using MUJOCO_GL=${MUJOCO_GL}"
# ====== USER CONFIG ====== #
task_suite_name="libero_100"
num_trials_per_task=50
run_id_note="eval_run"

# ====== READ INFO FROM FILE ====== #
if [ ! -f .libero_eval_port ]; then
  echo "ERROR: port file not found! Run run_deploy.sh first."
  exit 1
fi

if [ ! -f .libero_eval_ckpt ]; then
  echo "ERROR: ckpt file not found! Run run_deploy.sh first."
  exit 1
fi

port=$(cat .libero_eval_port)
ckpt_path=$(cat .libero_eval_ckpt)
local_log_dir="$(dirname "$(dirname "$ckpt_path")")/eval_libero/$(basename "$ckpt_path")"

echo ">>> 使用端口：${port}"
echo ">>> ckpt：${ckpt_path}"
echo ">>> log_dir：${local_log_dir}"

# ====== START EVAL ====== #
python evaluation/libero/eval_libero.py \
  --task_suite_name "${task_suite_name}" \
  --num_trials_per_task "${num_trials_per_task}" \
  --run_id_note "${run_id_note}" \
  --local_log_dir "${local_log_dir}" \
  --port "${port}"

echo ">>> Evaluation done!"
