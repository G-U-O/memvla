#!/bin/bash
set -euo pipefail
export MKL_INTERFACE_LAYER=GNU

# ====== USER CONFIG ====== #
gpu_id=0
ckpt_path="/home/guojiahui/MemoryVLA/log/libero/memvla_libero_100--image_aug/checkpoints/step-040000-epoch-05-loss=0.0261.pt"
action_chunking_window=8
unnorm_key="libero_100_no_noops"

# ====== FUNCTIONS ====== #
find_free_port() {
  local min=${1:-2000}
  local max=${2:-30000}
  local port
  for ((i=0; i<200; i++)); do
    port=$(shuf -i"${min}"-"${max}" -n1)
    if ! lsof -iTCP:"${port}" -sTCP:LISTEN &>/dev/null; then
      echo "${port}"
      return 0
    fi
  done
  echo "ERROR: Cannot find free port" >&2
  exit 1
}

# ====== START ====== #
export CUDA_VISIBLE_DEVICES=${gpu_id}
port=$(find_free_port)

echo ">>> 使用端口：${port}"
echo "${port}" > .libero_eval_port
echo "${ckpt_path}" > .libero_eval_ckpt

echo ">>> Running deploy.py in memvla env..."
python deploy.py \
  --saved_model_path "${ckpt_path}" \
  --unnorm_key "${unnorm_key}" \
  --adaptive_ensemble_alpha 0.1 \
  --cfg_scale 1.5 \
  --port "${port}" \
  --action_chunking \
  --action_chunking_window "${action_chunking_window}"

DEPLOY_PID=$!
echo "${DEPLOY_PID}" > .libero_deploy_pid
echo ">>> deploy PID = ${DEPLOY_PID}"

# ===== 确保脚本退出时自动杀进程 =====
cleanup() {
  if ps -p ${DEPLOY_PID} > /dev/null 2>&1; then
    echo ">>> killing deploy PID ${DEPLOY_PID}"
    kill ${DEPLOY_PID}
  fi
}
trap cleanup EXIT

wait ${DEPLOY_PID}