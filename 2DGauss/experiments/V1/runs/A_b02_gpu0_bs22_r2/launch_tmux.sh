#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

SESSION="2dgauss_a_b02_gpu0_bs22_r2_uhdhaze_patch_cache"
CONFIG="experiments/V1/templates/T01/configs/a_b02_gpu0_bs22_r2_uhdhaze_patch_cache.toml"
RUN_DIR="experiments/V1/runs/A_b02_gpu0_bs22_r2"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/train_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"

PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-0}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-42000}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-0}"

if [ "${PHYSICAL_GPU}" != "0" ]; then
  echo "Refusing to launch: this large-batch package is GPU0-only, got GPU${PHYSICAL_GPU}" >&2
  exit 1
fi

GPU_ROW="$(
  nvidia-smi \
    --query-gpu=memory.free,utilization.gpu \
    --format=csv,noheader,nounits \
    -i "${PHYSICAL_GPU}" \
    | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 " " $2}'
)"
read -r FREE_MIB GPU_UTIL <<< "${GPU_ROW}"

if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
  echo "Refusing to launch: GPU${PHYSICAL_GPU} free memory ${FREE_MIB}MiB < ${MIN_FREE_MIB}MiB" >&2
  nvidia-smi >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  echo "would_launch_session=${SESSION}"
  echo "physical_gpu=${PHYSICAL_GPU}"
  echo "gpu_free_mib=${FREE_MIB}"
  echo "gpu_util_percent=${GPU_UTIL}"
  echo "config=${CONFIG}"
  echo "run_dir=${RUN_DIR}"
  echo "log_path=${LOG_PATH}"
  echo "python_bin=${PYTHON_BIN}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  cd '${PROJECT_ROOT}' && \
  export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
  export CUDA_VISIBLE_DEVICES='${PHYSICAL_GPU}' && \
  export TWODGAUSS_ALLOWED_PHYSICAL_GPUS='0,3' && \
  export OMP_NUM_THREADS=2 && \
  export MKL_NUM_THREADS=2 && \
  export OPENBLAS_NUM_THREADS=2 && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  export PYTHONUNBUFFERED=1 && \
  export PYTHONFAULTHANDLER=1 && \
  set -o pipefail; \
  '${PYTHON_BIN}' -u scripts/train_v1_t01.py --config '${CONFIG}' 2>&1 | tee '${LOG_PATH}'; \
  status=\${PIPESTATUS[0]}; \
  printf 'train_exit_code=%s\n' \"\${status}\" | tee -a '${LOG_PATH}'; \
  printf '%s\n' \"\${status}\" > '${LOG_PATH}.exit_code'; \
  exit \"\${status}\"
"

echo "session=${SESSION}"
echo "physical_gpu=${PHYSICAL_GPU}"
echo "gpu_free_mib=${FREE_MIB}"
echo "gpu_util_percent=${GPU_UTIL}"
echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "log_path=${LOG_PATH}"
echo "python_bin=${PYTHON_BIN}"
