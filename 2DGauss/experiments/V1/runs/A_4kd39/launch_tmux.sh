#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

SESSION="2dgauss_a_4kd39_a37_teacher_control_refine"
CONFIG="experiments/V1/runs/A_4kd39/configs/a_4kd39_fast_4kdehaze_a37_teacher_control_refine.toml"
RUN_DIR="experiments/V1/runs/A_4kd39"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/train_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"

PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-3}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-8192}"
MAX_GPU_UTIL="${TWODGAUSS_MAX_GPU_UTIL:-100}"
CUDA_MEMORY_FRACTION="${TWODGAUSS_CUDA_MEMORY_FRACTION:-0.35}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-0}"
PREFLIGHT_ONLY="${TWODGAUSS_PREFLIGHT_ONLY:-0}"
PREFLIGHT_TRAIN_SAMPLES="${TWODGAUSS_PREFLIGHT_TRAIN_SAMPLES:-4}"
CONFIRM="${TWODGAUSS_A_4KD39_CONFIRM:-}"

case "${PHYSICAL_GPU}" in
  0|3) ;;
  *)
    echo "Refusing to launch: only physical GPU 0 or 3 is approved, got ${PHYSICAL_GPU}" >&2
    exit 1
    ;;
esac

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
  exit 1
fi

if [ "${GPU_UTIL}" -gt "${MAX_GPU_UTIL}" ]; then
  echo "Refusing to launch: GPU${PHYSICAL_GPU} util ${GPU_UTIL}% > ${MAX_GPU_UTIL}%" >&2
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
  echo "min_free_mib=${MIN_FREE_MIB}"
  echo "max_gpu_util=${MAX_GPU_UTIL}"
  echo "cuda_memory_fraction=${CUDA_MEMORY_FRACTION}"
  echo "config=${CONFIG}"
  echo "run_dir=${RUN_DIR}"
  echo "log_path=${LOG_PATH}"
  echo "python_bin=${PYTHON_BIN}"
  echo "preflight_only=${PREFLIGHT_ONLY}"
  echo "preflight_train_samples=${PREFLIGHT_TRAIN_SAMPLES}"
  exit 0
fi

if [ "${CONFIRM}" != "launch_a_4kd39_gpu_safe" ]; then
  echo "Refusing to launch: set TWODGAUSS_A_4KD39_CONFIRM=launch_a_4kd39_gpu_safe" >&2
  exit 1
fi

TRAIN_COMMAND="'${PYTHON_BIN}' -u scripts/train_v1_t01.py --config '${CONFIG}'"
if [ "${PREFLIGHT_ONLY}" = "1" ]; then
  TRAIN_COMMAND="'${PYTHON_BIN}' -u scripts/train_v1_t01.py --config '${CONFIG}' --preflight --preflight-train-samples '${PREFLIGHT_TRAIN_SAMPLES}'"
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
  export TWODGAUSS_CUDA_MEMORY_FRACTION='${CUDA_MEMORY_FRACTION}' && \
  export PYTHONUNBUFFERED=1 && \
  export PYTHONFAULTHANDLER=1 && \
  set -o pipefail; \
  printf 'run=A_4kd39 physical_gpu=${PHYSICAL_GPU} gpu_free_mib=${FREE_MIB} gpu_util_percent=${GPU_UTIL} min_free_mib=${MIN_FREE_MIB} max_gpu_util=${MAX_GPU_UTIL} cuda_memory_fraction=${CUDA_MEMORY_FRACTION} preflight_only=${PREFLIGHT_ONLY} preflight_train_samples=${PREFLIGHT_TRAIN_SAMPLES}\n' | tee '${LOG_PATH}'; \
  ${TRAIN_COMMAND} 2>&1 | tee -a '${LOG_PATH}'; \
  status=\${PIPESTATUS[0]}; \
  printf 'train_exit_code=%s\n' \"\${status}\" | tee -a '${LOG_PATH}'; \
  printf '%s\n' \"\${status}\" > '${LOG_PATH}.exit_code'; \
  exit \"\${status}\"
"

echo "session=${SESSION}"
echo "physical_gpu=${PHYSICAL_GPU}"
echo "gpu_free_mib=${FREE_MIB}"
echo "gpu_util_percent=${GPU_UTIL}"
echo "min_free_mib=${MIN_FREE_MIB}"
echo "max_gpu_util=${MAX_GPU_UTIL}"
echo "cuda_memory_fraction=${CUDA_MEMORY_FRACTION}"
echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "log_path=${LOG_PATH}"
echo "python_bin=${PYTHON_BIN}"
echo "preflight_only=${PREFLIGHT_ONLY}"
echo "preflight_train_samples=${PREFLIGHT_TRAIN_SAMPLES}"
