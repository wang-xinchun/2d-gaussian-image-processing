#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

SESSION="2dgauss_a_b02_ddp_0_3_bs12_uhdhaze_patch_cache"
CONFIG="experiments/V1/templates/T01/configs/a_b02_ddp_0_3_bs12_uhdhaze_patch_cache.toml"
RUN_DIR="experiments/V1/runs/A_b02_ddp_0_3_bs12"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/train_ddp_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"
CLASSIFICATION="uhdhaze_resource_calibration_only_not_current_deshake_task"

PHYSICAL_GPUS="${TWODGAUSS_PHYSICAL_GPUS:-0,3}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-30000}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-0}"
ALLOW_CALIBRATION="${TWODGAUSS_ALLOW_UHDHAZE_CALIBRATION:-0}"

if [ "${PHYSICAL_GPUS}" != "0,3" ]; then
  echo "Refusing to launch: this distributed package is GPU0+GPU3-only, got ${PHYSICAL_GPUS}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${PHYSICAL_GPUS}"
if [ "${#GPU_IDS[@]}" -ne 2 ]; then
  echo "Refusing to launch: expected exactly two GPUs, got ${PHYSICAL_GPUS}" >&2
  exit 1
fi

for gpu in "${GPU_IDS[@]}"; do
  GPU_ROW="$(
    nvidia-smi \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits \
      -i "${gpu}" \
      | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 " " $2}'
  )"
  read -r FREE_MIB GPU_UTIL <<< "${GPU_ROW}"
  if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
    echo "Refusing to launch: GPU${gpu} free memory ${FREE_MIB}MiB < ${MIN_FREE_MIB}MiB" >&2
    nvidia-smi >&2
    exit 1
  fi
  echo "gpu${gpu}_free_mib=${FREE_MIB}"
  echo "gpu${gpu}_util_percent=${GPU_UTIL}"
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  echo "would_launch_session=${SESSION}"
  echo "physical_gpus=${PHYSICAL_GPUS}"
  echo "min_free_mib=${MIN_FREE_MIB}"
  echo "config=${CONFIG}"
  echo "run_dir=${RUN_DIR}"
  echo "log_path=${LOG_PATH}"
  echo "python_bin=${PYTHON_BIN}"
  echo "classification=${CLASSIFICATION}"
  exit 0
fi

if [ "${ALLOW_CALIBRATION}" != "1" ]; then
  echo "Refusing to launch: ${SESSION} is an UHD_haze resource-calibration package, not the current UHD video deshaking task." >&2
  echo "Set TWODGAUSS_ALLOW_UHDHAZE_CALIBRATION=1 only if you explicitly want calibration, not deshaking training." >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" "
  cd '${PROJECT_ROOT}' && \
  export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
  export CUDA_VISIBLE_DEVICES='${PHYSICAL_GPUS}' && \
  export TWODGAUSS_ALLOWED_PHYSICAL_GPUS='0,3' && \
  export TWODGAUSS_ALLOW_MULTI_GPU_VISIBLE=1 && \
  export OMP_NUM_THREADS=2 && \
  export MKL_NUM_THREADS=2 && \
  export OPENBLAS_NUM_THREADS=2 && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  export PYTHONUNBUFFERED=1 && \
  export PYTHONFAULTHANDLER=1 && \
  export NCCL_ASYNC_ERROR_HANDLING=1 && \
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && \
  set -o pipefail; \
  printf 'classification=%s\n' '${CLASSIFICATION}' | tee '${LOG_PATH}'; \
  '${PYTHON_BIN}' -u -m torch.distributed.run --standalone --nproc_per_node=2 --max-restarts=0 scripts/train_v1_t01.py --config '${CONFIG}' 2>&1 | tee -a '${LOG_PATH}'; \
  status=\${PIPESTATUS[0]}; \
  printf 'train_exit_code=%s\n' \"\${status}\" | tee -a '${LOG_PATH}'; \
  printf '%s\n' \"\${status}\" > '${LOG_PATH}.exit_code'; \
  exit \"\${status}\"
"

echo "session=${SESSION}"
echo "physical_gpus=${PHYSICAL_GPUS}"
echo "min_free_mib=${MIN_FREE_MIB}"
echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "log_path=${LOG_PATH}"
echo "python_bin=${PYTHON_BIN}"
echo "classification=${CLASSIFICATION}"
echo "allow_calibration=${ALLOW_CALIBRATION}"
