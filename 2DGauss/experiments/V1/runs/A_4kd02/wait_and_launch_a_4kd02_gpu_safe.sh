#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

WAIT_SESSION="2dgauss_wait_a_4kd02_gpu_safe"
PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-3}"
SLEEP_SECONDS="${TWODGAUSS_WAIT_SLEEP_SECONDS:-120}"
MAX_ATTEMPTS="${TWODGAUSS_WAIT_MAX_ATTEMPTS:-0}"
PREFLIGHT_FIRST="${TWODGAUSS_PREFLIGHT_FIRST:-1}"
CONFIG="experiments/V1/runs/A_4kd02/configs/a_4kd02_short_4kdehaze_dehaze_prior_safe_probe.toml"
RUN_DIR="experiments/V1/runs/A_4kd02"
LOG_DIR="${RUN_DIR}/logs"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-8192}"
MAX_GPU_UTIL="${TWODGAUSS_MAX_GPU_UTIL:-100}"
CUDA_MEMORY_FRACTION="${TWODGAUSS_CUDA_MEMORY_FRACTION:-0.30}"

case "${PHYSICAL_GPU}" in
  0|3) ;;
  *)
    echo "Refusing to queue: only physical GPU 0 or 3 is approved, got ${PHYSICAL_GPU}" >&2
    exit 1
    ;;
esac

if tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${WAIT_SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${WAIT_SESSION}" "
  cd '${PROJECT_ROOT}' && \
  mkdir -p '${LOG_DIR}' && \
  attempt=0; \
  while true; do \
    attempt=\$((attempt + 1)); \
    echo \"[\$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')] A_4kd02 wait attempt \${attempt} on GPU${PHYSICAL_GPU}\"; \
    if TWODGAUSS_DRY_RUN=1 TWODGAUSS_PHYSICAL_GPU='${PHYSICAL_GPU}' TWODGAUSS_MIN_FREE_MIB='${MIN_FREE_MIB}' TWODGAUSS_MAX_GPU_UTIL='${MAX_GPU_UTIL}' TWODGAUSS_CUDA_MEMORY_FRACTION='${CUDA_MEMORY_FRACTION}' experiments/V1/runs/A_4kd02/launch_tmux.sh; then \
      echo \"[\$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')] A_4kd02 live gate passed on GPU${PHYSICAL_GPU}.\"; \
      if [ '${PREFLIGHT_FIRST}' = '1' ]; then \
        preflight_log='${LOG_DIR}/preflight_'\"\$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)\"'.log'; \
        echo \"[\$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')] Starting A_4kd02 formal preflight: \${preflight_log}\"; \
        export CUDA_DEVICE_ORDER=PCI_BUS_ID; \
        export CUDA_VISIBLE_DEVICES='${PHYSICAL_GPU}'; \
        export TWODGAUSS_ALLOWED_PHYSICAL_GPUS='0,3'; \
        export OMP_NUM_THREADS=2; \
        export MKL_NUM_THREADS=2; \
        export OPENBLAS_NUM_THREADS=2; \
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
        export TWODGAUSS_CUDA_MEMORY_FRACTION='${CUDA_MEMORY_FRACTION}'; \
        export PYTHONUNBUFFERED=1; \
        export PYTHONFAULTHANDLER=1; \
        set -o pipefail; \
        '${PYTHON_BIN}' -u scripts/train_v1_t01.py --config '${CONFIG}' --preflight --preflight-train-samples 8 2>&1 | tee \"\${preflight_log}\"; \
        preflight_status=\${PIPESTATUS[0]}; \
        printf 'preflight_exit_code=%s\n' \"\${preflight_status}\" | tee -a \"\${preflight_log}\"; \
        if [ \"\${preflight_status}\" != '0' ]; then \
          echo \"A_4kd02 preflight failed; waiter exits without training launch.\"; \
          exit \"\${preflight_status}\"; \
        fi; \
      fi; \
      if TWODGAUSS_DRY_RUN=0 TWODGAUSS_PREFLIGHT_ONLY=0 TWODGAUSS_PHYSICAL_GPU='${PHYSICAL_GPU}' TWODGAUSS_MIN_FREE_MIB='${MIN_FREE_MIB}' TWODGAUSS_MAX_GPU_UTIL='${MAX_GPU_UTIL}' TWODGAUSS_CUDA_MEMORY_FRACTION='${CUDA_MEMORY_FRACTION}' TWODGAUSS_A_4KD02_CONFIRM=launch_a_4kd02_gpu_safe experiments/V1/runs/A_4kd02/launch_tmux.sh; then \
        echo \"[\$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')] A_4kd02 training launched.\"; \
        exit 0; \
      fi; \
    fi; \
    if [ '${MAX_ATTEMPTS}' != '0' ] && [ \${attempt} -ge '${MAX_ATTEMPTS}' ]; then \
      echo \"Reached max attempts ${MAX_ATTEMPTS}; exiting without launch.\"; \
      exit 2; \
    fi; \
    sleep '${SLEEP_SECONDS}'; \
  done
"

echo "wait_session=${WAIT_SESSION}"
echo "physical_gpu=${PHYSICAL_GPU}"
echo "sleep_seconds=${SLEEP_SECONDS}"
echo "max_attempts=${MAX_ATTEMPTS}"
echo "preflight_first=${PREFLIGHT_FIRST}"
