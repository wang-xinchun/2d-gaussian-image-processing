#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

SESSION="2dgauss_g_4kd01_smoke02_eval"
CONFIG="experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke02_a08_hard_positive_short.toml"
CHECKPOINT="experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/best_val.pt"
RECORDS="experiments/V1/runs/A_4kd18/metrics/a08_a15_router_val_to_test_cpu.json"
RUN_DIR="experiments/G_RSCN/runs/G_4kd01_smoke02"
LOG_DIR="experiments/G_RSCN/runs/G_4kd01/logs"
METRIC_DIR="${RUN_DIR}/metrics"
mkdir -p "${LOG_DIR}" "${METRIC_DIR}"
LOG_PATH="${LOG_DIR}/eval_smoke02_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"

PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-3}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-8192}"
MAX_GPU_UTIL="${TWODGAUSS_MAX_GPU_UTIL:-20}"
CUDA_MEMORY_FRACTION="${TWODGAUSS_CUDA_MEMORY_FRACTION:-0.35}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-0}"
CONFIRM="${TWODGAUSS_G_4KD01_EVAL_SMOKE02_CONFIRM:-}"

case "${PHYSICAL_GPU}" in
  0|3) ;;
  *)
    echo "Refusing to launch eval: only physical GPU 0 or 3 is approved, got ${PHYSICAL_GPU}" >&2
    exit 1
    ;;
esac

if [[ "${PHYSICAL_GPU}" == *,* ]]; then
  echo "Refusing to launch eval: multi-GPU visibility is forbidden" >&2
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
  echo "Refusing to launch eval: GPU${PHYSICAL_GPU} free memory ${FREE_MIB}MiB < ${MIN_FREE_MIB}MiB" >&2
  exit 1
fi

if [ "${GPU_UTIL}" -gt "${MAX_GPU_UTIL}" ]; then
  echo "Refusing to launch eval: GPU${PHYSICAL_GPU} util ${GPU_UTIL}% > ${MAX_GPU_UTIL}%" >&2
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
  echo "checkpoint=${CHECKPOINT}"
  echo "records=${RECORDS}"
  echo "run_dir=${RUN_DIR}"
  echo "log_path=${LOG_PATH}"
  echo "python_bin=${PYTHON_BIN}"
  exit 0
fi

if [ "${CONFIRM}" != "launch_g_4kd01_smoke02_eval_gpu3" ]; then
  echo "Refusing to launch eval: set TWODGAUSS_G_4KD01_EVAL_SMOKE02_CONFIRM=launch_g_4kd01_smoke02_eval_gpu3" >&2
  exit 1
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
  printf 'run=G_4kd01_smoke02_eval physical_gpu=${PHYSICAL_GPU} gpu_free_mib=${FREE_MIB} gpu_util_percent=${GPU_UTIL} min_free_mib=${MIN_FREE_MIB} max_gpu_util=${MAX_GPU_UTIL} cuda_memory_fraction=${CUDA_MEMORY_FRACTION}\n' | tee '${LOG_PATH}'; \
  '${PYTHON_BIN}' -u scripts/evaluate_task_specific_metrics.py --config '${CONFIG}' --checkpoint '${CHECKPOINT}' --split test --device cuda --crop-size 512 --limit 100 --num-workers 0 --output '${METRIC_DIR}/test_task_metrics_best_val_gpu.json' 2>&1 | tee -a '${LOG_PATH}'; \
  status=\${PIPESTATUS[0]}; if [ \"\${status}\" -ne 0 ]; then printf 'eval_exit_code=%s\n' \"\${status}\" | tee -a '${LOG_PATH}'; printf '%s\n' \"\${status}\" > '${LOG_PATH}.exit_code'; exit \"\${status}\"; fi; \
  '${PYTHON_BIN}' -u scripts/evaluate_a18_record_alignment_v1_t01.py --config '${CONFIG}' --checkpoint '${CHECKPOINT}' --records '${RECORDS}' --split test --device cuda --crop-size 512 --num-workers 0 --output '${METRIC_DIR}/a18_record_alignment_best_val_gpu.json' 2>&1 | tee -a '${LOG_PATH}'; \
  status=\${PIPESTATUS[0]}; if [ \"\${status}\" -ne 0 ]; then printf 'eval_exit_code=%s\n' \"\${status}\" | tee -a '${LOG_PATH}'; printf '%s\n' \"\${status}\" > '${LOG_PATH}.exit_code'; exit \"\${status}\"; fi; \
  '${PYTHON_BIN}' -u scripts/evaluate_input_std_route_regimes_v1_t01.py --config '${CONFIG}' --checkpoint '${CHECKPOINT}' --split test --device cuda --crop-size 512 --limit 100 --num-workers 0 --output '${METRIC_DIR}/input_std_route_regime_best_val_gpu.json' 2>&1 | tee -a '${LOG_PATH}'; \
  status=\${PIPESTATUS[0]}; \
  printf 'eval_exit_code=%s\n' \"\${status}\" | tee -a '${LOG_PATH}'; \
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
echo "checkpoint=${CHECKPOINT}"
echo "records=${RECORDS}"
echo "run_dir=${RUN_DIR}"
echo "log_path=${LOG_PATH}"
echo "python_bin=${PYTHON_BIN}"
