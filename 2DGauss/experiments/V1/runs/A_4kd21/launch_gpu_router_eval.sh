#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/A_4kd21"
CONFIG="${RUN_DIR}/configs/a_4kd21_a08_a15_gpu_router_offset200_larger_probe.toml"
CONFIG_A="experiments/V1/runs/A_4kd08/configs/a_4kd08_fast_4kdehaze_medium_detail_protected_blended_demand_control_probe.toml"
CHECKPOINT_A="experiments/V1/runs/A_4kd08/checkpoints/best_val.pt"
CONFIG_B="experiments/V1/runs/A_4kd15/configs/a_4kd15_fast_4kdehaze_a08_best_no_haze_guard_ema995_finetune.toml"
CHECKPOINT_B="experiments/V1/runs/A_4kd15/checkpoints/best_val.pt"
LOG_DIR="${RUN_DIR}/logs"
METRICS_DIR="${RUN_DIR}/metrics"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}"

OUTPUT="${TWODGAUSS_ROUTER_OUTPUT:-${METRICS_DIR}/a08_a15_gpu_router_offset200_val200_test200.json}"
LOG_PATH="${TWODGAUSS_ROUTER_LOG:-${LOG_DIR}/a08_a15_gpu_router_offset200_val200_test200.stdout.log}"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"
PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-3}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-12288}"
MAX_GPU_UTIL="${TWODGAUSS_MAX_GPU_UTIL:-100}"
CUDA_MEMORY_FRACTION="${TWODGAUSS_CUDA_MEMORY_FRACTION:-0.30}"
VAL_LIMIT="${TWODGAUSS_VAL_LIMIT:-200}"
TEST_LIMIT="${TWODGAUSS_TEST_LIMIT:-200}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-0}"

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

COMMAND=(
  "${PYTHON_BIN}" -u scripts/evaluate_checkpoint_router_v1_t01.py
  --dataset-config "${CONFIG}"
  --config-a "${CONFIG_A}"
  --checkpoint-a "${CHECKPOINT_A}"
  --label-a A_4kd08
  --config-b "${CONFIG_B}"
  --checkpoint-b "${CHECKPOINT_B}"
  --label-b A_4kd15
  --device cuda
  --crop-size 512
  --val-limit "${VAL_LIMIT}"
  --test-limit "${TEST_LIMIT}"
  --num-workers 0
  --fixed-rule-feature input_std
  --fixed-rule-direction lte
  --fixed-rule-threshold 0.07026463374495506
  --output "${OUTPUT}"
)

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  echo "physical_gpu=${PHYSICAL_GPU}"
  echo "gpu_free_mib=${FREE_MIB}"
  echo "gpu_util_percent=${GPU_UTIL}"
  echo "min_free_mib=${MIN_FREE_MIB}"
  echo "max_gpu_util=${MAX_GPU_UTIL}"
  echo "cuda_memory_fraction=${CUDA_MEMORY_FRACTION}"
  echo "val_limit=${VAL_LIMIT}"
  echo "test_limit=${TEST_LIMIT}"
  echo "output=${OUTPUT}"
  echo "log_path=${LOG_PATH}"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export TWODGAUSS_ALLOWED_PHYSICAL_GPUS="0,3"
export TWODGAUSS_CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

{
  printf 'run=A_4kd21 physical_gpu=%s gpu_free_mib=%s gpu_util_percent=%s min_free_mib=%s max_gpu_util=%s cuda_memory_fraction=%s val_limit=%s test_limit=%s\n' \
    "${PHYSICAL_GPU}" "${FREE_MIB}" "${GPU_UTIL}" "${MIN_FREE_MIB}" "${MAX_GPU_UTIL}" "${CUDA_MEMORY_FRACTION}" "${VAL_LIMIT}" "${TEST_LIMIT}"
  "${COMMAND[@]}"
  status=$?
  printf 'eval_exit_code=%s\n' "${status}"
  exit "${status}"
} 2>&1 | tee "${LOG_PATH}"
