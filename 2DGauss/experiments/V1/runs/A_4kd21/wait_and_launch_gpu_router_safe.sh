#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/A_4kd21"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

WAIT_LOG="${LOG_DIR}/wait_gpu_router_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-3}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-12288}"
MAX_GPU_UTIL="${TWODGAUSS_MAX_GPU_UTIL:-100}"
POLL_SECONDS="${TWODGAUSS_POLL_SECONDS:-120}"
SMOKE_OUTPUT="${RUN_DIR}/metrics/a08_a15_gpu_router_offset200_smoke_val2_test2.json"
SMOKE_LOG="${RUN_DIR}/logs/a08_a15_gpu_router_offset200_smoke_val2_test2.stdout.log"
FULL_OUTPUT="${RUN_DIR}/metrics/a08_a15_gpu_router_offset200_val200_test200.json"
FULL_LOG="${RUN_DIR}/logs/a08_a15_gpu_router_offset200_val200_test200.stdout.log"

case "${PHYSICAL_GPU}" in
  0|3) ;;
  *)
    echo "Refusing to wait: only physical GPU 0 or 3 is approved, got ${PHYSICAL_GPU}" >&2
    exit 1
    ;;
esac

log() {
  printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')" "$*" | tee -a "${WAIT_LOG}"
}

log "A_4kd21 wait queue started physical_gpu=${PHYSICAL_GPU} min_free_mib=${MIN_FREE_MIB} max_gpu_util=${MAX_GPU_UTIL} poll_seconds=${POLL_SECONDS}"

while true; do
  GPU_ROW="$(
    nvidia-smi \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits \
      -i "${PHYSICAL_GPU}" \
      | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 " " $2}'
  )"
  read -r FREE_MIB GPU_UTIL <<< "${GPU_ROW}"
  if [ "${FREE_MIB}" -ge "${MIN_FREE_MIB}" ] && [ "${GPU_UTIL}" -le "${MAX_GPU_UTIL}" ]; then
    log "GPU gate passed free_mib=${FREE_MIB} util=${GPU_UTIL}; running smoke"
    TWODGAUSS_PHYSICAL_GPU="${PHYSICAL_GPU}" \
      TWODGAUSS_MIN_FREE_MIB="${MIN_FREE_MIB}" \
      TWODGAUSS_MAX_GPU_UTIL="${MAX_GPU_UTIL}" \
      TWODGAUSS_VAL_LIMIT=2 \
      TWODGAUSS_TEST_LIMIT=2 \
      TWODGAUSS_ROUTER_OUTPUT="${SMOKE_OUTPUT}" \
      TWODGAUSS_ROUTER_LOG="${SMOKE_LOG}" \
      "${RUN_DIR}/launch_gpu_router_eval.sh"
    log "Smoke completed; running full eval"
    TWODGAUSS_PHYSICAL_GPU="${PHYSICAL_GPU}" \
      TWODGAUSS_MIN_FREE_MIB="${MIN_FREE_MIB}" \
      TWODGAUSS_MAX_GPU_UTIL="${MAX_GPU_UTIL}" \
      TWODGAUSS_VAL_LIMIT=200 \
      TWODGAUSS_TEST_LIMIT=200 \
      TWODGAUSS_ROUTER_OUTPUT="${FULL_OUTPUT}" \
      TWODGAUSS_ROUTER_LOG="${FULL_LOG}" \
      "${RUN_DIR}/launch_gpu_router_eval.sh"
    log "A_4kd21 full eval completed"
    exit 0
  fi
  log "GPU gate refused free_mib=${FREE_MIB} util=${GPU_UTIL}; sleeping ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done
