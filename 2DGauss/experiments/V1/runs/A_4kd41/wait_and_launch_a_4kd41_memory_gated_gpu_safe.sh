#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-3}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-8192}"
MAX_GPU_UTIL="${TWODGAUSS_MAX_GPU_UTIL:-100}"
SLEEP_SECONDS="${TWODGAUSS_WAIT_SLEEP_SECONDS:-60}"
MAX_WAIT_SECONDS="${TWODGAUSS_MAX_WAIT_SECONDS:-0}"
START_TS="$(date +%s)"

while true; do
  GPU_ROW="$(
    nvidia-smi \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits \
      -i "${PHYSICAL_GPU}" \
      | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 " " $2}'
  )"
  read -r FREE_MIB GPU_UTIL <<< "${GPU_ROW}"
  NOW="$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')"
  echo "${NOW} gpu=${PHYSICAL_GPU} free_mib=${FREE_MIB} util_percent=${GPU_UTIL} min_free_mib=${MIN_FREE_MIB} max_gpu_util=${MAX_GPU_UTIL}"

  if [ "${FREE_MIB}" -ge "${MIN_FREE_MIB}" ] && [ "${GPU_UTIL}" -le "${MAX_GPU_UTIL}" ]; then
    exec ./experiments/V1/runs/A_4kd41/launch_tmux.sh
  fi

  if [ "${MAX_WAIT_SECONDS}" -gt 0 ]; then
    ELAPSED=$(( $(date +%s) - START_TS ))
    if [ "${ELAPSED}" -ge "${MAX_WAIT_SECONDS}" ]; then
      echo "Timed out waiting for GPU${PHYSICAL_GPU}" >&2
      exit 1
    fi
  fi

  sleep "${SLEEP_SECONDS}"
done
