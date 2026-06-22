#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

SLEEP_SECONDS="${TWODGAUSS_WAIT_SLEEP_SECONDS:-120}"
GPU_CANDIDATES="${TWODGAUSS_GPU_CANDIDATES:-3 0}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-12288}"
MAX_GPU_UTIL="${TWODGAUSS_MAX_GPU_UTIL:-100}"

while true; do
  for PHYSICAL_GPU in ${GPU_CANDIDATES}; do
    if TWODGAUSS_DRY_RUN=1 \
      TWODGAUSS_PHYSICAL_GPU="${PHYSICAL_GPU}" \
      TWODGAUSS_MIN_FREE_MIB="${MIN_FREE_MIB}" \
      TWODGAUSS_MAX_GPU_UTIL="${MAX_GPU_UTIL}" \
      ./experiments/V1/runs/A_4kd17/launch_tmux.sh >/tmp/a_4kd17_gpu_gate.out 2>/tmp/a_4kd17_gpu_gate.err; then
      TWODGAUSS_A_4KD17_CONFIRM=launch_a_4kd17_gpu_safe \
        TWODGAUSS_PHYSICAL_GPU="${PHYSICAL_GPU}" \
        TWODGAUSS_MIN_FREE_MIB="${MIN_FREE_MIB}" \
        TWODGAUSS_MAX_GPU_UTIL="${MAX_GPU_UTIL}" \
        ./experiments/V1/runs/A_4kd17/launch_tmux.sh
      exit $?
    fi
    TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST'
    echo "A_4kd17 gate failed on GPU${PHYSICAL_GPU}:"
    cat /tmp/a_4kd17_gpu_gate.err || true
  done
  sleep "${SLEEP_SECONDS}"
done
