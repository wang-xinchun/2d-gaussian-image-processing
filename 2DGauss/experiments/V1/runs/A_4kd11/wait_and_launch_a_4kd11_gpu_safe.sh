#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

SLEEP_SECONDS="${TWODGAUSS_WAIT_SLEEP_SECONDS:-120}"
PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-3}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-8192}"
MAX_GPU_UTIL="${TWODGAUSS_MAX_GPU_UTIL:-100}"

while true; do
  if TWODGAUSS_DRY_RUN=1 \
    TWODGAUSS_PHYSICAL_GPU="${PHYSICAL_GPU}" \
    TWODGAUSS_MIN_FREE_MIB="${MIN_FREE_MIB}" \
    TWODGAUSS_MAX_GPU_UTIL="${MAX_GPU_UTIL}" \
    ./experiments/V1/runs/A_4kd11/launch_tmux.sh >/tmp/a_4kd11_gpu_gate.out 2>/tmp/a_4kd11_gpu_gate.err; then
    TWODGAUSS_A_4KD11_CONFIRM=launch_a_4kd11_gpu_safe \
      TWODGAUSS_PHYSICAL_GPU="${PHYSICAL_GPU}" \
      TWODGAUSS_MIN_FREE_MIB="${MIN_FREE_MIB}" \
      TWODGAUSS_MAX_GPU_UTIL="${MAX_GPU_UTIL}" \
      ./experiments/V1/runs/A_4kd11/launch_tmux.sh
    exit $?
  fi
  TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST'
  cat /tmp/a_4kd11_gpu_gate.err || true
  sleep "${SLEEP_SECONDS}"
done
