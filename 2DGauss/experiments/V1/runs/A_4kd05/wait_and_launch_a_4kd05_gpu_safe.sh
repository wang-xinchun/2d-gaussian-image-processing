#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-3}"
SLEEP_SECONDS="${TWODGAUSS_WAIT_SLEEP_SECONDS:-120}"

case "${PHYSICAL_GPU}" in
  0|3) ;;
  *)
    echo "Refusing to wait/launch: only physical GPU 0 or 3 is approved, got ${PHYSICAL_GPU}" >&2
    exit 1
    ;;
esac

while true; do
  if TWODGAUSS_DRY_RUN=1 TWODGAUSS_PHYSICAL_GPU="${PHYSICAL_GPU}" \
    ./experiments/V1/runs/A_4kd05/launch_tmux.sh >/tmp/a_4kd05_launch_dry_run.txt 2>&1; then
    echo "A_4kd05 gate passed for physical GPU${PHYSICAL_GPU}; launching preflight."
    TWODGAUSS_A_4KD05_CONFIRM=launch_a_4kd05_gpu_safe \
      TWODGAUSS_PREFLIGHT_ONLY=1 \
      TWODGAUSS_PHYSICAL_GPU="${PHYSICAL_GPU}" \
      ./experiments/V1/runs/A_4kd05/launch_tmux.sh
    exit 0
  fi
  TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST'
  cat /tmp/a_4kd05_launch_dry_run.txt
  echo "A_4kd05 gate not open; sleeping ${SLEEP_SECONDS}s."
  sleep "${SLEEP_SECONDS}"
done
