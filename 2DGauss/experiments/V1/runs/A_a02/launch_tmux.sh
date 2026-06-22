#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/2DGauss

SESSION="2dgauss_a_a02_uhdhaze"
CONFIG="experiments/V1/templates/T01/configs/a_a02_short_uhdhaze_patch_iobalanced.toml"
RUN_DIR="experiments/V1/runs/A_a02"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 1
fi

tmux new-session -d -s "${SESSION}" "cd /root/autodl-tmp/2DGauss && set -o pipefail && python3 scripts/train_v1_t01.py --config '${CONFIG}' 2>&1 | tee '${LOG_PATH}'"

echo "session=${SESSION}"
echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "log_path=${LOG_PATH}"
