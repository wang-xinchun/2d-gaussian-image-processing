#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/2DGauss

SESSION="2dgauss_a_b03_uhddeblur_fullval"
CONFIG="experiments/V1/templates/T01/configs/a_b03_short_uhddeblur_fullval.toml"
LOG_DIR="experiments/V1/runs/A_b03/logs"
mkdir -p "$LOG_DIR"

STAMP="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/train_${STAMP}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /root/autodl-tmp/2DGauss && python3 scripts/train_v1_t01.py --config '$CONFIG' 2>&1 | tee '$LOG_FILE'"

echo "launched session=$SESSION"
echo "config=$CONFIG"
echo "log=$LOG_FILE"
