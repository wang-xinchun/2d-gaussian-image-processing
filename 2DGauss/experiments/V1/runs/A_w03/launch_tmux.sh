#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/2DGauss

SESSION="2dgauss_a_w03_uhddeblur_feature_refinement_plus_edge_weighted_rebalanced_probe"
CONFIG="experiments/V1/runs/A_w03/configs/a_w03_short_uhddeblur_feature_refinement_plus_edge_weighted_rebalanced_probe.toml"
LOG_DIR="experiments/V1/runs/A_w03/logs"
mkdir -p "$LOG_DIR"

STAMP="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/train_${STAMP}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /root/autodl-tmp/2DGauss && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/train_v1_t01.py --config '$CONFIG' 2>&1 | tee '$LOG_FILE'"

echo "launched session=$SESSION"
echo "config=$CONFIG"
echo "log=$LOG_FILE"
