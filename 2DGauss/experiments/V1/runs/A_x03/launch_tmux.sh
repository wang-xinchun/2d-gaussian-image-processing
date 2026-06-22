#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/2DGauss

SESSION="2dgauss_a_x03_uhddeblur_feature_refinement_plus_edge_weighted_lightedge_5h"
CONFIG="experiments/V1/runs/A_x03/configs/a_x03_medium_uhddeblur_feature_refinement_plus_edge_weighted_lightedge_5h.toml"
LOG_DIR="experiments/V1/runs/A_x03/logs"
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
