#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/root/autodl-tmp/2DGauss"
RUN_DIR="$ROOT_DIR/experiments/V1/runs/A03"
LOG_DIR="$RUN_DIR/logs"
SESSION_NAME="2dgauss_a03_uhddeblur"
CONFIG_PATH="$ROOT_DIR/experiments/V1/templates/T01/configs/a03_short_uhddeblur_patch.toml"

mkdir -p "$LOG_DIR"
TIMESTAMP="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
LOG_PATH="$LOG_DIR/train_${TIMESTAMP}.log"

cd "$ROOT_DIR"
tmux new-session -d -s "$SESSION_NAME" "python3 scripts/train_v1_t01.py --config $CONFIG_PATH 2>&1 | tee $LOG_PATH"

echo "session=$SESSION_NAME"
echo "log=$LOG_PATH"
