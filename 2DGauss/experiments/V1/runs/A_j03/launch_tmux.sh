#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/2DGauss

SESSION="2dgauss_a_j03_uhddeblur_context_probe"
SCRIPT="experiments/V1/runs/A_j03/launch_when_gpu_free.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" "cd /root/autodl-tmp/2DGauss && bash '$SCRIPT'"

echo "launched session=$SESSION"
echo "launcher=$SCRIPT"
