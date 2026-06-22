#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/2DGauss

SESSION="2dgauss_a_ac03_then_a_ad03_uhddeblur_serial_short_scout"
A_AC03_CONFIG="experiments/V1/runs/A_ac03/configs/a_ac03_short_uhddeblur_confidence_normalized_feature_refinement_probe.toml"
A_AD03_CONFIG="experiments/V1/runs/A_ad03/configs/a_ad03_short_uhddeblur_shared_prior_coupled_gaussian_probe.toml"
A_AC03_LOG_DIR="experiments/V1/runs/A_ac03/logs"
A_AD03_LOG_DIR="experiments/V1/runs/A_ad03/logs"
CHAIN_LOG_DIR="experiments/V1/runs/serial_logs"
mkdir -p "$A_AC03_LOG_DIR" "$A_AD03_LOG_DIR" "$CHAIN_LOG_DIR"

A_AC03_STAMP="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
A_AC03_LOG_FILE="$A_AC03_LOG_DIR/train_${A_AC03_STAMP}.log"
A_AD03_LOG_FILE="$A_AD03_LOG_DIR/train_${A_AC03_STAMP}_after_ac03.log"
CHAIN_LOG_FILE="$CHAIN_LOG_DIR/serial_a_ac03_then_a_ad03_${A_AC03_STAMP}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" "
  cd /root/autodl-tmp/2DGauss && \
  export OMP_NUM_THREADS=8 && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  {
    echo '[serial] start A_ac03';
    python3 scripts/train_v1_t01.py --config '$A_AC03_CONFIG' 2>&1 | tee '$A_AC03_LOG_FILE';
    echo '[serial] A_ac03 finished';
    echo '[serial] start A_ad03';
    python3 scripts/train_v1_t01.py --config '$A_AD03_CONFIG' 2>&1 | tee '$A_AD03_LOG_FILE';
    echo '[serial] A_ad03 finished';
  } 2>&1 | tee '$CHAIN_LOG_FILE'
"

echo "launched session=$SESSION"
echo "a_ac03_config=$A_AC03_CONFIG"
echo "a_ac03_log=$A_AC03_LOG_FILE"
echo "a_ad03_config=$A_AD03_CONFIG"
echo "a_ad03_log=$A_AD03_LOG_FILE"
echo "chain_log=$CHAIN_LOG_FILE"
