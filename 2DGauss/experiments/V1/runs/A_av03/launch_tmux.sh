#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/2DGauss
mkdir -p experiments/V1/runs/A_av03/logs
LOG_PATH="experiments/V1/runs/A_av03/logs/train_$(date +%Y%m%d_%H%M%S).log"
python3 scripts/train_v1_t01.py \
  --config experiments/V1/runs/A_av03/configs/a_av03_short_uhddeblur_demand_budget_selective_feedback_probe.toml \
  2>&1 | tee "${LOG_PATH}"
