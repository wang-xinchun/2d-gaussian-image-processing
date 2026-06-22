#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/2DGauss

RUN_DIR="experiments/V1/runs/A_j03"
CONFIG="experiments/V1/templates/T01/configs/a_j03_short_uhddeblur_tlc_context864_probe.toml"
LOG_DIR="$RUN_DIR/logs"
METRIC_DIR="$RUN_DIR/metrics"
mkdir -p "$LOG_DIR" "$METRIC_DIR"

STAMP="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
QUEUE_LOG="$LOG_DIR/queue_${STAMP}.log"
exec > >(tee -a "$QUEUE_LOG") 2>&1

echo "[queue] started at $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')"
echo "[queue] config=$CONFIG"

wait_for_clean_gpu() {
  while true; do
    local used_mib
    local proc_count
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    proc_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
    if [ "${used_mib:-999999}" -le 1024 ] && [ "${proc_count:-0}" -eq 0 ]; then
      echo "[queue] clean gpu detected: used=${used_mib}MiB compute_procs=${proc_count}"
      return 0
    fi
    echo "[queue] waiting for gpu release: used=${used_mib}MiB compute_procs=${proc_count}"
    sleep 60
  done
}

wait_for_clean_gpu

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SMOKE_JSON="$METRIC_DIR/smoke_${STAMP}.json"
echo "[smoke] running batch=6 context=864"
python3 scripts/smoke_calibrate_v1_t01.py \
  --config "$CONFIG" \
  --batch-size 6 \
  --crop-size 864 \
  --steps 20 \
  --num-workers 2 > "$SMOKE_JSON"
cat "$SMOKE_JSON"

python3 - "$SMOKE_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

if data.get("status") != "ok":
    raise SystemExit(f"smoke failed: {data}")

if not data.get("within_target_reserved_memory_window", False):
    raise SystemExit(
        "smoke outside 85-93G window: "
        f"reserved={data.get('max_memory_reserved_gb')}, "
        f"window=({data.get('target_reserved_memory_gb_min')}, {data.get('target_reserved_memory_gb_max')})"
    )
PY

PREFLIGHT_LOG="$LOG_DIR/preflight_${STAMP}.log"
echo "[preflight] running"
python3 scripts/train_v1_t01.py \
  --config "$CONFIG" \
  --preflight \
  --preflight-train-samples 6 2>&1 | tee "$PREFLIGHT_LOG"

TRAIN_LOG="$LOG_DIR/train_${STAMP}.log"
echo "[train] launching"
python3 scripts/train_v1_t01.py --config "$CONFIG" 2>&1 | tee "$TRAIN_LOG"
