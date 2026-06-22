#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/deshake_ddp_0_3_pending"
METRICS_DIR="${RUN_DIR}/metrics"
mkdir -p "${METRICS_DIR}"

STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"

DATA_ROOT="${TWODGAUSS_DESHAKE_DATA_ROOT:-/home/ldj/wxc/4K_8K}"
CONFIG="${TWODGAUSS_DESHAKE_CONFIG:-/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh01_predeploy/configs/a_dsh01_predeploy_uhd_video_deshake.toml}"
PHYSICAL_GPUS="${TWODGAUSS_PHYSICAL_GPUS:-0,3}"

MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-30000}"
MAX_GPU_UTILIZATION="${TWODGAUSS_MAX_GPU_UTILIZATION:-20}"
GPU_UTILIZATION_POLICY="${TWODGAUSS_GPU_UTILIZATION_POLICY:-block}"
MAX_PER_RANK_BATCH_SIZE="${TWODGAUSS_MAX_PER_RANK_BATCH_SIZE:-2}"
MAX_GLOBAL_BATCH_SIZE="${TWODGAUSS_MAX_GLOBAL_BATCH_SIZE:-4}"
MAX_NUM_WORKERS="${TWODGAUSS_MAX_NUM_WORKERS:-2}"
MAX_TRAIN_CROP_SIZE="${TWODGAUSS_MAX_TRAIN_CROP_SIZE:-0}"
MAX_VAL_CROP_SIZE="${TWODGAUSS_MAX_VAL_CROP_SIZE:-0}"
MIN_DATA_BYTES="${TWODGAUSS_DESHAKE_MIN_DATA_BYTES:-322122547200}"
MIN_VIDEO_FILES="${TWODGAUSS_DESHAKE_MIN_VIDEO_FILES:-1}"
MIN_FRAME_FILES="${TWODGAUSS_DESHAKE_MIN_FRAME_FILES:-0}"
MIN_TRAIN_PAIRS="${TWODGAUSS_DESHAKE_MIN_TRAIN_PAIRS:-1}"
MIN_VAL_PAIRS="${TWODGAUSS_DESHAKE_MIN_VAL_PAIRS:-1}"
MAX_SCAN_FILES="${TWODGAUSS_DESHAKE_MAX_SCAN_FILES:-500000}"
MIN_FILE_AGE_SECONDS="${TWODGAUSS_DESHAKE_MIN_FILE_AGE_SECONDS:-600}"

DATASET_JSON="${METRICS_DIR}/readiness_dataset_${STAMP}.json"
PREFLIGHT_JSON="${METRICS_DIR}/readiness_preflight_${STAMP}.json"
SUMMARY_JSON="${METRICS_DIR}/readiness_summary_${STAMP}.json"

"${PYTHON_BIN}" scripts/inspect_deshake_dataset.py \
  --root "${DATA_ROOT}" \
  --output "${DATASET_JSON}" \
  --allow-unpaired > /dev/null
DATASET_STATUS=$?

set +e
"${PYTHON_BIN}" scripts/preflight_deshake_ddp_gpu0_3.py \
  --data-root "${DATA_ROOT}" \
  --config "${CONFIG}" \
  --physical-gpus "${PHYSICAL_GPUS}" \
  --min-free-mib "${MIN_FREE_MIB}" \
  --max-gpu-utilization "${MAX_GPU_UTILIZATION}" \
  --gpu-utilization-policy "${GPU_UTILIZATION_POLICY}" \
  --max-per-rank-batch-size "${MAX_PER_RANK_BATCH_SIZE}" \
  --max-global-batch-size "${MAX_GLOBAL_BATCH_SIZE}" \
  --max-num-workers "${MAX_NUM_WORKERS}" \
  --max-train-crop-size "${MAX_TRAIN_CROP_SIZE}" \
  --max-val-crop-size "${MAX_VAL_CROP_SIZE}" \
  --min-data-bytes "${MIN_DATA_BYTES}" \
  --min-video-files "${MIN_VIDEO_FILES}" \
  --min-frame-files "${MIN_FRAME_FILES}" \
  --min-train-pairs "${MIN_TRAIN_PAIRS}" \
  --min-val-pairs "${MIN_VAL_PAIRS}" \
  --max-scan-files "${MAX_SCAN_FILES}" \
  --min-file-age-seconds "${MIN_FILE_AGE_SECONDS}" \
  --json > "${PREFLIGHT_JSON}"
PREFLIGHT_STATUS=$?
set -e

set +e
"${PYTHON_BIN}" - \
  "${DATASET_JSON}" \
  "${PREFLIGHT_JSON}" \
  "${SUMMARY_JSON}" \
  "${DATASET_STATUS}" \
  "${PREFLIGHT_STATUS}" \
  "${DATA_ROOT}" \
  "${CONFIG}" \
  "${PHYSICAL_GPUS}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
preflight_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
dataset_status = int(sys.argv[4])
preflight_status = int(sys.argv[5])
data_root = sys.argv[6]
config = sys.argv[7]
physical_gpus = sys.argv[8]

dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
issues = list(preflight.get("issues", []))
warnings = []

dataset_inspection_ready = dataset.get("status") == "ok"
preflight_ready = preflight_status == 0 and bool(preflight.get("ok"))
ready_for_short_ddp_smoke = preflight_ready

if not dataset_inspection_ready and dataset.get("best_pair_candidate") is None:
    message = "dataset inspection found no paired frame candidate"
    if preflight_ready:
        warnings.append(message)
    else:
        issues.append(message)
elif not dataset_inspection_ready:
    message = f"dataset inspection status is {dataset.get('status')!r}"
    if preflight_ready:
        warnings.append(message)
    else:
        issues.append(message)

summary = {
    "ready_for_short_ddp_smoke": ready_for_short_ddp_smoke,
    "formal_launch_approved": False,
    "data_root": data_root,
    "config": config,
    "physical_gpus": physical_gpus,
    "dataset_status": dataset.get("status"),
    "dataset_inspection_ready": dataset_inspection_ready,
    "dataset_video_count": dataset.get("video_count"),
    "dataset_image_count": dataset.get("image_count"),
    "dataset_best_pair_candidate": dataset.get("best_pair_candidate"),
    "preflight_status": preflight_status,
    "preflight_ok": preflight.get("ok"),
    "preflight_dataset_bytes": preflight.get("dataset", {}).get("bytes_scanned"),
    "preflight_recent_file_count": preflight.get("dataset", {}).get("recent_file_count"),
    "preflight_latest_mtime_age_seconds": preflight.get("dataset", {}).get(
        "latest_mtime_age_seconds"
    ),
    "issues": issues,
    "warnings": warnings,
    "artifacts": {
        "dataset_json": str(dataset_path),
        "preflight_json": str(preflight_path),
        "summary_json": str(summary_path),
    },
    "next_action": (
        "ready for an explicit short DDP smoke approval"
        if ready_for_short_ddp_smoke
        else "do not launch; resolve issues first"
    ),
}

text = json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
summary_path.write_text(text + "\n", encoding="utf-8")
print(text)
raise SystemExit(0 if ready_for_short_ddp_smoke else 3)
PY
SUMMARY_STATUS=$?
set -e

if [ "${PREFLIGHT_STATUS}" -ne 0 ]; then
  exit "${PREFLIGHT_STATUS}"
fi

if [ "${DATASET_STATUS}" -ne 0 ]; then
  exit "${DATASET_STATUS}"
fi

exit "${SUMMARY_STATUS}"
