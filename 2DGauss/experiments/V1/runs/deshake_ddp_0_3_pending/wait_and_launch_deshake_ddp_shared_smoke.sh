#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/deshake_ddp_0_3_pending"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"
READINESS_SCRIPT="${RUN_DIR}/check_deshake_ddp_readiness.sh"
SHARED_SMOKE_WRAPPER="${RUN_DIR}/launch_deshake_ddp_shared_smoke_once.sh"

DATA_ROOT="${TWODGAUSS_DESHAKE_DATA_ROOT:-}"
CONFIG="${TWODGAUSS_DESHAKE_CONFIG:-}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-1}"
CONFIRM_QUEUE="${TWODGAUSS_DESHAKE_CONFIRM_SHARED_QUEUE:-}"
CONFIRM_QUEUE_TOKEN="wait_and_launch_shared_gpu0_3_deshake_ddp_smoke"
MAX_CHECKS="${TWODGAUSS_DESHAKE_SHARED_QUEUE_MAX_CHECKS:-12}"
INTERVAL_SECONDS="${TWODGAUSS_DESHAKE_SHARED_QUEUE_INTERVAL_SECONDS:-300}"

GPU_UTILIZATION_POLICY="${TWODGAUSS_GPU_UTILIZATION_POLICY:-warn}"
CUDA_MEMORY_FRACTION="${TWODGAUSS_CUDA_MEMORY_FRACTION:-0.50}"
MAX_PER_RANK_BATCH_SIZE="${TWODGAUSS_MAX_PER_RANK_BATCH_SIZE:-12}"
MAX_GLOBAL_BATCH_SIZE="${TWODGAUSS_MAX_GLOBAL_BATCH_SIZE:-24}"
MAX_NUM_WORKERS="${TWODGAUSS_MAX_NUM_WORKERS:-0}"
MAX_TRAIN_CROP_SIZE="${TWODGAUSS_MAX_TRAIN_CROP_SIZE:-512}"
MAX_VAL_CROP_SIZE="${TWODGAUSS_MAX_VAL_CROP_SIZE:-512}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-32000}"
RUNTIME_MIN_FREE_MIB="${TWODGAUSS_RUNTIME_MIN_FREE_MIB:-8192}"
RUNTIME_POLL_SECONDS="${TWODGAUSS_RUNTIME_POLL_SECONDS:-5}"
RUNTIME_STOP_GRACE_SECONDS="${TWODGAUSS_RUNTIME_STOP_GRACE_SECONDS:-20}"

QUEUE_JSONL="${LOG_DIR}/queued_shared_ddp_smoke_attempts_${STAMP}.jsonl"
SUMMARY_JSON="${LOG_DIR}/queued_shared_ddp_smoke_summary_${STAMP}.json"

if [ -z "${DATA_ROOT}" ]; then
  echo "Refusing queued shared DDP smoke: TWODGAUSS_DESHAKE_DATA_ROOT is required." >&2
  exit 1
fi

if [ -z "${CONFIG}" ]; then
  echo "Refusing queued shared DDP smoke: TWODGAUSS_DESHAKE_CONFIG is required." >&2
  exit 1
fi

for value_name in MAX_CHECKS INTERVAL_SECONDS MAX_PER_RANK_BATCH_SIZE MAX_GLOBAL_BATCH_SIZE MAX_NUM_WORKERS MAX_TRAIN_CROP_SIZE MAX_VAL_CROP_SIZE MIN_FREE_MIB RUNTIME_MIN_FREE_MIB RUNTIME_POLL_SECONDS RUNTIME_STOP_GRACE_SECONDS; do
  value="${!value_name}"
  case "${value}" in
    ''|*[!0-9]*)
      echo "Refusing queued shared DDP smoke: ${value_name} must be a non-negative integer." >&2
      exit 1
      ;;
  esac
done

if [ "${MAX_CHECKS}" -lt 1 ]; then
  echo "Refusing queued shared DDP smoke: TWODGAUSS_DESHAKE_SHARED_QUEUE_MAX_CHECKS must be >= 1." >&2
  exit 1
fi

if [ "${MAX_PER_RANK_BATCH_SIZE}" -lt 1 ] || [ "${MAX_GLOBAL_BATCH_SIZE}" -lt 1 ]; then
  echo "Refusing queued shared DDP smoke: shared batch caps must be >= 1." >&2
  exit 1
fi

if [ "${GPU_UTILIZATION_POLICY}" != "warn" ]; then
  echo "Refusing queued shared DDP smoke: TWODGAUSS_GPU_UTILIZATION_POLICY must be warn." >&2
  exit 1
fi

if [ "${DRY_RUN}" != "1" ] && [ "${CONFIRM_QUEUE}" != "${CONFIRM_QUEUE_TOKEN}" ]; then
  echo "Refusing queued shared DDP smoke: set TWODGAUSS_DRY_RUN=1 for dry-run, or set TWODGAUSS_DESHAKE_CONFIRM_SHARED_QUEUE=${CONFIRM_QUEUE_TOKEN} for an explicit queued real shared smoke." >&2
  exit 1
fi

export TWODGAUSS_DESHAKE_DATA_ROOT="${DATA_ROOT}"
export TWODGAUSS_DESHAKE_CONFIG="${CONFIG}"
export TWODGAUSS_DRY_RUN="${DRY_RUN}"
export TWODGAUSS_GPU_UTILIZATION_POLICY="${GPU_UTILIZATION_POLICY}"
export TWODGAUSS_CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION}"
export TWODGAUSS_MAX_PER_RANK_BATCH_SIZE="${MAX_PER_RANK_BATCH_SIZE}"
export TWODGAUSS_MAX_GLOBAL_BATCH_SIZE="${MAX_GLOBAL_BATCH_SIZE}"
export TWODGAUSS_MAX_NUM_WORKERS="${MAX_NUM_WORKERS}"
export TWODGAUSS_MAX_TRAIN_CROP_SIZE="${MAX_TRAIN_CROP_SIZE}"
export TWODGAUSS_MAX_VAL_CROP_SIZE="${MAX_VAL_CROP_SIZE}"
export TWODGAUSS_MIN_FREE_MIB="${MIN_FREE_MIB}"
export TWODGAUSS_RUNTIME_MIN_FREE_MIB="${RUNTIME_MIN_FREE_MIB}"
export TWODGAUSS_RUNTIME_POLL_SECONDS="${RUNTIME_POLL_SECONDS}"
export TWODGAUSS_RUNTIME_STOP_GRACE_SECONDS="${RUNTIME_STOP_GRACE_SECONDS}"

write_record() {
  local output_jsonl="$1"
  local timestamp_bj="$2"
  local attempt="$3"
  local readiness_status="$4"
  local readiness_stdout="$5"
  local launch_status="${6:-}"
  local launch_stdout="${7:-}"
  "${PYTHON_BIN}" - \
    "${output_jsonl}" \
    "${timestamp_bj}" \
    "${attempt}" \
    "${readiness_status}" \
    "${readiness_stdout}" \
    "${launch_status}" \
    "${launch_stdout}" \
    "${DATA_ROOT}" \
    "${CONFIG}" \
    "${DRY_RUN}" \
    "${GPU_UTILIZATION_POLICY}" \
    "${CUDA_MEMORY_FRACTION}" \
    "${MAX_PER_RANK_BATCH_SIZE}" \
    "${MAX_GLOBAL_BATCH_SIZE}" \
    "${MAX_NUM_WORKERS}" \
    "${MIN_FREE_MIB}" \
    "${MAX_TRAIN_CROP_SIZE}" \
    "${MAX_VAL_CROP_SIZE}" \
    "${RUNTIME_MIN_FREE_MIB}" \
    "${RUNTIME_POLL_SECONDS}" \
    "${RUNTIME_STOP_GRACE_SECONDS}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
timestamp_bj = sys.argv[2]
attempt = int(sys.argv[3])
readiness_status = int(sys.argv[4])
readiness_stdout = Path(sys.argv[5])
launch_status_raw = sys.argv[6]
launch_stdout_raw = sys.argv[7]
data_root = sys.argv[8]
config = sys.argv[9]
dry_run = sys.argv[10]
gpu_utilization_policy = sys.argv[11]
cuda_memory_fraction = sys.argv[12]
max_per_rank_batch_size = int(sys.argv[13])
max_global_batch_size = int(sys.argv[14])
max_num_workers = int(sys.argv[15])
min_free_mib = int(sys.argv[16])
max_train_crop_size = int(sys.argv[17])
max_val_crop_size = int(sys.argv[18])
runtime_min_free_mib = int(sys.argv[19])
runtime_poll_seconds = int(sys.argv[20])
runtime_stop_grace_seconds = int(sys.argv[21])

try:
    readiness = json.loads(readiness_stdout.read_text(encoding="utf-8"))
except Exception as exc:  # noqa: BLE001
    readiness = {"parse_error": str(exc), "path": str(readiness_stdout)}

record = {
    "attempt": attempt,
    "config": config,
    "cuda_memory_fraction": cuda_memory_fraction,
    "data_root": data_root,
    "dry_run": dry_run,
    "gpu_utilization_policy": gpu_utilization_policy,
    "max_global_batch_size": max_global_batch_size,
    "max_num_workers": max_num_workers,
    "max_per_rank_batch_size": max_per_rank_batch_size,
    "max_train_crop_size": max_train_crop_size,
    "max_val_crop_size": max_val_crop_size,
    "min_free_mib": min_free_mib,
    "runtime_min_free_mib": runtime_min_free_mib,
    "runtime_poll_seconds": runtime_poll_seconds,
    "runtime_stop_grace_seconds": runtime_stop_grace_seconds,
    "readiness_status": readiness_status,
    "readiness_stdout": str(readiness_stdout),
    "readiness_summary": readiness,
    "ready": readiness_status == 0,
    "timestamp_bj": timestamp_bj,
}
if launch_status_raw:
    record["launch_status"] = int(launch_status_raw)
    record["launch_stdout"] = launch_stdout_raw
jsonl_path.parent.mkdir(parents=True, exist_ok=True)
with jsonl_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

write_summary() {
  local output_json="$1"
  local timestamp_bj="$2"
  local status="$3"
  local reason="$4"
  local attempts="$5"
  local latest_readiness_status="$6"
  local launch_status="${7:-}"
  local launch_stdout="${8:-}"
  "${PYTHON_BIN}" - \
    "${output_json}" \
    "${timestamp_bj}" \
    "${status}" \
    "${reason}" \
    "${attempts}" \
    "${latest_readiness_status}" \
    "${launch_status}" \
    "${launch_stdout}" \
    "${QUEUE_JSONL}" \
    "${DATA_ROOT}" \
    "${CONFIG}" \
    "${DRY_RUN}" \
    "${GPU_UTILIZATION_POLICY}" \
    "${CUDA_MEMORY_FRACTION}" \
    "${MAX_PER_RANK_BATCH_SIZE}" \
    "${MAX_GLOBAL_BATCH_SIZE}" \
    "${MAX_NUM_WORKERS}" \
    "${MIN_FREE_MIB}" \
    "${MAX_TRAIN_CROP_SIZE}" \
    "${MAX_VAL_CROP_SIZE}" \
    "${RUNTIME_MIN_FREE_MIB}" \
    "${RUNTIME_POLL_SECONDS}" \
    "${RUNTIME_STOP_GRACE_SECONDS}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
timestamp_bj = sys.argv[2]
status = sys.argv[3]
reason = sys.argv[4]
attempts = int(sys.argv[5])
latest_readiness_status = int(sys.argv[6])
launch_status_raw = sys.argv[7]
launch_stdout_raw = sys.argv[8]
queue_jsonl = Path(sys.argv[9])
data_root = sys.argv[10]
config = sys.argv[11]
dry_run = sys.argv[12]
gpu_utilization_policy = sys.argv[13]
cuda_memory_fraction = sys.argv[14]
max_per_rank_batch_size = int(sys.argv[15])
max_global_batch_size = int(sys.argv[16])
max_num_workers = int(sys.argv[17])
min_free_mib = int(sys.argv[18])
max_train_crop_size = int(sys.argv[19])
max_val_crop_size = int(sys.argv[20])
runtime_min_free_mib = int(sys.argv[21])
runtime_poll_seconds = int(sys.argv[22])
runtime_stop_grace_seconds = int(sys.argv[23])

summary = {
    "attempts": attempts,
    "config": config,
    "cuda_memory_fraction": cuda_memory_fraction,
    "data_root": data_root,
    "dry_run": dry_run,
    "gpu_utilization_policy": gpu_utilization_policy,
    "latest_readiness_status": latest_readiness_status,
    "max_global_batch_size": max_global_batch_size,
    "max_num_workers": max_num_workers,
    "max_per_rank_batch_size": max_per_rank_batch_size,
    "max_train_crop_size": max_train_crop_size,
    "max_val_crop_size": max_val_crop_size,
    "min_free_mib": min_free_mib,
    "runtime_min_free_mib": runtime_min_free_mib,
    "runtime_poll_seconds": runtime_poll_seconds,
    "runtime_stop_grace_seconds": runtime_stop_grace_seconds,
    "queue_jsonl": str(queue_jsonl),
    "reason": reason,
    "status": status,
    "timestamp_bj": timestamp_bj,
}
if launch_status_raw:
    summary["launch_status"] = int(launch_status_raw)
    summary["launch_stdout"] = launch_stdout_raw
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
PY
}

latest_readiness_status=99
attempt=1
while [ "${attempt}" -le "${MAX_CHECKS}" ]; do
  now_bj="$(TZ=Asia/Shanghai date '+%F %T %Z')"
  attempt_stamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$_${attempt}"
  readiness_stdout="${LOG_DIR}/queued_shared_ddp_smoke_readiness_${attempt_stamp}.json"

  set +e
  "${READINESS_SCRIPT}" > "${readiness_stdout}" 2>&1
  latest_readiness_status=$?
  set -e

  cat "${readiness_stdout}"

  if [ "${latest_readiness_status}" -eq 0 ]; then
    launch_stdout="${LOG_DIR}/queued_shared_ddp_smoke_launch_${attempt_stamp}.log"
    set +e
    "${SHARED_SMOKE_WRAPPER}" > "${launch_stdout}" 2>&1
    launch_status=$?
    set -e
    cat "${launch_stdout}"
    write_record \
      "${QUEUE_JSONL}" \
      "${now_bj}" \
      "${attempt}" \
      "${latest_readiness_status}" \
      "${readiness_stdout}" \
      "${launch_status}" \
      "${launch_stdout}"
    if [ "${DRY_RUN}" = "1" ]; then
      summary_status="dry_run_invoked"
      summary_reason="readiness passed and shared smoke wrapper dry-run was invoked"
    else
      summary_status="shared_smoke_invoked"
      summary_reason="readiness passed and shared smoke wrapper was invoked"
    fi
    write_summary \
      "${SUMMARY_JSON}" \
      "${now_bj}" \
      "${summary_status}" \
      "${summary_reason}" \
      "${attempt}" \
      "${latest_readiness_status}" \
      "${launch_status}" \
      "${launch_stdout}"
    exit "${launch_status}"
  fi

  write_record \
    "${QUEUE_JSONL}" \
    "${now_bj}" \
    "${attempt}" \
    "${latest_readiness_status}" \
    "${readiness_stdout}"

  if [ "${attempt}" -lt "${MAX_CHECKS}" ]; then
    sleep "${INTERVAL_SECONDS}"
  fi
  attempt=$((attempt + 1))
done

now_bj="$(TZ=Asia/Shanghai date '+%F %T %Z')"
write_summary \
  "${SUMMARY_JSON}" \
  "${now_bj}" \
  "not_ready" \
  "readiness did not pass before shared queue max checks" \
  "${MAX_CHECKS}" \
  "${latest_readiness_status}"
exit 3
