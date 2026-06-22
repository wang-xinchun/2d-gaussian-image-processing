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
SMOKE_WRAPPER="${RUN_DIR}/launch_deshake_ddp_smoke_once.sh"

DATA_ROOT="${TWODGAUSS_DESHAKE_DATA_ROOT:-}"
CONFIG="${TWODGAUSS_DESHAKE_CONFIG:-}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-1}"
CONFIRM_QUEUE="${TWODGAUSS_DESHAKE_CONFIRM_QUEUE:-}"
CONFIRM_QUEUE_TOKEN="wait_and_launch_gpu0_3_deshake_smoke"
MAX_CHECKS="${TWODGAUSS_DESHAKE_QUEUE_MAX_CHECKS:-12}"
INTERVAL_SECONDS="${TWODGAUSS_DESHAKE_QUEUE_INTERVAL_SECONDS:-300}"

QUEUE_JSONL="${LOG_DIR}/queued_ddp_smoke_attempts_${STAMP}.jsonl"
SUMMARY_JSON="${LOG_DIR}/queued_ddp_smoke_summary_${STAMP}.json"

if [ -z "${DATA_ROOT}" ]; then
  echo "Refusing queued DDP smoke: TWODGAUSS_DESHAKE_DATA_ROOT is required." >&2
  exit 1
fi

if [ -z "${CONFIG}" ]; then
  echo "Refusing queued DDP smoke: TWODGAUSS_DESHAKE_CONFIG is required." >&2
  exit 1
fi

for value_name in MAX_CHECKS INTERVAL_SECONDS; do
  value="${!value_name}"
  case "${value}" in
    ''|*[!0-9]*)
      echo "Refusing queued DDP smoke: ${value_name} must be a non-negative integer." >&2
      exit 1
      ;;
  esac
done

if [ "${MAX_CHECKS}" -lt 1 ]; then
  echo "Refusing queued DDP smoke: TWODGAUSS_DESHAKE_QUEUE_MAX_CHECKS must be >= 1." >&2
  exit 1
fi

if [ "${DRY_RUN}" != "1" ] && [ "${CONFIRM_QUEUE}" != "${CONFIRM_QUEUE_TOKEN}" ]; then
  echo "Refusing queued DDP smoke: set TWODGAUSS_DRY_RUN=1 for dry-run, or set TWODGAUSS_DESHAKE_CONFIRM_QUEUE=${CONFIRM_QUEUE_TOKEN} for an explicit queued real smoke." >&2
  exit 1
fi

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
    "${DRY_RUN}" <<'PY'
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

def parse_json_file(path: Path) -> dict:
    if not path:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"parse_error": str(exc), "path": str(path)}

readiness = parse_json_file(readiness_stdout)
record = {
    "timestamp_bj": timestamp_bj,
    "attempt": attempt,
    "data_root": data_root,
    "config": config,
    "dry_run": dry_run,
    "readiness_status": readiness_status,
    "readiness_stdout": str(readiness_stdout),
    "ready": readiness_status == 0,
    "readiness_summary": readiness,
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
    "${DRY_RUN}" <<'PY'
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

summary = {
    "timestamp_bj": timestamp_bj,
    "status": status,
    "reason": reason,
    "attempts": attempts,
    "latest_readiness_status": latest_readiness_status,
    "data_root": data_root,
    "config": config,
    "dry_run": dry_run,
    "queue_jsonl": str(queue_jsonl),
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
  readiness_stdout="${LOG_DIR}/queued_ddp_smoke_readiness_${attempt_stamp}.json"

  set +e
  "${READINESS_SCRIPT}" > "${readiness_stdout}" 2>&1
  latest_readiness_status=$?
  set -e

  cat "${readiness_stdout}"

  if [ "${latest_readiness_status}" -eq 0 ]; then
    launch_stdout="${LOG_DIR}/queued_ddp_smoke_launch_${attempt_stamp}.log"
    set +e
    "${SMOKE_WRAPPER}" > "${launch_stdout}" 2>&1
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
      summary_reason="readiness passed and smoke wrapper dry-run was invoked"
    else
      summary_status="smoke_invoked"
      summary_reason="readiness passed and smoke wrapper was invoked"
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
  "readiness did not pass before queue max checks" \
  "${MAX_CHECKS}" \
  "${latest_readiness_status}"
exit 3
