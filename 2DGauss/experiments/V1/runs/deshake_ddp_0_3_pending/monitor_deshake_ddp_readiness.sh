#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/deshake_ddp_0_3_pending"
METRICS_DIR="${RUN_DIR}/metrics"
mkdir -p "${METRICS_DIR}"

STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"
READINESS_SCRIPT="${TWODGAUSS_DESHAKE_READINESS_SCRIPT:-${RUN_DIR}/check_deshake_ddp_readiness.sh}"

INTERVAL_SECONDS="${TWODGAUSS_DESHAKE_MONITOR_INTERVAL_SECONDS:-300}"
MAX_CHECKS="${TWODGAUSS_DESHAKE_MONITOR_MAX_CHECKS:-12}"
STOP_ON_READY="${TWODGAUSS_DESHAKE_MONITOR_STOP_ON_READY:-1}"

MONITOR_JSONL="${METRICS_DIR}/readiness_monitor_${STAMP}.jsonl"
SUMMARY_JSON="${METRICS_DIR}/readiness_monitor_summary_${STAMP}.json"

case "${INTERVAL_SECONDS}" in
  ''|*[!0-9]*)
    echo "Refusing to monitor: TWODGAUSS_DESHAKE_MONITOR_INTERVAL_SECONDS must be a non-negative integer." >&2
    exit 1
    ;;
esac

case "${MAX_CHECKS}" in
  ''|*[!0-9]*)
    echo "Refusing to monitor: TWODGAUSS_DESHAKE_MONITOR_MAX_CHECKS must be a positive integer." >&2
    exit 1
    ;;
esac

if [ "${MAX_CHECKS}" -lt 1 ]; then
  echo "Refusing to monitor: TWODGAUSS_DESHAKE_MONITOR_MAX_CHECKS must be >= 1." >&2
  exit 1
fi

if [ ! -x "${READINESS_SCRIPT}" ]; then
  echo "Refusing to monitor: readiness script is not executable: ${READINESS_SCRIPT}" >&2
  exit 1
fi

echo "monitor_jsonl=${MONITOR_JSONL}"
echo "summary_json=${SUMMARY_JSON}"
echo "readiness_script=${READINESS_SCRIPT}"
echo "max_checks=${MAX_CHECKS}"
echo "interval_seconds=${INTERVAL_SECONDS}"
echo "stop_on_ready=${STOP_ON_READY}"

attempt=1
while [ "${attempt}" -le "${MAX_CHECKS}" ]; do
  started_at="$(TZ=Asia/Shanghai date +%Y-%m-%dT%H:%M:%S%z)"
  output_path="${METRICS_DIR}/readiness_monitor_${STAMP}_attempt_${attempt}.stdout"

  set +e
  "${READINESS_SCRIPT}" > "${output_path}" 2>&1
  readiness_status=$?
  set -e

  ended_at="$(TZ=Asia/Shanghai date +%Y-%m-%dT%H:%M:%S%z)"

  ready="$("${PYTHON_BIN}" - \
    "${output_path}" \
    "${MONITOR_JSONL}" \
    "${attempt}" \
    "${readiness_status}" \
    "${started_at}" \
    "${ended_at}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
monitor_jsonl = Path(sys.argv[2])
attempt = int(sys.argv[3])
readiness_status = int(sys.argv[4])
started_at = sys.argv[5]
ended_at = sys.argv[6]

raw = output_path.read_text(encoding="utf-8", errors="replace")
try:
    summary = json.loads(raw)
except json.JSONDecodeError as exc:
    summary = {
        "ready_for_short_ddp_smoke": False,
        "issues": [f"readiness output was not valid JSON: {exc}"],
        "warnings": [],
        "artifacts": {},
    }

artifacts = summary.get("artifacts", {})
record = {
    "attempt": attempt,
    "started_at_bj_iso": started_at,
    "ended_at_bj_iso": ended_at,
    "readiness_status": readiness_status,
    "ready_for_short_ddp_smoke": bool(summary.get("ready_for_short_ddp_smoke")),
    "formal_launch_approved": bool(summary.get("formal_launch_approved")),
    "next_action": summary.get("next_action"),
    "data_root": summary.get("data_root"),
    "config": summary.get("config"),
    "physical_gpus": summary.get("physical_gpus"),
    "preflight_status": summary.get("preflight_status"),
    "preflight_ok": summary.get("preflight_ok"),
    "preflight_dataset_bytes": summary.get("preflight_dataset_bytes"),
    "preflight_recent_file_count": summary.get("preflight_recent_file_count"),
    "preflight_latest_mtime_age_seconds": summary.get(
        "preflight_latest_mtime_age_seconds"
    ),
    "dataset_status": summary.get("dataset_status"),
    "dataset_video_count": summary.get("dataset_video_count"),
    "dataset_image_count": summary.get("dataset_image_count"),
    "issue_count": len(summary.get("issues", [])),
    "warning_count": len(summary.get("warnings", [])),
    "issues": summary.get("issues", []),
    "warnings": summary.get("warnings", []),
    "readiness_stdout": str(output_path),
    "dataset_json": artifacts.get("dataset_json"),
    "preflight_json": artifacts.get("preflight_json"),
    "summary_json": artifacts.get("summary_json"),
}

with monitor_jsonl.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

print("1" if record["ready_for_short_ddp_smoke"] else "0")
PY
)"

  echo "attempt=${attempt} readiness_status=${readiness_status} ready_for_short_ddp_smoke=${ready} output=${output_path}"

  if [ "${ready}" = "1" ] && [ "${STOP_ON_READY}" = "1" ]; then
    break
  fi

  if [ "${attempt}" -lt "${MAX_CHECKS}" ] && [ "${INTERVAL_SECONDS}" -gt 0 ]; then
    sleep "${INTERVAL_SECONDS}"
  fi

  attempt=$((attempt + 1))
done

"${PYTHON_BIN}" - "${MONITOR_JSONL}" "${SUMMARY_JSON}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

monitor_jsonl = Path(sys.argv[1])
summary_json = Path(sys.argv[2])

records = []
with monitor_jsonl.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            records.append(json.loads(line))

latest = records[-1] if records else {}
ever_ready = any(record.get("ready_for_short_ddp_smoke") for record in records)
summary = {
    "monitor_jsonl": str(monitor_jsonl),
    "attempts": len(records),
    "ever_ready_for_short_ddp_smoke": ever_ready,
    "latest_ready_for_short_ddp_smoke": bool(
        latest.get("ready_for_short_ddp_smoke")
    ),
    "latest_readiness_status": latest.get("readiness_status"),
    "latest_next_action": latest.get("next_action"),
    "latest_data_root": latest.get("data_root"),
    "latest_config": latest.get("config"),
    "latest_physical_gpus": latest.get("physical_gpus"),
    "latest_preflight_status": latest.get("preflight_status"),
    "latest_preflight_ok": latest.get("preflight_ok"),
    "latest_preflight_dataset_bytes": latest.get("preflight_dataset_bytes"),
    "latest_preflight_recent_file_count": latest.get(
        "preflight_recent_file_count"
    ),
    "latest_preflight_latest_mtime_age_seconds": latest.get(
        "preflight_latest_mtime_age_seconds"
    ),
    "latest_issues": latest.get("issues", []),
    "latest_warnings": latest.get("warnings", []),
    "latest_readiness_stdout": latest.get("readiness_stdout"),
    "latest_summary_json": latest.get("summary_json"),
}
summary_json.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if summary["latest_ready_for_short_ddp_smoke"] else 3)
PY
