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
LAUNCHER="${RUN_DIR}/launch_deshake_ddp_tmux.sh"

DATA_ROOT="${TWODGAUSS_DESHAKE_DATA_ROOT:-}"
CONFIG="${TWODGAUSS_DESHAKE_CONFIG:-}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-1}"
CONFIRM="${TWODGAUSS_DESHAKE_CONFIRM_SMOKE:-}"
MAX_STEPS_CAP="${TWODGAUSS_DESHAKE_SMOKE_MAX_STEPS_CAP:-50}"
WALL_CLOCK_MINUTES_CAP="${TWODGAUSS_DESHAKE_SMOKE_WALL_CLOCK_MINUTES_CAP:-30}"
CONFIRM_TOKEN="launch_gpu0_3_deshake_smoke"

READINESS_STDOUT="${LOG_DIR}/smoke_readiness_${STAMP}.json"
CONFIG_GATE_JSON="${LOG_DIR}/smoke_config_gate_${STAMP}.json"

if [ -z "${DATA_ROOT}" ]; then
  echo "Refusing smoke: TWODGAUSS_DESHAKE_DATA_ROOT is required." >&2
  exit 1
fi

if [ -z "${CONFIG}" ]; then
  echo "Refusing smoke: TWODGAUSS_DESHAKE_CONFIG is required." >&2
  exit 1
fi

case "${MAX_STEPS_CAP}" in
  ''|*[!0-9]*)
    echo "Refusing smoke: TWODGAUSS_DESHAKE_SMOKE_MAX_STEPS_CAP must be a positive integer." >&2
    exit 1
    ;;
esac

if [ "${MAX_STEPS_CAP}" -lt 1 ]; then
  echo "Refusing smoke: TWODGAUSS_DESHAKE_SMOKE_MAX_STEPS_CAP must be >= 1." >&2
  exit 1
fi

set +e
"${READINESS_SCRIPT}" > "${READINESS_STDOUT}" 2>&1
READINESS_STATUS=$?
set -e

cat "${READINESS_STDOUT}"
if [ "${READINESS_STATUS}" -ne 0 ]; then
  echo "Refusing smoke: readiness failed with status ${READINESS_STATUS}." >&2
  exit "${READINESS_STATUS}"
fi

set +e
"${PYTHON_BIN}" - \
  "${CONFIG}" \
  "${CONFIG_GATE_JSON}" \
  "${MAX_STEPS_CAP}" \
  "${WALL_CLOCK_MINUTES_CAP}" <<'PY'
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
max_steps_cap = int(sys.argv[3])
wall_clock_minutes_cap = float(sys.argv[4])

issues: list[str] = []
if not config_path.is_file():
    issues.append(f"config does not exist: {config_path}")
    parsed = {}
else:
    with config_path.open("rb") as handle:
        parsed = tomllib.load(handle)

identity = parsed.get("identity", {}) if isinstance(parsed, dict) else {}
training = parsed.get("training", {}) if isinstance(parsed, dict) else {}

run_id = identity.get("run")
max_steps = training.get("max_steps")
wall_clock_minutes = training.get("wall_clock_minutes")
batch_size = training.get("batch_size")

try:
    max_steps_int = int(max_steps)
except (TypeError, ValueError):
    issues.append(f"training.max_steps is not an integer: {max_steps!r}")
    max_steps_int = None

try:
    wall_clock_float = float(wall_clock_minutes)
except (TypeError, ValueError):
    issues.append(f"training.wall_clock_minutes is not numeric: {wall_clock_minutes!r}")
    wall_clock_float = None

try:
    batch_size_int = int(batch_size)
except (TypeError, ValueError):
    issues.append(f"training.batch_size is not an integer: {batch_size!r}")
    batch_size_int = None

if max_steps_int is not None and max_steps_int > max_steps_cap:
    issues.append(f"training.max_steps {max_steps_int} > smoke cap {max_steps_cap}")
if wall_clock_float is not None and wall_clock_float > wall_clock_minutes_cap:
    issues.append(
        f"training.wall_clock_minutes {wall_clock_float} > smoke cap {wall_clock_minutes_cap}"
    )
if batch_size_int is not None and batch_size_int > 2:
    issues.append(f"training.batch_size {batch_size_int} > smoke per-rank cap 2")
if run_id and not any(token in str(run_id).lower() for token in ("smoke", "dsh", "deshake")):
    issues.append(f"identity.run does not look like a deshake smoke run: {run_id!r}")

result = {
    "ok": not issues,
    "issues": issues,
    "config": str(config_path),
    "run_id": run_id,
    "max_steps": max_steps_int,
    "max_steps_cap": max_steps_cap,
    "wall_clock_minutes": wall_clock_float,
    "wall_clock_minutes_cap": wall_clock_minutes_cap,
    "batch_size": batch_size_int,
}
output_path.write_text(
    json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if result["ok"] else 2)
PY
CONFIG_GATE_STATUS=$?
set -e

if [ "${CONFIG_GATE_STATUS}" -ne 0 ]; then
  echo "Refusing smoke: config smoke gate failed with status ${CONFIG_GATE_STATUS}." >&2
  exit "${CONFIG_GATE_STATUS}"
fi

if [ "${DRY_RUN}" != "1" ] && [ "${CONFIRM}" != "${CONFIRM_TOKEN}" ]; then
  echo "Refusing smoke: set TWODGAUSS_DRY_RUN=1 for a dry-run, or set TWODGAUSS_DESHAKE_CONFIRM_SMOKE=${CONFIRM_TOKEN} for an explicit real smoke launch." >&2
  exit 1
fi

echo "smoke_dry_run=${DRY_RUN}"
echo "smoke_readiness_stdout=${READINESS_STDOUT}"
echo "smoke_config_gate_json=${CONFIG_GATE_JSON}"

export TWODGAUSS_DRY_RUN="${DRY_RUN}"
exec "${LAUNCHER}"
