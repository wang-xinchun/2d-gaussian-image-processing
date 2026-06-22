#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/deshake_ddp_0_3_pending"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
SESSION="${TWODGAUSS_DESHAKE_GPU0_SESSION:-2dgauss_deshake_gpu0_bs8_smoke}"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"

DATA_ROOT="${TWODGAUSS_DESHAKE_DATA_ROOT:-}"
CONFIG="${TWODGAUSS_DESHAKE_CONFIG:-}"
PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-0}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-1}"
CONFIRM="${TWODGAUSS_DESHAKE_CONFIRM_GPU0_SMOKE:-}"
CONFIRM_TOKEN="launch_gpu0_deshake_bs8_smoke"

MIN_FREE_MIB="${TWODGAUSS_GPU0_SMOKE_MIN_FREE_MIB:-40000}"
MAX_GPU_UTILIZATION="${TWODGAUSS_GPU0_SMOKE_MAX_GPU_UTILIZATION:-20}"
MAX_BATCH_SIZE="${TWODGAUSS_GPU0_SMOKE_MAX_BATCH_SIZE:-8}"
MAX_GLOBAL_BATCH_SIZE="${TWODGAUSS_GPU0_SMOKE_MAX_GLOBAL_BATCH_SIZE:-8}"
MAX_NUM_WORKERS="${TWODGAUSS_GPU0_SMOKE_MAX_NUM_WORKERS:-2}"
MIN_DATA_BYTES="${TWODGAUSS_GPU0_SMOKE_MIN_DATA_BYTES:-322122547200}"
MIN_VIDEO_FILES="${TWODGAUSS_DESHAKE_MIN_VIDEO_FILES:-1}"
MIN_FRAME_FILES="${TWODGAUSS_DESHAKE_MIN_FRAME_FILES:-0}"
MIN_TRAIN_PAIRS="${TWODGAUSS_DESHAKE_MIN_TRAIN_PAIRS:-1}"
MIN_VAL_PAIRS="${TWODGAUSS_DESHAKE_MIN_VAL_PAIRS:-1}"
MAX_SCAN_FILES="${TWODGAUSS_DESHAKE_MAX_SCAN_FILES:-500000}"
MIN_FILE_AGE_SECONDS="${TWODGAUSS_DESHAKE_MIN_FILE_AGE_SECONDS:-600}"
MAX_STEPS_CAP="${TWODGAUSS_GPU0_SMOKE_MAX_STEPS_CAP:-50}"
WALL_CLOCK_MINUTES_CAP="${TWODGAUSS_GPU0_SMOKE_WALL_CLOCK_MINUTES_CAP:-30}"
CUDA_MEMORY_FRACTION="${TWODGAUSS_CUDA_MEMORY_FRACTION:-}"
CLASSIFICATION="current_deshake_gpu0_bs${MAX_BATCH_SIZE}_smoke"

LOG_PATH="${LOG_DIR}/train_deshake_gpu0_smoke_${STAMP}.log"
PREFLIGHT_JSON="${LOG_PATH}.preflight.json"
CONFIG_GATE_JSON="${LOG_DIR}/gpu0_smoke_config_gate_${STAMP}.json"

if [ -z "${DATA_ROOT}" ]; then
  echo "Refusing GPU0 smoke: TWODGAUSS_DESHAKE_DATA_ROOT is required." >&2
  exit 1
fi

if [ -z "${CONFIG}" ]; then
  echo "Refusing GPU0 smoke: TWODGAUSS_DESHAKE_CONFIG is required." >&2
  exit 1
fi

if [ "${PHYSICAL_GPU}" != "0" ]; then
  echo "Refusing GPU0 smoke: this launcher is physical GPU0-only, got ${PHYSICAL_GPU}." >&2
  exit 1
fi

for value_name in \
  MIN_FREE_MIB MAX_GPU_UTILIZATION MAX_BATCH_SIZE MAX_GLOBAL_BATCH_SIZE MAX_NUM_WORKERS \
  MIN_DATA_BYTES MIN_VIDEO_FILES MIN_FRAME_FILES MIN_TRAIN_PAIRS MIN_VAL_PAIRS MAX_SCAN_FILES \
  MIN_FILE_AGE_SECONDS MAX_STEPS_CAP; do
  value="${!value_name}"
  case "${value}" in
    ''|*[!0-9]*)
      echo "Refusing GPU0 smoke: ${value_name} must be a non-negative integer." >&2
      exit 1
      ;;
  esac
done

if [ "${MAX_BATCH_SIZE}" -lt 1 ]; then
  echo "Refusing GPU0 smoke: TWODGAUSS_GPU0_SMOKE_MAX_BATCH_SIZE must be >= 1." >&2
  exit 1
fi

if [ "${MAX_STEPS_CAP}" -lt 1 ]; then
  echo "Refusing GPU0 smoke: TWODGAUSS_GPU0_SMOKE_MAX_STEPS_CAP must be >= 1." >&2
  exit 1
fi

set +e
"${PYTHON_BIN}" scripts/preflight_deshake_ddp_gpu0_3.py \
  --data-root "${DATA_ROOT}" \
  --config "${CONFIG}" \
  --physical-gpus "${PHYSICAL_GPU}" \
  --allow-single-gpu0-smoke \
  --min-free-mib "${MIN_FREE_MIB}" \
  --max-gpu-utilization "${MAX_GPU_UTILIZATION}" \
  --max-per-rank-batch-size "${MAX_BATCH_SIZE}" \
  --max-global-batch-size "${MAX_GLOBAL_BATCH_SIZE}" \
  --max-num-workers "${MAX_NUM_WORKERS}" \
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

cat "${PREFLIGHT_JSON}"
if [ "${PREFLIGHT_STATUS}" -ne 0 ]; then
  echo "Refusing GPU0 smoke: preflight failed with status ${PREFLIGHT_STATUS}." >&2
  exit "${PREFLIGHT_STATUS}"
fi

set +e
"${PYTHON_BIN}" - \
  "${CONFIG}" \
  "${CONFIG_GATE_JSON}" \
  "${MAX_BATCH_SIZE}" \
  "${MAX_STEPS_CAP}" \
  "${WALL_CLOCK_MINUTES_CAP}" <<'PY'
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
max_batch_size = int(sys.argv[3])
max_steps_cap = int(sys.argv[4])
wall_clock_minutes_cap = float(sys.argv[5])

issues: list[str] = []
if not config_path.is_file():
    issues.append(f"config does not exist: {config_path}")
    parsed = {}
else:
    with config_path.open("rb") as handle:
        parsed = tomllib.load(handle)

identity = parsed.get("identity", {}) if isinstance(parsed, dict) else {}
training = parsed.get("training", {}) if isinstance(parsed, dict) else {}
task = parsed.get("task", {}) if isinstance(parsed, dict) else {}

run_id = identity.get("run")
task_text = json.dumps(task, ensure_ascii=False).lower()
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
    issues.append(f"training.max_steps {max_steps_int} > GPU0 smoke cap {max_steps_cap}")
if wall_clock_float is not None and wall_clock_float > wall_clock_minutes_cap:
    issues.append(
        f"training.wall_clock_minutes {wall_clock_float} > GPU0 smoke cap "
        f"{wall_clock_minutes_cap}"
    )
if batch_size_int is not None and batch_size_int > max_batch_size:
    issues.append(f"training.batch_size {batch_size_int} > GPU0 smoke cap {max_batch_size}")
if run_id and not any(token in str(run_id).lower() for token in ("gpu0", "smoke", "dsh", "deshake")):
    issues.append(f"identity.run does not look like a GPU0 deshake smoke run: {run_id!r}")
if "deshake" not in task_text and "stabil" not in task_text and "video" not in task_text:
    issues.append("task section does not look like UHD video deshaking/stabilization")

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
    "batch_size_cap": max_batch_size,
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
  echo "Refusing GPU0 smoke: config gate failed with status ${CONFIG_GATE_STATUS}." >&2
  exit "${CONFIG_GATE_STATUS}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

if [ "${DRY_RUN}" != "1" ] && [ "${CONFIRM}" != "${CONFIRM_TOKEN}" ]; then
  echo "Refusing GPU0 smoke: set TWODGAUSS_DRY_RUN=1 for a dry-run, or set TWODGAUSS_DESHAKE_CONFIRM_GPU0_SMOKE=${CONFIRM_TOKEN} for an explicit real smoke launch." >&2
  exit 1
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  echo "would_launch_session=${SESSION}"
  echo "physical_gpu=${PHYSICAL_GPU}"
  echo "max_batch_size=${MAX_BATCH_SIZE}"
  echo "min_free_mib=${MIN_FREE_MIB}"
  echo "max_gpu_utilization=${MAX_GPU_UTILIZATION}"
  echo "cuda_memory_fraction=${CUDA_MEMORY_FRACTION:-unset}"
  echo "data_root=${DATA_ROOT}"
  echo "config=${CONFIG}"
  echo "log_path=${LOG_PATH}"
  echo "preflight_json=${PREFLIGHT_JSON}"
  echo "config_gate_json=${CONFIG_GATE_JSON}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  cd '${PROJECT_ROOT}' && \
  export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
  export CUDA_VISIBLE_DEVICES='${PHYSICAL_GPU}' && \
  export TWODGAUSS_ALLOWED_PHYSICAL_GPUS='0,3' && \
  export OMP_NUM_THREADS=2 && \
  export MKL_NUM_THREADS=2 && \
  export OPENBLAS_NUM_THREADS=2 && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  export TWODGAUSS_CUDA_MEMORY_FRACTION='${CUDA_MEMORY_FRACTION}' && \
  export PYTHONUNBUFFERED=1 && \
  export PYTHONFAULTHANDLER=1 && \
  set -o pipefail; \
  printf 'classification=%s\n' '${CLASSIFICATION}' | tee '${LOG_PATH}'; \
  printf 'data_root=%s\n' '${DATA_ROOT}' | tee -a '${LOG_PATH}'; \
  printf 'config=%s\n' '${CONFIG}' | tee -a '${LOG_PATH}'; \
  '${PYTHON_BIN}' -u scripts/train_v1_t01.py --config '${CONFIG}' 2>&1 | tee -a '${LOG_PATH}'; \
  status=\${PIPESTATUS[0]}; \
  printf 'train_exit_code=%s\n' \"\${status}\" | tee -a '${LOG_PATH}'; \
  printf '%s\n' \"\${status}\" > '${LOG_PATH}.exit_code'; \
  exit \"\${status}\"
"

echo "session=${SESSION}"
echo "physical_gpu=${PHYSICAL_GPU}"
echo "batch_size_cap=${MAX_BATCH_SIZE}"
echo "data_root=${DATA_ROOT}"
echo "config=${CONFIG}"
echo "log_path=${LOG_PATH}"
echo "preflight_json=${PREFLIGHT_JSON}"
echo "config_gate_json=${CONFIG_GATE_JSON}"
