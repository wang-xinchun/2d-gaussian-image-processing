#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/deshake_ddp_0_3_pending"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
SESSION="${TWODGAUSS_DESHAKE_GPU3_SESSION:-2dgauss_deshake_gpu3_train}"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"

DATA_ROOT="${TWODGAUSS_DESHAKE_DATA_ROOT:-}"
CONFIG="${TWODGAUSS_DESHAKE_CONFIG:-}"
PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-3}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-1}"
CONFIRM="${TWODGAUSS_DESHAKE_CONFIRM_GPU3_TRAIN:-}"

MIN_FREE_MIB="${TWODGAUSS_GPU3_TRAIN_MIN_FREE_MIB:-30000}"
MAX_GPU_UTILIZATION="${TWODGAUSS_GPU3_TRAIN_MAX_GPU_UTILIZATION:-20}"
MAX_BATCH_SIZE="${TWODGAUSS_GPU3_TRAIN_MAX_BATCH_SIZE:-24}"
MAX_GLOBAL_BATCH_SIZE="${TWODGAUSS_GPU3_TRAIN_MAX_GLOBAL_BATCH_SIZE:-24}"
MAX_NUM_WORKERS="${TWODGAUSS_GPU3_TRAIN_MAX_NUM_WORKERS:-0}"
MAX_TRAIN_CROP_SIZE="${TWODGAUSS_GPU3_TRAIN_MAX_TRAIN_CROP_SIZE:-512}"
MAX_VAL_CROP_SIZE="${TWODGAUSS_GPU3_TRAIN_MAX_VAL_CROP_SIZE:-512}"
MIN_DATA_BYTES="${TWODGAUSS_DESHAKE_MIN_DATA_BYTES:-1}"
MIN_VIDEO_FILES="${TWODGAUSS_DESHAKE_MIN_VIDEO_FILES:-0}"
MIN_FRAME_FILES="${TWODGAUSS_DESHAKE_MIN_FRAME_FILES:-1}"
MIN_TRAIN_PAIRS="${TWODGAUSS_DESHAKE_MIN_TRAIN_PAIRS:-112}"
MIN_VAL_PAIRS="${TWODGAUSS_DESHAKE_MIN_VAL_PAIRS:-32}"
MAX_SCAN_FILES="${TWODGAUSS_DESHAKE_MAX_SCAN_FILES:-500000}"
MIN_FILE_AGE_SECONDS="${TWODGAUSS_DESHAKE_MIN_FILE_AGE_SECONDS:-0}"
MAX_STEPS_CAP="${TWODGAUSS_GPU3_TRAIN_MAX_STEPS_CAP:-2000}"
WALL_CLOCK_MINUTES_CAP="${TWODGAUSS_GPU3_TRAIN_WALL_CLOCK_MINUTES_CAP:-120}"
CUDA_MEMORY_FRACTION="${TWODGAUSS_CUDA_MEMORY_FRACTION:-0.85}"
WAIT_TIMEOUT_SECONDS="${TWODGAUSS_GPU3_TRAIN_WAIT_TIMEOUT_SECONDS:-7200}"
RUNTIME_MIN_FREE_MIB="${TWODGAUSS_RUNTIME_MIN_FREE_MIB:-4096}"
RUNTIME_POLL_SECONDS="${TWODGAUSS_RUNTIME_POLL_SECONDS:-5}"
RUNTIME_STOP_GRACE_SECONDS="${TWODGAUSS_RUNTIME_STOP_GRACE_SECONDS:-20}"

MAX_TRAIN_CONTEXT_SIZE="${TWODGAUSS_GPU3_TRAIN_MAX_TRAIN_CONTEXT_SIZE:-0}"
REQUIRED_TRAIN_SUPERVISION_CROP_SIZE="${TWODGAUSS_GPU3_TRAIN_REQUIRED_SUPERVISION_CROP_SIZE:-0}"

CLASSIFICATION="${TWODGAUSS_DESHAKE_CLASSIFICATION:-current_deshake_gpu3_full_train}"
CONFIRM_TOKEN="${TWODGAUSS_DESHAKE_CONFIRM_GPU3_TRAIN_TOKEN:-launch_gpu3_deshake_full_train}"
LOG_PATH="${LOG_DIR}/train_deshake_gpu3_train_${STAMP}.log"
PREFLIGHT_JSON="${LOG_PATH}.preflight.json"
CONFIG_GATE_JSON="${LOG_DIR}/gpu3_train_config_gate_${STAMP}.json"

if [ -z "${DATA_ROOT}" ]; then
  echo "Refusing GPU3 train: TWODGAUSS_DESHAKE_DATA_ROOT is required." >&2
  exit 1
fi

if [ -z "${CONFIG}" ]; then
  echo "Refusing GPU3 train: TWODGAUSS_DESHAKE_CONFIG is required." >&2
  exit 1
fi

if [ "${PHYSICAL_GPU}" != "3" ]; then
  echo "Refusing GPU3 train: this launcher is physical GPU3-only, got ${PHYSICAL_GPU}." >&2
  exit 1
fi

for value_name in \
  MIN_FREE_MIB MAX_GPU_UTILIZATION MAX_BATCH_SIZE MAX_GLOBAL_BATCH_SIZE MAX_NUM_WORKERS \
  MAX_TRAIN_CROP_SIZE MAX_VAL_CROP_SIZE MIN_DATA_BYTES MIN_VIDEO_FILES MIN_FRAME_FILES \
  MIN_TRAIN_PAIRS MIN_VAL_PAIRS MAX_SCAN_FILES MIN_FILE_AGE_SECONDS MAX_STEPS_CAP \
  WAIT_TIMEOUT_SECONDS RUNTIME_MIN_FREE_MIB RUNTIME_POLL_SECONDS RUNTIME_STOP_GRACE_SECONDS \
  MAX_TRAIN_CONTEXT_SIZE REQUIRED_TRAIN_SUPERVISION_CROP_SIZE; do
  value="${!value_name}"
  case "${value}" in
    ''|*[!0-9]*)
      echo "Refusing GPU3 train: ${value_name} must be a non-negative integer." >&2
      exit 1
      ;;
  esac
done

if [ "${MAX_BATCH_SIZE}" -lt 1 ] || [ "${MAX_GLOBAL_BATCH_SIZE}" -lt 1 ] || [ "${MAX_STEPS_CAP}" -lt 1 ]; then
  echo "Refusing GPU3 train: batch and step caps must be >= 1." >&2
  exit 1
fi

set +e
"${PYTHON_BIN}" scripts/preflight_deshake_ddp_gpu0_3.py \
  --data-root "${DATA_ROOT}" \
  --config "${CONFIG}" \
  --physical-gpus "${PHYSICAL_GPU}" \
  --allow-single-gpu3-train \
  --min-free-mib "${MIN_FREE_MIB}" \
  --max-gpu-utilization "${MAX_GPU_UTILIZATION}" \
  --max-per-rank-batch-size "${MAX_BATCH_SIZE}" \
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

cat "${PREFLIGHT_JSON}"
if [ "${PREFLIGHT_STATUS}" -ne 0 ]; then
  echo "Refusing GPU3 train: preflight failed with status ${PREFLIGHT_STATUS}." >&2
  exit "${PREFLIGHT_STATUS}"
fi

set +e
"${PYTHON_BIN}" - \
  "${CONFIG}" \
  "${CONFIG_GATE_JSON}" \
  "${MAX_BATCH_SIZE}" \
  "${MAX_GLOBAL_BATCH_SIZE}" \
  "${MAX_STEPS_CAP}" \
  "${WALL_CLOCK_MINUTES_CAP}" \
  "${MAX_TRAIN_CONTEXT_SIZE}" \
  "${REQUIRED_TRAIN_SUPERVISION_CROP_SIZE}" <<'PY'
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
max_batch_size = int(sys.argv[3])
max_global_batch_size = int(sys.argv[4])
max_steps_cap = int(sys.argv[5])
wall_clock_minutes_cap = float(sys.argv[6])
max_train_context_size = int(sys.argv[7])
required_train_supervision_crop_size = int(sys.argv[8])

issues: list[str] = []
if not config_path.is_file():
    issues.append(f"config does not exist: {config_path}")
    parsed = {}
else:
    with config_path.open("rb") as handle:
        parsed = tomllib.load(handle)

identity = parsed.get("identity", {}) if isinstance(parsed, dict) else {}
training = parsed.get("training", {}) if isinstance(parsed, dict) else {}
dataset = parsed.get("dataset", {}) if isinstance(parsed, dict) else {}
task = parsed.get("task", {}) if isinstance(parsed, dict) else {}
model = parsed.get("model", {}) if isinstance(parsed, dict) else {}

run_id = identity.get("run")
task_text = json.dumps(task, ensure_ascii=False).lower()
max_steps = training.get("max_steps")
wall_clock_minutes = training.get("wall_clock_minutes")
batch_size = training.get("batch_size")
num_workers = dataset.get("num_workers")
train_crop_size = dataset.get("train_crop_size")
val_crop_size = dataset.get("val_crop_size")
use_train_inference_consistency = bool(model.get("use_train_inference_consistency", False))
train_context_size = model.get("train_context_size")
train_supervision_crop_size = model.get("train_supervision_crop_size")

def as_int(name: str, value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        issues.append(f"{name} is not an integer: {value!r}")
        return None

def as_float(name: str, value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        issues.append(f"{name} is not numeric: {value!r}")
        return None

max_steps_int = as_int("training.max_steps", max_steps)
wall_clock_float = as_float("training.wall_clock_minutes", wall_clock_minutes)
batch_size_int = as_int("training.batch_size", batch_size)
num_workers_int = as_int("dataset.num_workers", num_workers)
train_crop_size_int = as_int("dataset.train_crop_size", train_crop_size)
val_crop_size_int = as_int("dataset.val_crop_size", val_crop_size)
train_context_size_int = as_int("model.train_context_size", train_context_size)
train_supervision_crop_size_int = as_int("model.train_supervision_crop_size", train_supervision_crop_size)

if max_steps_int is not None and max_steps_int > max_steps_cap:
    issues.append(f"training.max_steps {max_steps_int} > GPU3 train cap {max_steps_cap}")
if wall_clock_float is not None and wall_clock_float > wall_clock_minutes_cap:
    issues.append(
        f"training.wall_clock_minutes {wall_clock_float} > GPU3 train cap {wall_clock_minutes_cap}"
    )
if batch_size_int is not None and batch_size_int > max_batch_size:
    issues.append(f"training.batch_size {batch_size_int} > GPU3 train cap {max_batch_size}")
if batch_size_int is not None and batch_size_int > max_global_batch_size:
    issues.append(f"single-GPU global batch_size {batch_size_int} > cap {max_global_batch_size}")
if num_workers_int is not None and num_workers_int > 0:
    issues.append(f"dataset.num_workers {num_workers_int} > GPU3 train cap 0")
if train_crop_size_int is not None and train_crop_size_int > 512:
    issues.append(f"dataset.train_crop_size {train_crop_size_int} > cap 512")
if val_crop_size_int is not None and val_crop_size_int > 512:
    issues.append(f"dataset.val_crop_size {val_crop_size_int} > cap 512")
if max_train_context_size > 0:
    if not use_train_inference_consistency:
        issues.append(
            "model.use_train_inference_consistency must be true for the context-train wrapper"
        )
    if train_context_size_int is None:
        issues.append("model.train_context_size is required for the context-train wrapper")
    elif train_context_size_int <= 0:
        issues.append(f"model.train_context_size must be positive; got {train_context_size_int}")
    elif train_context_size_int > max_train_context_size:
        issues.append(
            f"model.train_context_size {train_context_size_int} > cap {max_train_context_size}"
        )
if required_train_supervision_crop_size > 0:
    if train_supervision_crop_size_int is None:
        issues.append("model.train_supervision_crop_size is required for the context-train wrapper")
    elif train_supervision_crop_size_int != required_train_supervision_crop_size:
        issues.append(
            f"model.train_supervision_crop_size {train_supervision_crop_size_int} != "
            f"required {required_train_supervision_crop_size}"
        )
if run_id and not any(token in str(run_id).lower() for token in ("gpu3", "dsh", "deshake")):
    issues.append(f"identity.run does not look like a GPU3 deshake train run: {run_id!r}")
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
    "global_batch_size": batch_size_int,
    "global_batch_size_cap": max_global_batch_size,
    "num_workers": num_workers_int,
    "train_crop_size": train_crop_size_int,
    "val_crop_size": val_crop_size_int,
    "use_train_inference_consistency": use_train_inference_consistency,
    "train_context_size": train_context_size_int,
    "train_context_size_cap": max_train_context_size,
    "train_supervision_crop_size": train_supervision_crop_size_int,
    "required_train_supervision_crop_size": required_train_supervision_crop_size,
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
  echo "Refusing GPU3 train: config gate failed with status ${CONFIG_GATE_STATUS}." >&2
  exit "${CONFIG_GATE_STATUS}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

if [ "${DRY_RUN}" != "1" ] && [ "${CONFIRM}" != "${CONFIRM_TOKEN}" ]; then
  echo "Refusing GPU3 train: set TWODGAUSS_DRY_RUN=1 for a dry-run, or TWODGAUSS_DESHAKE_CONFIRM_GPU3_TRAIN=${CONFIRM_TOKEN} for an explicit real launch." >&2
  exit 1
fi

check_runtime_headroom() {
  local row status free_mib
  set +e
  row="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i "${PHYSICAL_GPU}" 2>&1)"
  status=$?
  set -e
  if [ "${status}" -ne 0 ]; then
    echo "runtime_monitor_error=${row}" >&2
    return 2
  fi
  free_mib="${row#*,}"
  free_mib="${free_mib//[[:space:]]/}"
  case "${free_mib}" in
    ''|*[!0-9]*)
      echo "runtime_monitor_parse_error=${row}" >&2
      return 2
      ;;
  esac
  if [ "${free_mib}" -lt "${RUNTIME_MIN_FREE_MIB}" ]; then
    echo "runtime_low_free_mib=GPU${PHYSICAL_GPU}:${free_mib}<${RUNTIME_MIN_FREE_MIB}" >&2
    return 1
  fi
  return 0
}

stop_session_for_runtime_headroom() {
  local deadline
  echo "runtime_monitor_action=stop_own_tmux_session session=${SESSION}" >&2
  tmux send-keys -t "${SESSION}" C-c 2>/dev/null || true
  deadline=$((SECONDS + RUNTIME_STOP_GRACE_SECONDS))
  while tmux has-session -t "${SESSION}" 2>/dev/null; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      echo "runtime_monitor_action=kill_own_tmux_session session=${SESSION}" >&2
      tmux kill-session -t "${SESSION}" 2>/dev/null || true
      break
    fi
    sleep 1
  done
}

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  echo "would_launch_session=${SESSION}"
  echo "physical_gpu=${PHYSICAL_GPU}"
  echo "min_free_mib=${MIN_FREE_MIB}"
  echo "max_gpu_utilization=${MAX_GPU_UTILIZATION}"
  echo "max_batch_size=${MAX_BATCH_SIZE}"
  echo "max_global_batch_size=${MAX_GLOBAL_BATCH_SIZE}"
  echo "max_num_workers=${MAX_NUM_WORKERS}"
  echo "max_train_crop_size=${MAX_TRAIN_CROP_SIZE}"
  echo "max_val_crop_size=${MAX_VAL_CROP_SIZE}"
  echo "max_train_context_size=${MAX_TRAIN_CONTEXT_SIZE}"
  echo "required_train_supervision_crop_size=${REQUIRED_TRAIN_SUPERVISION_CROP_SIZE}"
  echo "min_data_bytes=${MIN_DATA_BYTES}"
  echo "min_video_files=${MIN_VIDEO_FILES}"
  echo "min_frame_files=${MIN_FRAME_FILES}"
  echo "min_train_pairs=${MIN_TRAIN_PAIRS}"
  echo "min_val_pairs=${MIN_VAL_PAIRS}"
  echo "max_scan_files=${MAX_SCAN_FILES}"
  echo "min_file_age_seconds=${MIN_FILE_AGE_SECONDS}"
  echo "cuda_memory_fraction=${CUDA_MEMORY_FRACTION}"
  echo "data_root=${DATA_ROOT}"
  echo "config=${CONFIG}"
  echo "run_dir=${RUN_DIR}"
  echo "log_path=${LOG_PATH}"
  echo "preflight_json=${PREFLIGHT_JSON}"
  echo "config_gate_json=${CONFIG_GATE_JSON}"
  echo "python_bin=${PYTHON_BIN}"
  echo "runtime_min_free_mib=${RUNTIME_MIN_FREE_MIB}"
  echo "runtime_poll_seconds=${RUNTIME_POLL_SECONDS}"
  echo "runtime_stop_grace_seconds=${RUNTIME_STOP_GRACE_SECONDS}"
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
  printf 'physical_gpu=%s\n' '${PHYSICAL_GPU}' | tee -a '${LOG_PATH}'; \
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
echo "min_free_mib=${MIN_FREE_MIB}"
echo "max_gpu_utilization=${MAX_GPU_UTILIZATION}"
echo "max_batch_size=${MAX_BATCH_SIZE}"
echo "max_global_batch_size=${MAX_GLOBAL_BATCH_SIZE}"
echo "max_num_workers=${MAX_NUM_WORKERS}"
echo "max_train_crop_size=${MAX_TRAIN_CROP_SIZE}"
echo "max_val_crop_size=${MAX_VAL_CROP_SIZE}"
echo "max_train_context_size=${MAX_TRAIN_CONTEXT_SIZE}"
echo "required_train_supervision_crop_size=${REQUIRED_TRAIN_SUPERVISION_CROP_SIZE}"
echo "cuda_memory_fraction=${CUDA_MEMORY_FRACTION}"
echo "data_root=${DATA_ROOT}"
echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "log_path=${LOG_PATH}"
echo "preflight_json=${PREFLIGHT_JSON}"
echo "config_gate_json=${CONFIG_GATE_JSON}"
echo "python_bin=${PYTHON_BIN}"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
RUNTIME_ABORTED=0
while tmux has-session -t "${SESSION}" 2>/dev/null; do
  if [ "${SECONDS}" -ge "${deadline}" ]; then
    echo "GPU3 train timed out waiting for tmux session ${SESSION}." >&2
    exit 1
  fi
  set +e
  check_runtime_headroom
  HEADROOM_STATUS=$?
  set -e
  if [ "${HEADROOM_STATUS}" -ne 0 ]; then
    RUNTIME_ABORTED=1
    stop_session_for_runtime_headroom
    break
  fi
  sleep "${RUNTIME_POLL_SECONDS}"
done

EXIT_CODE_PATH="${LOG_PATH}.exit_code"
if [ ! -f "${EXIT_CODE_PATH}" ]; then
  if [ "${RUNTIME_ABORTED}" -eq 1 ]; then
    echo "gpu3_train_status=runtime_aborted"
    echo "runtime_abort_reason=low_or_unobservable_gpu_free_memory"
    echo "log_path=${LOG_PATH}"
    echo "exit_code_path=${EXIT_CODE_PATH}"
    exit 125
  fi
  echo "GPU3 train did not write exit code: ${EXIT_CODE_PATH}" >&2
  exit 1
fi

TRAIN_EXIT_CODE="$(tr -d '[:space:]' < "${EXIT_CODE_PATH}")"
if [ "${RUNTIME_ABORTED}" -eq 1 ]; then
  echo "gpu3_train_status=runtime_aborted"
  echo "runtime_abort_reason=low_or_unobservable_gpu_free_memory"
  echo "train_exit_code=${TRAIN_EXIT_CODE}"
  echo "log_path=${LOG_PATH}"
  echo "exit_code_path=${EXIT_CODE_PATH}"
  exit 125
fi
echo "gpu3_train_status=completed"
echo "train_exit_code=${TRAIN_EXIT_CODE}"
echo "log_path=${LOG_PATH}"
echo "exit_code_path=${EXIT_CODE_PATH}"
exit "${TRAIN_EXIT_CODE}"
