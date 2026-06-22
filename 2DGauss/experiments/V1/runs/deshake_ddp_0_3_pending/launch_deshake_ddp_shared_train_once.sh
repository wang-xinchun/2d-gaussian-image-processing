#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/deshake_ddp_0_3_pending"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"
LAUNCHER="${RUN_DIR}/launch_deshake_ddp_tmux.sh"

DATA_ROOT="${TWODGAUSS_DESHAKE_DATA_ROOT:-}"
CONFIG="${TWODGAUSS_DESHAKE_CONFIG:-}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-1}"
CONFIRM="${TWODGAUSS_DESHAKE_CONFIRM_SHARED_TRAIN:-}"
CONFIRM_TOKEN="launch_shared_gpu0_3_deshake_ddp_train"
SESSION="${TWODGAUSS_DESHAKE_SESSION:-2dgauss_deshake_ddp_0_3_shared_train}"
LAUNCH_STDOUT="${LOG_DIR}/shared_ddp_train_launch_${STAMP}.out"

MAX_STEPS_CAP="${TWODGAUSS_DESHAKE_SHARED_TRAIN_MAX_STEPS_CAP:-2000}"
WALL_CLOCK_MINUTES_CAP="${TWODGAUSS_DESHAKE_SHARED_TRAIN_WALL_CLOCK_MINUTES_CAP:-120}"
MAX_PER_RANK_BATCH_SIZE="${TWODGAUSS_MAX_PER_RANK_BATCH_SIZE:-12}"
MAX_GLOBAL_BATCH_SIZE="${TWODGAUSS_MAX_GLOBAL_BATCH_SIZE:-24}"
MAX_NUM_WORKERS="${TWODGAUSS_MAX_NUM_WORKERS:-0}"
MAX_TRAIN_CROP_SIZE="${TWODGAUSS_MAX_TRAIN_CROP_SIZE:-512}"
MAX_VAL_CROP_SIZE="${TWODGAUSS_MAX_VAL_CROP_SIZE:-512}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-18000}"
MAX_GPU_UTILIZATION="${TWODGAUSS_MAX_GPU_UTILIZATION:-20}"
GPU_UTILIZATION_POLICY="${TWODGAUSS_GPU_UTILIZATION_POLICY:-warn}"
CUDA_MEMORY_FRACTION="${TWODGAUSS_CUDA_MEMORY_FRACTION:-0.50}"
MIN_DATA_BYTES="${TWODGAUSS_DESHAKE_MIN_DATA_BYTES:-1}"
MIN_VIDEO_FILES="${TWODGAUSS_DESHAKE_MIN_VIDEO_FILES:-0}"
MIN_FRAME_FILES="${TWODGAUSS_DESHAKE_MIN_FRAME_FILES:-1}"
MIN_TRAIN_PAIRS="${TWODGAUSS_DESHAKE_MIN_TRAIN_PAIRS:-112}"
MIN_VAL_PAIRS="${TWODGAUSS_DESHAKE_MIN_VAL_PAIRS:-32}"
MAX_SCAN_FILES="${TWODGAUSS_DESHAKE_MAX_SCAN_FILES:-500000}"
MIN_FILE_AGE_SECONDS="${TWODGAUSS_DESHAKE_MIN_FILE_AGE_SECONDS:-0}"
WAIT_TIMEOUT_SECONDS="${TWODGAUSS_DESHAKE_SHARED_TRAIN_WAIT_TIMEOUT_SECONDS:-7200}"
RUNTIME_MIN_FREE_MIB="${TWODGAUSS_RUNTIME_MIN_FREE_MIB:-8192}"
RUNTIME_POLL_SECONDS="${TWODGAUSS_RUNTIME_POLL_SECONDS:-5}"
RUNTIME_STOP_GRACE_SECONDS="${TWODGAUSS_RUNTIME_STOP_GRACE_SECONDS:-20}"

if [ -z "${DATA_ROOT}" ]; then
  echo "Refusing shared train: TWODGAUSS_DESHAKE_DATA_ROOT is required." >&2
  exit 1
fi

if [ -z "${CONFIG}" ]; then
  echo "Refusing shared train: TWODGAUSS_DESHAKE_CONFIG is required." >&2
  exit 1
fi

for value_name in MAX_STEPS_CAP MAX_PER_RANK_BATCH_SIZE MAX_GLOBAL_BATCH_SIZE MAX_NUM_WORKERS MAX_TRAIN_CROP_SIZE MAX_VAL_CROP_SIZE MIN_FREE_MIB MAX_GPU_UTILIZATION MIN_DATA_BYTES MIN_VIDEO_FILES MIN_FRAME_FILES MIN_TRAIN_PAIRS MIN_VAL_PAIRS MAX_SCAN_FILES MIN_FILE_AGE_SECONDS WAIT_TIMEOUT_SECONDS RUNTIME_MIN_FREE_MIB RUNTIME_POLL_SECONDS RUNTIME_STOP_GRACE_SECONDS; do
  value="${!value_name}"
  case "${value}" in
    ''|*[!0-9]*)
      echo "Refusing shared train: ${value_name} must be a non-negative integer." >&2
      exit 1
      ;;
  esac
done

if [ "${MAX_STEPS_CAP}" -lt 1 ] || [ "${MAX_PER_RANK_BATCH_SIZE}" -lt 1 ] || [ "${MAX_GLOBAL_BATCH_SIZE}" -lt 1 ]; then
  echo "Refusing shared train: step and batch caps must be >= 1." >&2
  exit 1
fi

if [ "${GPU_UTILIZATION_POLICY}" != "warn" ]; then
  echo "Refusing shared train: TWODGAUSS_GPU_UTILIZATION_POLICY must be warn for this explicit shared-GPU train wrapper." >&2
  exit 1
fi

if [ "${DRY_RUN}" != "1" ] && [ "${CONFIRM}" != "${CONFIRM_TOKEN}" ]; then
  echo "Refusing shared train: set TWODGAUSS_DRY_RUN=1 for dry-run or TWODGAUSS_DESHAKE_CONFIRM_SHARED_TRAIN=${CONFIRM_TOKEN} for a real shared train." >&2
  exit 1
fi

check_runtime_headroom() {
  local rows status line gpu_index free_mib
  set +e
  rows="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i 0,3 2>&1)"
  status=$?
  set -e
  if [ "${status}" -ne 0 ]; then
    echo "runtime_monitor_error=${rows}" >&2
    return 2
  fi
  while IFS= read -r line; do
    [ -z "${line}" ] && continue
    gpu_index="${line%%,*}"
    free_mib="${line#*,}"
    gpu_index="${gpu_index//[[:space:]]/}"
    free_mib="${free_mib//[[:space:]]/}"
    case "${free_mib}" in
      ''|*[!0-9]*)
        echo "runtime_monitor_parse_error=${line}" >&2
        return 2
        ;;
    esac
    if [ "${free_mib}" -lt "${RUNTIME_MIN_FREE_MIB}" ]; then
      echo "runtime_low_free_mib=GPU${gpu_index}:${free_mib}<${RUNTIME_MIN_FREE_MIB}" >&2
      return 1
    fi
  done <<< "${rows}"
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

set +e
"${PYTHON_BIN}" - \
  "${CONFIG}" \
  "${MAX_STEPS_CAP}" \
  "${WALL_CLOCK_MINUTES_CAP}" \
  "${MAX_PER_RANK_BATCH_SIZE}" \
  "${MAX_GLOBAL_BATCH_SIZE}" \
  "${MAX_NUM_WORKERS}" \
  "${MAX_TRAIN_CROP_SIZE}" \
  "${MAX_VAL_CROP_SIZE}" <<'PY'
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
max_steps_cap = int(sys.argv[2])
wall_clock_minutes_cap = float(sys.argv[3])
max_per_rank_batch_size = int(sys.argv[4])
max_global_batch_size = int(sys.argv[5])
max_num_workers = int(sys.argv[6])
max_train_crop_size = int(sys.argv[7])
max_val_crop_size = int(sys.argv[8])

issues: list[str] = []
with config_path.open("rb") as handle:
    config = tomllib.load(handle)

training = config.get("training", {})
dataset = config.get("dataset", {})
task_text = str(config.get("task", {})).lower()
run_id = str(config.get("identity", {}).get("run", ""))

max_steps = int(training.get("max_steps", 0))
wall_clock_minutes = float(training.get("wall_clock_minutes", 0))
batch_size = int(training.get("batch_size", 0))
num_workers = int(dataset.get("num_workers", 2))
train_crop_size = int(dataset.get("train_crop_size", 0))
val_crop_size = int(dataset.get("val_crop_size", 0))

if "deshake" not in task_text and "stabil" not in task_text:
    issues.append("config task does not look like deshake/stabilization")
if "dsh" not in run_id.lower():
    issues.append(f"identity.run does not look like a deshake run: {run_id!r}")
if max_steps < 1 or max_steps > max_steps_cap:
    issues.append(f"training.max_steps {max_steps} is outside shared train cap 1..{max_steps_cap}")
if wall_clock_minutes <= 0 or wall_clock_minutes > wall_clock_minutes_cap:
    issues.append(
        f"training.wall_clock_minutes {wall_clock_minutes} is outside shared train cap (0,{wall_clock_minutes_cap}]"
    )
if batch_size < 1 or batch_size > max_per_rank_batch_size:
    issues.append(f"training.batch_size {batch_size} > shared per-rank cap {max_per_rank_batch_size}")
if batch_size * 2 > max_global_batch_size:
    issues.append(f"global batch_size {batch_size * 2} > shared global cap {max_global_batch_size}")
if num_workers > max_num_workers:
    issues.append(f"dataset.num_workers {num_workers} > shared cap {max_num_workers}")
if max_train_crop_size > 0 and (train_crop_size < 1 or train_crop_size > max_train_crop_size):
    issues.append(f"dataset.train_crop_size {train_crop_size} outside shared cap 1..{max_train_crop_size}")
if max_val_crop_size > 0 and (val_crop_size < 1 or val_crop_size > max_val_crop_size):
    issues.append(f"dataset.val_crop_size {val_crop_size} outside shared cap 1..{max_val_crop_size}")

for issue in issues:
    print(f"issue={issue}", file=sys.stderr)
raise SystemExit(0 if not issues else 2)
PY
CONFIG_STATUS=$?
set -e

if [ "${CONFIG_STATUS}" -ne 0 ]; then
  echo "Refusing shared train: config gate failed with status ${CONFIG_STATUS}." >&2
  exit "${CONFIG_STATUS}"
fi

export TWODGAUSS_DRY_RUN="${DRY_RUN}"
export TWODGAUSS_DESHAKE_DATA_ROOT="${DATA_ROOT}"
export TWODGAUSS_DESHAKE_CONFIG="${CONFIG}"
export TWODGAUSS_DESHAKE_SESSION="${SESSION}"
export TWODGAUSS_PHYSICAL_GPUS=0,3
export TWODGAUSS_MIN_FREE_MIB="${MIN_FREE_MIB}"
export TWODGAUSS_MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION}"
export TWODGAUSS_GPU_UTILIZATION_POLICY="${GPU_UTILIZATION_POLICY}"
export TWODGAUSS_MAX_PER_RANK_BATCH_SIZE="${MAX_PER_RANK_BATCH_SIZE}"
export TWODGAUSS_MAX_GLOBAL_BATCH_SIZE="${MAX_GLOBAL_BATCH_SIZE}"
export TWODGAUSS_MAX_NUM_WORKERS="${MAX_NUM_WORKERS}"
export TWODGAUSS_MAX_TRAIN_CROP_SIZE="${MAX_TRAIN_CROP_SIZE}"
export TWODGAUSS_MAX_VAL_CROP_SIZE="${MAX_VAL_CROP_SIZE}"
export TWODGAUSS_CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION}"
export TWODGAUSS_DESHAKE_MIN_DATA_BYTES="${MIN_DATA_BYTES}"
export TWODGAUSS_DESHAKE_MIN_VIDEO_FILES="${MIN_VIDEO_FILES}"
export TWODGAUSS_DESHAKE_MIN_FRAME_FILES="${MIN_FRAME_FILES}"
export TWODGAUSS_DESHAKE_MIN_TRAIN_PAIRS="${MIN_TRAIN_PAIRS}"
export TWODGAUSS_DESHAKE_MIN_VAL_PAIRS="${MIN_VAL_PAIRS}"
export TWODGAUSS_DESHAKE_MAX_SCAN_FILES="${MAX_SCAN_FILES}"
export TWODGAUSS_DESHAKE_MIN_FILE_AGE_SECONDS="${MIN_FILE_AGE_SECONDS}"

set +e
"${LAUNCHER}" > "${LAUNCH_STDOUT}" 2>&1
LAUNCH_STATUS=$?
set -e
cat "${LAUNCH_STDOUT}"

if [ "${LAUNCH_STATUS}" -ne 0 ]; then
  echo "Refusing shared train: launcher exited with status ${LAUNCH_STATUS}." >&2
  exit "${LAUNCH_STATUS}"
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "shared_train_status=dry_run_ok"
  echo "launch_stdout=${LAUNCH_STDOUT}"
  echo "runtime_min_free_mib=${RUNTIME_MIN_FREE_MIB}"
  echo "runtime_poll_seconds=${RUNTIME_POLL_SECONDS}"
  echo "runtime_stop_grace_seconds=${RUNTIME_STOP_GRACE_SECONDS}"
  exit 0
fi

LOG_PATH="$(awk -F= '/^log_path=/{print $2}' "${LAUNCH_STDOUT}" | tail -n 1)"
if [ -z "${LOG_PATH}" ]; then
  echo "Shared train could not find launcher log_path in ${LAUNCH_STDOUT}." >&2
  exit 1
fi

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
RUNTIME_ABORTED=0
while tmux has-session -t "${SESSION}" 2>/dev/null; do
  if [ "${SECONDS}" -ge "${deadline}" ]; then
    echo "Shared train timed out waiting for tmux session ${SESSION}." >&2
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
    echo "shared_train_status=runtime_aborted"
    echo "runtime_abort_reason=low_or_unobservable_gpu_free_memory"
    echo "log_path=${LOG_PATH}"
    echo "exit_code_path=${EXIT_CODE_PATH}"
    exit 125
  fi
  echo "Shared train did not write exit code: ${EXIT_CODE_PATH}" >&2
  exit 1
fi

TRAIN_EXIT_CODE="$(tr -d '[:space:]' < "${EXIT_CODE_PATH}")"
if [ "${RUNTIME_ABORTED}" -eq 1 ]; then
  echo "shared_train_status=runtime_aborted"
  echo "runtime_abort_reason=low_or_unobservable_gpu_free_memory"
  echo "train_exit_code=${TRAIN_EXIT_CODE}"
  echo "log_path=${LOG_PATH}"
  echo "exit_code_path=${EXIT_CODE_PATH}"
  exit 125
fi
echo "shared_train_status=completed"
echo "train_exit_code=${TRAIN_EXIT_CODE}"
echo "log_path=${LOG_PATH}"
echo "exit_code_path=${EXIT_CODE_PATH}"
exit "${TRAIN_EXIT_CODE}"
