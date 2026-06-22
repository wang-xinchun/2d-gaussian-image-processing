#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/deshake_ddp_0_3_pending"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"
PREP_SCRIPT="scripts/prepare_deshake_paired_frames.py"
CONFIG_SCRIPT="scripts/create_deshake_ddp_config.py"
QUEUE_WRAPPER="${RUN_DIR}/wait_and_launch_deshake_ddp_shared_smoke.sh"

DATA_ROOT="${TWODGAUSS_DESHAKE_DATA_ROOT:-}"
OUTPUT_ROOT="${TWODGAUSS_DESHAKE_PREP_OUTPUT_ROOT:-}"
TRAIN_PAIR="${TWODGAUSS_DESHAKE_TRAIN_PAIR:-}"
VAL_PAIR="${TWODGAUSS_DESHAKE_VAL_PAIR:-}"
WRITE_FRAMES="${TWODGAUSS_DESHAKE_WRITE_FRAMES:-0}"
CONFIRM_PREP="${TWODGAUSS_DESHAKE_CONFIRM_PREP_WRITE:-}"
CONFIRM_PREP_TOKEN="prepare_explicit_deshake_pairs"
CONFIG_OUTPUT="${TWODGAUSS_DESHAKE_CONFIG_OUTPUT:-}"
RUN_ID="${TWODGAUSS_DESHAKE_RUN_ID:-A_dsh11_prepared_shared_ddp_bs12_smoke}"
TRAIN_FRAMES_PER_PAIR="${TWODGAUSS_DESHAKE_TRAIN_FRAMES_PER_PAIR:-16}"
VAL_FRAMES_PER_PAIR="${TWODGAUSS_DESHAKE_VAL_FRAMES_PER_PAIR:-4}"
START_FRAME="${TWODGAUSS_DESHAKE_START_FRAME:-0}"
STRIDE="${TWODGAUSS_DESHAKE_FRAME_STRIDE:-60}"
TRAIN_LIMIT="${TWODGAUSS_DESHAKE_TRAIN_LIMIT:-64}"
VAL_LIMIT="${TWODGAUSS_DESHAKE_VAL_LIMIT:-4}"
MAX_STEPS="${TWODGAUSS_DESHAKE_MAX_STEPS:-5}"
WALL_CLOCK_MINUTES="${TWODGAUSS_DESHAKE_WALL_CLOCK_MINUTES:-10}"
TRAIN_CROP_SIZE="${TWODGAUSS_DESHAKE_TRAIN_CROP_SIZE:-512}"
VAL_CROP_SIZE="${TWODGAUSS_DESHAKE_VAL_CROP_SIZE:-512}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-1}"

SHARED_BATCH_SIZE="${TWODGAUSS_DESHAKE_SHARED_BATCH_SIZE:-12}"
SHARED_NUM_WORKERS="${TWODGAUSS_DESHAKE_SHARED_NUM_WORKERS:-0}"
GPU_UTILIZATION_POLICY="${TWODGAUSS_GPU_UTILIZATION_POLICY:-warn}"
CUDA_MEMORY_FRACTION="${TWODGAUSS_CUDA_MEMORY_FRACTION:-0.50}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-32000}"
MAX_PER_RANK_BATCH_SIZE="${TWODGAUSS_MAX_PER_RANK_BATCH_SIZE:-12}"
MAX_GLOBAL_BATCH_SIZE="${TWODGAUSS_MAX_GLOBAL_BATCH_SIZE:-24}"
MAX_NUM_WORKERS="${TWODGAUSS_MAX_NUM_WORKERS:-0}"
MAX_TRAIN_CROP_SIZE="${TWODGAUSS_MAX_TRAIN_CROP_SIZE:-512}"
MAX_VAL_CROP_SIZE="${TWODGAUSS_MAX_VAL_CROP_SIZE:-512}"
RUNTIME_MIN_FREE_MIB="${TWODGAUSS_RUNTIME_MIN_FREE_MIB:-8192}"
RUNTIME_POLL_SECONDS="${TWODGAUSS_RUNTIME_POLL_SECONDS:-5}"
RUNTIME_STOP_GRACE_SECONDS="${TWODGAUSS_RUNTIME_STOP_GRACE_SECONDS:-20}"

PLAN_JSON="${LOG_DIR}/prepare_and_queue_shared_plan_${STAMP}.json"
CONFIG_JSON="${LOG_DIR}/prepare_and_queue_shared_config_${STAMP}.json"
QUEUE_LOG="${LOG_DIR}/prepare_and_queue_shared_queue_${STAMP}.log"

if [ -z "${DATA_ROOT}" ]; then
  echo "Refusing prepare+queue shared: TWODGAUSS_DESHAKE_DATA_ROOT is required." >&2
  exit 1
fi

if [ -z "${TRAIN_PAIR}" ]; then
  echo "Refusing prepare+queue shared: TWODGAUSS_DESHAKE_TRAIN_PAIR is required as INPUT_VIDEO:TARGET_VIDEO." >&2
  exit 1
fi

if [ -z "${VAL_PAIR}" ]; then
  echo "Refusing prepare+queue shared: TWODGAUSS_DESHAKE_VAL_PAIR is required as INPUT_VIDEO:TARGET_VIDEO." >&2
  exit 1
fi

for value_name in TRAIN_FRAMES_PER_PAIR VAL_FRAMES_PER_PAIR START_FRAME STRIDE TRAIN_LIMIT VAL_LIMIT MAX_STEPS TRAIN_CROP_SIZE VAL_CROP_SIZE SHARED_BATCH_SIZE SHARED_NUM_WORKERS MIN_FREE_MIB MAX_PER_RANK_BATCH_SIZE MAX_GLOBAL_BATCH_SIZE MAX_NUM_WORKERS MAX_TRAIN_CROP_SIZE MAX_VAL_CROP_SIZE RUNTIME_MIN_FREE_MIB RUNTIME_POLL_SECONDS RUNTIME_STOP_GRACE_SECONDS; do
  value="${!value_name}"
  case "${value}" in
    ''|*[!0-9]*)
      echo "Refusing prepare+queue shared: ${value_name} must be a non-negative integer." >&2
      exit 1
      ;;
  esac
done

if [ "${TRAIN_FRAMES_PER_PAIR}" -lt 1 ] || [ "${VAL_FRAMES_PER_PAIR}" -lt 1 ]; then
  echo "Refusing prepare+queue shared: frame counts per pair must be >= 1." >&2
  exit 1
fi

if [ "${STRIDE}" -lt 1 ]; then
  echo "Refusing prepare+queue shared: TWODGAUSS_DESHAKE_FRAME_STRIDE must be >= 1." >&2
  exit 1
fi

if [ "${MAX_STEPS}" -lt 1 ]; then
  echo "Refusing prepare+queue shared: TWODGAUSS_DESHAKE_MAX_STEPS must be >= 1." >&2
  exit 1
fi

if [ "${SHARED_BATCH_SIZE}" -lt 1 ] || [ "${SHARED_BATCH_SIZE}" -gt "${MAX_PER_RANK_BATCH_SIZE}" ]; then
  echo "Refusing prepare+queue shared: shared batch size must be in 1..${MAX_PER_RANK_BATCH_SIZE} per rank." >&2
  exit 1
fi

if [ $((SHARED_BATCH_SIZE * 2)) -gt "${MAX_GLOBAL_BATCH_SIZE}" ]; then
  echo "Refusing prepare+queue shared: shared global batch $((SHARED_BATCH_SIZE * 2)) exceeds cap ${MAX_GLOBAL_BATCH_SIZE}." >&2
  exit 1
fi

if [ "${SHARED_NUM_WORKERS}" -ne 0 ] || [ "${MAX_NUM_WORKERS}" -ne 0 ]; then
  echo "Refusing prepare+queue shared: shared smoke num_workers must remain 0." >&2
  exit 1
fi

if [ "${MAX_TRAIN_CROP_SIZE}" -lt 1 ] || [ "${MAX_VAL_CROP_SIZE}" -lt 1 ]; then
  echo "Refusing prepare+queue shared: shared crop caps must be >= 1." >&2
  exit 1
fi

if [ "${TRAIN_CROP_SIZE}" -lt 1 ] || [ "${TRAIN_CROP_SIZE}" -gt "${MAX_TRAIN_CROP_SIZE}" ]; then
  echo "Refusing prepare+queue shared: train crop size must be in 1..${MAX_TRAIN_CROP_SIZE}." >&2
  exit 1
fi

if [ "${VAL_CROP_SIZE}" -lt 1 ] || [ "${VAL_CROP_SIZE}" -gt "${MAX_VAL_CROP_SIZE}" ]; then
  echo "Refusing prepare+queue shared: val crop size must be in 1..${MAX_VAL_CROP_SIZE}." >&2
  exit 1
fi

if [ "${GPU_UTILIZATION_POLICY}" != "warn" ]; then
  echo "Refusing prepare+queue shared: TWODGAUSS_GPU_UTILIZATION_POLICY must be warn." >&2
  exit 1
fi

if [ -z "${OUTPUT_ROOT}" ]; then
  OUTPUT_ROOT="${DATA_ROOT%/}/generated_pairs/shared_ddp_smoke"
fi

if [ -z "${CONFIG_OUTPUT}" ]; then
  CONFIG_OUTPUT="${RUN_DIR}/configs/a_dsh11_prepared_shared_ddp_bs12_smoke_uhd_video_deshake.toml"
fi

if [ "${WRITE_FRAMES}" != "0" ] && [ "${WRITE_FRAMES}" != "1" ]; then
  echo "Refusing prepare+queue shared: TWODGAUSS_DESHAKE_WRITE_FRAMES must be 0 or 1." >&2
  exit 1
fi

if [ "${WRITE_FRAMES}" = "1" ] && [ "${CONFIRM_PREP}" != "${CONFIRM_PREP_TOKEN}" ]; then
  echo "Refusing prepare+queue shared: writing frames requires TWODGAUSS_DESHAKE_CONFIRM_PREP_WRITE=${CONFIRM_PREP_TOKEN}." >&2
  exit 1
fi

PREP_ARGS=(
  "${PREP_SCRIPT}"
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --train-pair "${TRAIN_PAIR}"
  --val-pair "${VAL_PAIR}"
  --train-frames-per-pair "${TRAIN_FRAMES_PER_PAIR}"
  --val-frames-per-pair "${VAL_FRAMES_PER_PAIR}"
  --start-frame "${START_FRAME}"
  --stride "${STRIDE}"
  --json
)

if [ "${WRITE_FRAMES}" = "1" ]; then
  PREP_ARGS+=(--write --force)
fi

set +e
"${PYTHON_BIN}" "${PREP_ARGS[@]}" > "${PLAN_JSON}" 2>&1
PREP_STATUS=$?
set -e
cat "${PLAN_JSON}"

if [ "${PREP_STATUS}" -ne 0 ]; then
  echo "Refusing prepare+queue shared: paired-frame preparation failed with status ${PREP_STATUS}." >&2
  exit "${PREP_STATUS}"
fi

if [ "${WRITE_FRAMES}" != "1" ]; then
  echo "prepare_and_queue_shared_status=plan_only"
  echo "plan_json=${PLAN_JSON}"
  echo "No frames, config, or shared queue smoke were created. Set TWODGAUSS_DESHAKE_WRITE_FRAMES=1 with the confirmation token after verifying explicit input/target pairs." >&2
  exit 0
fi

TRAIN_INPUT="${OUTPUT_ROOT%/}/train/input"
TRAIN_TARGET="${OUTPUT_ROOT%/}/train/target"
VAL_INPUT="${OUTPUT_ROOT%/}/val/input"
VAL_TARGET="${OUTPUT_ROOT%/}/val/target"

set +e
"${PYTHON_BIN}" "${CONFIG_SCRIPT}" \
  --data-root "${DATA_ROOT}" \
  --train-input "${TRAIN_INPUT}" \
  --train-target "${TRAIN_TARGET}" \
  --val-input "${VAL_INPUT}" \
  --val-target "${VAL_TARGET}" \
  --output "${CONFIG_OUTPUT}" \
  --run-id "${RUN_ID}" \
  --batch-size "${SHARED_BATCH_SIZE}" \
  --max-ddp-batch-size "${MAX_PER_RANK_BATCH_SIZE}" \
  --num-workers "${SHARED_NUM_WORKERS}" \
  --max-steps "${MAX_STEPS}" \
  --wall-clock-minutes "${WALL_CLOCK_MINUTES}" \
  --train-limit "${TRAIN_LIMIT}" \
  --val-limit "${VAL_LIMIT}" \
  --train-crop-size "${TRAIN_CROP_SIZE}" \
  --val-crop-size "${VAL_CROP_SIZE}" \
  --force \
  --json > "${CONFIG_JSON}" 2>&1
CONFIG_STATUS=$?
set -e
cat "${CONFIG_JSON}"

if [ "${CONFIG_STATUS}" -ne 0 ]; then
  echo "Refusing prepare+queue shared: config generation failed with status ${CONFIG_STATUS}." >&2
  exit "${CONFIG_STATUS}"
fi

export TWODGAUSS_DESHAKE_DATA_ROOT="${DATA_ROOT}"
export TWODGAUSS_DESHAKE_CONFIG="${CONFIG_OUTPUT}"
export TWODGAUSS_DRY_RUN="${DRY_RUN}"
export TWODGAUSS_GPU_UTILIZATION_POLICY="${GPU_UTILIZATION_POLICY}"
export TWODGAUSS_CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION}"
export TWODGAUSS_MIN_FREE_MIB="${MIN_FREE_MIB}"
export TWODGAUSS_MAX_PER_RANK_BATCH_SIZE="${MAX_PER_RANK_BATCH_SIZE}"
export TWODGAUSS_MAX_GLOBAL_BATCH_SIZE="${MAX_GLOBAL_BATCH_SIZE}"
export TWODGAUSS_MAX_NUM_WORKERS="${MAX_NUM_WORKERS}"
export TWODGAUSS_MAX_TRAIN_CROP_SIZE="${MAX_TRAIN_CROP_SIZE}"
export TWODGAUSS_MAX_VAL_CROP_SIZE="${MAX_VAL_CROP_SIZE}"
export TWODGAUSS_RUNTIME_MIN_FREE_MIB="${RUNTIME_MIN_FREE_MIB}"
export TWODGAUSS_RUNTIME_POLL_SECONDS="${RUNTIME_POLL_SECONDS}"
export TWODGAUSS_RUNTIME_STOP_GRACE_SECONDS="${RUNTIME_STOP_GRACE_SECONDS}"

set +e
"${QUEUE_WRAPPER}" > "${QUEUE_LOG}" 2>&1
QUEUE_STATUS=$?
set -e
cat "${QUEUE_LOG}"

echo "prepare_and_queue_shared_status=queued_shared_wrapper_exited"
echo "plan_json=${PLAN_JSON}"
echo "config_json=${CONFIG_JSON}"
echo "queue_log=${QUEUE_LOG}"
exit "${QUEUE_STATUS}"
