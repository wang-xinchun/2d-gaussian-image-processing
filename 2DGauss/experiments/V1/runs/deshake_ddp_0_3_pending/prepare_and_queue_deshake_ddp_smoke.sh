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
QUEUE_WRAPPER="${RUN_DIR}/wait_and_launch_deshake_ddp_smoke.sh"

DATA_ROOT="${TWODGAUSS_DESHAKE_DATA_ROOT:-}"
OUTPUT_ROOT="${TWODGAUSS_DESHAKE_PREP_OUTPUT_ROOT:-}"
TRAIN_PAIR="${TWODGAUSS_DESHAKE_TRAIN_PAIR:-}"
VAL_PAIR="${TWODGAUSS_DESHAKE_VAL_PAIR:-}"
WRITE_FRAMES="${TWODGAUSS_DESHAKE_WRITE_FRAMES:-0}"
CONFIRM_PREP="${TWODGAUSS_DESHAKE_CONFIRM_PREP_WRITE:-}"
CONFIRM_PREP_TOKEN="prepare_explicit_deshake_pairs"
CONFIG_OUTPUT="${TWODGAUSS_DESHAKE_CONFIG_OUTPUT:-}"
RUN_ID="${TWODGAUSS_DESHAKE_RUN_ID:-A_dsh05_prepared_ddp_smoke}"
TRAIN_FRAMES_PER_PAIR="${TWODGAUSS_DESHAKE_TRAIN_FRAMES_PER_PAIR:-16}"
VAL_FRAMES_PER_PAIR="${TWODGAUSS_DESHAKE_VAL_FRAMES_PER_PAIR:-4}"
START_FRAME="${TWODGAUSS_DESHAKE_START_FRAME:-0}"
STRIDE="${TWODGAUSS_DESHAKE_FRAME_STRIDE:-60}"
TRAIN_LIMIT="${TWODGAUSS_DESHAKE_TRAIN_LIMIT:-128}"
VAL_LIMIT="${TWODGAUSS_DESHAKE_VAL_LIMIT:-8}"
MAX_STEPS="${TWODGAUSS_DESHAKE_MAX_STEPS:-20}"
WALL_CLOCK_MINUTES="${TWODGAUSS_DESHAKE_WALL_CLOCK_MINUTES:-20}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-1}"

PLAN_JSON="${LOG_DIR}/prepare_and_queue_plan_${STAMP}.json"
CONFIG_JSON="${LOG_DIR}/prepare_and_queue_config_${STAMP}.json"
QUEUE_LOG="${LOG_DIR}/prepare_and_queue_queue_${STAMP}.log"

if [ -z "${DATA_ROOT}" ]; then
  echo "Refusing prepare+queue: TWODGAUSS_DESHAKE_DATA_ROOT is required." >&2
  exit 1
fi

if [ -z "${TRAIN_PAIR}" ]; then
  echo "Refusing prepare+queue: TWODGAUSS_DESHAKE_TRAIN_PAIR is required as INPUT_VIDEO:TARGET_VIDEO." >&2
  exit 1
fi

if [ -z "${VAL_PAIR}" ]; then
  echo "Refusing prepare+queue: TWODGAUSS_DESHAKE_VAL_PAIR is required as INPUT_VIDEO:TARGET_VIDEO." >&2
  exit 1
fi

for value_name in TRAIN_FRAMES_PER_PAIR VAL_FRAMES_PER_PAIR START_FRAME STRIDE TRAIN_LIMIT VAL_LIMIT MAX_STEPS; do
  value="${!value_name}"
  case "${value}" in
    ''|*[!0-9]*)
      echo "Refusing prepare+queue: ${value_name} must be a non-negative integer." >&2
      exit 1
      ;;
  esac
done

if [ "${TRAIN_FRAMES_PER_PAIR}" -lt 1 ] || [ "${VAL_FRAMES_PER_PAIR}" -lt 1 ]; then
  echo "Refusing prepare+queue: frame counts per pair must be >= 1." >&2
  exit 1
fi

if [ "${STRIDE}" -lt 1 ]; then
  echo "Refusing prepare+queue: TWODGAUSS_DESHAKE_FRAME_STRIDE must be >= 1." >&2
  exit 1
fi

if [ "${MAX_STEPS}" -lt 1 ]; then
  echo "Refusing prepare+queue: TWODGAUSS_DESHAKE_MAX_STEPS must be >= 1." >&2
  exit 1
fi

if [ -z "${OUTPUT_ROOT}" ]; then
  OUTPUT_ROOT="${DATA_ROOT%/}/generated_pairs/ddp_smoke"
fi

if [ -z "${CONFIG_OUTPUT}" ]; then
  CONFIG_OUTPUT="${RUN_DIR}/configs/a_dsh05_prepared_ddp_smoke_uhd_video_deshake.toml"
fi

if [ "${WRITE_FRAMES}" != "0" ] && [ "${WRITE_FRAMES}" != "1" ]; then
  echo "Refusing prepare+queue: TWODGAUSS_DESHAKE_WRITE_FRAMES must be 0 or 1." >&2
  exit 1
fi

if [ "${WRITE_FRAMES}" = "1" ] && [ "${CONFIRM_PREP}" != "${CONFIRM_PREP_TOKEN}" ]; then
  echo "Refusing prepare+queue: writing frames requires TWODGAUSS_DESHAKE_CONFIRM_PREP_WRITE=${CONFIRM_PREP_TOKEN}." >&2
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
  echo "Refusing prepare+queue: paired-frame preparation failed with status ${PREP_STATUS}." >&2
  exit "${PREP_STATUS}"
fi

if [ "${WRITE_FRAMES}" != "1" ]; then
  echo "prepare_and_queue_status=plan_only"
  echo "plan_json=${PLAN_JSON}"
  echo "No frames, config, or queue smoke were created. Set TWODGAUSS_DESHAKE_WRITE_FRAMES=1 with the confirmation token after verifying explicit input/target pairs." >&2
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
  --max-steps "${MAX_STEPS}" \
  --wall-clock-minutes "${WALL_CLOCK_MINUTES}" \
  --train-limit "${TRAIN_LIMIT}" \
  --val-limit "${VAL_LIMIT}" \
  --force \
  --json > "${CONFIG_JSON}" 2>&1
CONFIG_STATUS=$?
set -e
cat "${CONFIG_JSON}"

if [ "${CONFIG_STATUS}" -ne 0 ]; then
  echo "Refusing prepare+queue: config generation failed with status ${CONFIG_STATUS}." >&2
  exit "${CONFIG_STATUS}"
fi

export TWODGAUSS_DESHAKE_DATA_ROOT="${DATA_ROOT}"
export TWODGAUSS_DESHAKE_CONFIG="${CONFIG_OUTPUT}"
export TWODGAUSS_DRY_RUN="${DRY_RUN}"

set +e
"${QUEUE_WRAPPER}" > "${QUEUE_LOG}" 2>&1
QUEUE_STATUS=$?
set -e
cat "${QUEUE_LOG}"

echo "prepare_and_queue_status=queued_wrapper_exited"
echo "plan_json=${PLAN_JSON}"
echo "config_json=${CONFIG_JSON}"
echo "queue_log=${QUEUE_LOG}"
exit "${QUEUE_STATUS}"
