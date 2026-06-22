#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/A_dsh01_predeploy"
DATA_ROOT="${DESHAKE_DATA_ROOT:-}"
INPUT_DIR="${DESHAKE_INPUT_DIR:-}"
TARGET_DIR="${DESHAKE_TARGET_DIR:-}"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
OUTPUT="${RUN_DIR}/metrics/dataset_inspection_${STAMP}.json"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"

if [ -z "${DATA_ROOT}" ]; then
  echo "Set DESHAKE_DATA_ROOT to the incoming UHD video deshaking dataset root." >&2
  exit 2
fi

ARGS=(--root "${DATA_ROOT}" --output "${OUTPUT}" --allow-unpaired)
if [ -n "${INPUT_DIR}" ] && [ -n "${TARGET_DIR}" ]; then
  ARGS+=(--input-dir "${INPUT_DIR}" --target-dir "${TARGET_DIR}")
fi

"${PYTHON_BIN}" scripts/inspect_deshake_dataset.py "${ARGS[@]}"
echo "dataset_inspection=${OUTPUT}"
