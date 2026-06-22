#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

RUN_DIR="experiments/V1/runs/A_dsh01_predeploy"
INPUT_FRAMES="${DESHAKE_INPUT_FRAMES:-}"
PHYSICAL_GPU="${TWODGAUSS_PHYSICAL_GPU:-3}"
TILE_SIZE="${DESHAKE_TILE_SIZE:-1024}"
TILE_OVERLAP="${DESHAKE_TILE_OVERLAP:-96}"
LIMIT="${DESHAKE_FRAME_LIMIT:-0}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-24576}"
MAX_UTIL="${TWODGAUSS_MAX_UTIL:-60}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-0}"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${RUN_DIR}/baseline_outputs/frame_enhancer_${STAMP}"
SUMMARY="${RUN_DIR}/metrics/frame_enhancer_${STAMP}.json"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"
CONFIG="experiments/V1/runs/A_y03/configs/a_y03_medium_uhddeblur_feature_refinement_plus_edge_weighted_lightedge_5h_confirmation.toml"
CHECKPOINT="experiments/V1/runs/A_y03/checkpoints/final.pt"

if [ -z "${INPUT_FRAMES}" ]; then
  echo "Set DESHAKE_INPUT_FRAMES to a frame directory. This is a frame-enhancement baseline, not full deshaking." >&2
  exit 2
fi

if [ "${PHYSICAL_GPU}" != "0" ] && [ "${PHYSICAL_GPU}" != "3" ]; then
  echo "Refusing physical GPU ${PHYSICAL_GPU}; allowed project GPUs are 0 and 3." >&2
  exit 3
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  IFS=',' read -r FREE_MIB UTIL_PCT < <(
    nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i "${PHYSICAL_GPU}" |
      head -n 1 |
      tr -d ' '
  )
  echo "gpu=${PHYSICAL_GPU} free_mib=${FREE_MIB} util_pct=${UTIL_PCT} min_free_mib=${MIN_FREE_MIB} max_util_pct=${MAX_UTIL}"
  if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
    echo "Refusing to start: GPU${PHYSICAL_GPU} free memory is below the shared-server gate." >&2
    exit 4
  fi
  if [ "${UTIL_PCT}" -gt "${MAX_UTIL}" ]; then
    echo "Refusing to start: GPU${PHYSICAL_GPU} utilization is above the shared-server gate." >&2
    exit 5
  fi
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  echo "classification=frame_enhancement_baseline_not_full_video_deshaking"
  echo "input_frames=${INPUT_FRAMES}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "summary=${SUMMARY}"
  exit 0
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export TWODGAUSS_ALLOWED_PHYSICAL_GPUS="0,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

"${PYTHON_BIN}" scripts/run_v1_frame_enhancer.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --input-dir "${INPUT_FRAMES}" \
  --output-dir "${OUTPUT_DIR}" \
  --device auto \
  --tile-size "${TILE_SIZE}" \
  --tile-overlap "${TILE_OVERLAP}" \
  --limit "${LIMIT}" \
  --summary "${SUMMARY}"

echo "frame_enhancer_output=${OUTPUT_DIR}"
echo "frame_enhancer_summary=${SUMMARY}"
