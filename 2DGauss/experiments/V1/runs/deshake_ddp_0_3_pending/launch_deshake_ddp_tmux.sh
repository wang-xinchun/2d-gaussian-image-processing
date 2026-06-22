#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/ldj/wxc/2DGauss"
cd "${PROJECT_ROOT}"

SESSION="${TWODGAUSS_DESHAKE_SESSION:-2dgauss_deshake_ddp_0_3_pending}"
RUN_DIR="experiments/V1/runs/deshake_ddp_0_3_pending"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/train_deshake_ddp_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log"
PREFLIGHT_JSON="${LOG_PATH}.preflight.json"
PYTHON_BIN="${TWODGAUSS_PYTHON:-/home/ldj/wxc/.venvs/2dgauss-py313/bin/python}"

PHYSICAL_GPUS="${TWODGAUSS_PHYSICAL_GPUS:-0,3}"
MIN_FREE_MIB="${TWODGAUSS_MIN_FREE_MIB:-30000}"
MAX_GPU_UTILIZATION="${TWODGAUSS_MAX_GPU_UTILIZATION:-20}"
GPU_UTILIZATION_POLICY="${TWODGAUSS_GPU_UTILIZATION_POLICY:-block}"
MAX_PER_RANK_BATCH_SIZE="${TWODGAUSS_MAX_PER_RANK_BATCH_SIZE:-2}"
MAX_GLOBAL_BATCH_SIZE="${TWODGAUSS_MAX_GLOBAL_BATCH_SIZE:-4}"
MAX_NUM_WORKERS="${TWODGAUSS_MAX_NUM_WORKERS:-2}"
MAX_TRAIN_CROP_SIZE="${TWODGAUSS_MAX_TRAIN_CROP_SIZE:-0}"
MAX_VAL_CROP_SIZE="${TWODGAUSS_MAX_VAL_CROP_SIZE:-0}"
MIN_DATA_BYTES="${TWODGAUSS_DESHAKE_MIN_DATA_BYTES:-322122547200}"
MIN_VIDEO_FILES="${TWODGAUSS_DESHAKE_MIN_VIDEO_FILES:-1}"
MIN_FRAME_FILES="${TWODGAUSS_DESHAKE_MIN_FRAME_FILES:-0}"
MIN_TRAIN_PAIRS="${TWODGAUSS_DESHAKE_MIN_TRAIN_PAIRS:-1}"
MIN_VAL_PAIRS="${TWODGAUSS_DESHAKE_MIN_VAL_PAIRS:-1}"
MAX_SCAN_FILES="${TWODGAUSS_DESHAKE_MAX_SCAN_FILES:-500000}"
MIN_FILE_AGE_SECONDS="${TWODGAUSS_DESHAKE_MIN_FILE_AGE_SECONDS:-600}"
CUDA_MEMORY_FRACTION="${TWODGAUSS_CUDA_MEMORY_FRACTION:-}"
DATA_ROOT="${TWODGAUSS_DESHAKE_DATA_ROOT:-}"
CONFIG="${TWODGAUSS_DESHAKE_CONFIG:-}"
DRY_RUN="${TWODGAUSS_DRY_RUN:-0}"

if [ -z "${DATA_ROOT}" ]; then
  echo "Refusing to launch: TWODGAUSS_DESHAKE_DATA_ROOT is required." >&2
  exit 1
fi

if [ -z "${CONFIG}" ]; then
  echo "Refusing to launch: TWODGAUSS_DESHAKE_CONFIG is required." >&2
  exit 1
fi

if [ "${PHYSICAL_GPUS}" != "0,3" ]; then
  echo "Refusing to launch: deshake DDP package is GPU0+GPU3-only, got ${PHYSICAL_GPUS}" >&2
  exit 1
fi

set +e
"${PYTHON_BIN}" scripts/preflight_deshake_ddp_gpu0_3.py \
  --data-root "${DATA_ROOT}" \
  --config "${CONFIG}" \
  --physical-gpus "${PHYSICAL_GPUS}" \
  --min-free-mib "${MIN_FREE_MIB}" \
  --max-gpu-utilization "${MAX_GPU_UTILIZATION}" \
  --gpu-utilization-policy "${GPU_UTILIZATION_POLICY}" \
  --max-per-rank-batch-size "${MAX_PER_RANK_BATCH_SIZE}" \
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
  echo "Refusing to launch: deshake DDP preflight failed with status ${PREFLIGHT_STATUS}" >&2
  exit "${PREFLIGHT_STATUS}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "dry_run=1"
  echo "would_launch_session=${SESSION}"
  echo "physical_gpus=${PHYSICAL_GPUS}"
  echo "min_free_mib=${MIN_FREE_MIB}"
  echo "max_gpu_utilization=${MAX_GPU_UTILIZATION}"
  echo "gpu_utilization_policy=${GPU_UTILIZATION_POLICY}"
  echo "max_per_rank_batch_size=${MAX_PER_RANK_BATCH_SIZE}"
  echo "max_global_batch_size=${MAX_GLOBAL_BATCH_SIZE}"
  echo "max_num_workers=${MAX_NUM_WORKERS}"
  echo "max_train_crop_size=${MAX_TRAIN_CROP_SIZE}"
  echo "max_val_crop_size=${MAX_VAL_CROP_SIZE}"
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
  echo "python_bin=${PYTHON_BIN}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  cd '${PROJECT_ROOT}' && \
  export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
  export CUDA_VISIBLE_DEVICES='${PHYSICAL_GPUS}' && \
  export TWODGAUSS_ALLOWED_PHYSICAL_GPUS='0,3' && \
  export TWODGAUSS_ALLOW_MULTI_GPU_VISIBLE=1 && \
  export OMP_NUM_THREADS=2 && \
  export MKL_NUM_THREADS=2 && \
  export OPENBLAS_NUM_THREADS=2 && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  export TWODGAUSS_CUDA_MEMORY_FRACTION='${CUDA_MEMORY_FRACTION}' && \
  export PYTHONUNBUFFERED=1 && \
  export PYTHONFAULTHANDLER=1 && \
  export NCCL_ASYNC_ERROR_HANDLING=1 && \
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 && \
  set -o pipefail; \
  printf 'classification=%s\n' 'current_deshake_ddp_gpu0_3' | tee '${LOG_PATH}'; \
  printf 'data_root=%s\n' '${DATA_ROOT}' | tee -a '${LOG_PATH}'; \
  printf 'config=%s\n' '${CONFIG}' | tee -a '${LOG_PATH}'; \
  '${PYTHON_BIN}' -u -m torch.distributed.run --standalone --nproc_per_node=2 --max-restarts=0 scripts/train_v1_t01.py --config '${CONFIG}' 2>&1 | tee -a '${LOG_PATH}'; \
  status=\${PIPESTATUS[0]}; \
  printf 'train_exit_code=%s\n' \"\${status}\" | tee -a '${LOG_PATH}'; \
  printf '%s\n' \"\${status}\" > '${LOG_PATH}.exit_code'; \
  exit \"\${status}\"
"

echo "session=${SESSION}"
echo "physical_gpus=${PHYSICAL_GPUS}"
echo "min_free_mib=${MIN_FREE_MIB}"
echo "max_gpu_utilization=${MAX_GPU_UTILIZATION}"
echo "gpu_utilization_policy=${GPU_UTILIZATION_POLICY}"
echo "max_per_rank_batch_size=${MAX_PER_RANK_BATCH_SIZE}"
echo "max_global_batch_size=${MAX_GLOBAL_BATCH_SIZE}"
echo "max_num_workers=${MAX_NUM_WORKERS}"
echo "max_train_crop_size=${MAX_TRAIN_CROP_SIZE}"
echo "max_val_crop_size=${MAX_VAL_CROP_SIZE}"
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
echo "python_bin=${PYTHON_BIN}"
