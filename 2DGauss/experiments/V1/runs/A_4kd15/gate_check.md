# Gate Check: V1/T01/A_4kd15

Status: completed; static checks, formal GPU3 preflight, training, CPU standard test eval, and CPU task diagnostics passed.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd15`.
- [x] Full run identifier: `V1/T01/A_4kd15`.
- [x] Run folder: `experiments/V1/runs/A_4kd15`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd08`, `A_4kd14`, and earlier diagnostics.

## Dataset Check

- [x] Dataset name: `4KDehaze_local_90_5_5_fast_probe_512`.
- [x] Train split expected: `512` paired files.
- [x] Validation split expected: `100` paired files.
- [x] Test split expected: `100` paired files.
- [x] Formal preflight data load.

## Runtime Check

- [x] Planned batch size: `2`.
- [x] Planned gradient accumulation: `4`.
- [x] Planned workers: `1`.
- [x] Planned train crop: `512`.
- [x] Planned validation crop: `512`.
- [x] Planned max steps: `300`.
- [x] Planned wall-clock cap: `45` minutes.
- [x] Formal launch free-memory floor: `12288MiB`.
- [x] Best checkpoint selection metric: `val_psnr`.
- [x] Loads `A_4kd08` best-val checkpoint strictly.
- [x] Uses low LR `3e-5`, linear decay to `0.25x`, and `grad_clip_norm=1.0`.
- [x] Uses EMA validation/checkpointing with `ema_decay=0.995`.
- [x] Only physical GPU `0` or `3` may be used.
- [x] Default launch preference is physical GPU `3`.
- [x] GPU1/GPU2 are forbidden.
- [x] Formal GPU preflight.
- [x] Formal launch.
- [x] CPU standard test eval.
- [x] CPU task-specific diagnostics.

## Model / Loss Check

- [x] Keeps base lightweight `V1/T01` geometry.
- [x] Keeps `A_4kd08` medium detail gate and blended shared demand-control.
- [x] Does not enable reliability gating.
- [x] Removes haze guards from `A_4kd14`: `dark_channel_weight=0.0`, `saturation_weight=0.0`.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe only if
free memory stays above the configured floor and the project self-limits CUDA
memory.

## Static Check Result

- TOML parse and key checks passed for template/run-local configs.
- `scripts/train_v1_t01.py`, `scripts/evaluate_v1_t01.py`, and `scripts/evaluate_task_specific_metrics.py` compile under the project venv.
- `launch_tmux.sh` and `wait_and_launch_a_4kd15_gpu_safe.sh` pass `bash -n`.
- Launcher dry-run passed with `TWODGAUSS_MIN_FREE_MIB=0`.
- Formal preflight wait queue started as `2dgauss_wait_a_4kd15_preflight_gpu_safe`; initial safety gate refusal was expected because GPU3 free memory was below `12288MiB`.
- Formal preflight later passed on GPU3 with `train_exit_code=0`, `val_psnr=18.54360101346333`, and `val_loss=0.11857988886535167`.
- Formal training launched on GPU3 with log `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/logs/train_20260611_210055.log`.

## Results

- Training exit code: `0`.
- Best EMA validation: step `300`, `val_psnr=18.24311035106881`, `val_loss=0.12273863352835178`.
- CPU standard test: `psnr_mean=17.781017099305146`, `loss_mean=0.12060526199638844`, `total_loss_mean=0.12383682064712048`.
- CPU task diagnostics: `ssim_mean=0.6972798517346382`, `dark_channel_l1_mean=0.14506316002458333`, `saturation_l1_mean=0.055948681086301806`, `edge_magnitude_l1_mean=0.16086940564215182`, `high_frequency_l1_mean=0.021652142368257046`, `local_contrast_l1_mean=0.02542852663435042`, `luminance_l1_mean=0.1051734821870923`, `chroma_l1_mean=0.030280336271971464`.
- Candidate decision: not a strict `A_4kd08` PSNR supersede; improves over `A_4kd14` but still misses `A_4kd08` by about `0.00812 dB`.
