# Gate Check: V1/T01/A_4kd14

Status: completed; static checks, formal GPU3 preflight, training, CPU standard test eval, and CPU task diagnostics passed.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd14`.
- [x] Full run identifier: `V1/T01/A_4kd14`.
- [x] Run folder: `experiments/V1/runs/A_4kd14`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd08`, `A_4kd11`, and `A_4kd12`.

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
- [x] Keeps relaxed haze guards from `A_4kd13`: `dark_channel_weight=0.010`, `saturation_weight=0.002`.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe only if
free memory stays above the configured floor and the project self-limits CUDA
memory.

## Results

- Formal preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd14/logs/train_20260611_191853.log`.
- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd14/logs/train_20260611_192115.log`.
- Training exit code: `0`.
- Best EMA validation: step `250`, `val_psnr=18.223114904672048`, `val_loss=0.12495512887835503`.
- CPU standard test: `psnr_mean=17.774995041050754`, `loss_mean=0.12244070835411548`, `total_loss_mean=0.12567226648330687`.
- CPU task diagnostics: `ssim_mean=0.6976614135503769`, `dark_channel_l1_mean=0.14586309671401979`, `saturation_l1_mean=0.055857277773320675`, `edge_magnitude_l1_mean=0.16153054259717464`, `high_frequency_l1_mean=0.021660706866532563`, `local_contrast_l1_mean=0.025637199571356177`, `luminance_l1_mean=0.10508509073406458`, `chroma_l1_mean=0.03105577051639557`.
- Candidate decision: not a strict `A_4kd08` PSNR supersede; useful high-EMA diagnostic and closest EMA continuation to `A_4kd08`.
