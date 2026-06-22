# Gate Check: V1/T01/A_4kd13

Status: completed relaxed haze-guard EMA diagnostic; not promoted over `A_4kd08`.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd13`.
- [x] Full run identifier: `V1/T01/A_4kd13`.
- [x] Run folder: `experiments/V1/runs/A_4kd13`.
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
- [x] Uses EMA validation/checkpointing with `ema_decay=0.99`.
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
- [x] Relaxes haze guards from `A_4kd12`: `dark_channel_weight=0.010`, `saturation_weight=0.002`.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe only if
free memory stays above the configured floor and the project self-limits CUDA
memory.

## Preflight Result

- Static checks passed: TOML parse, identity check, train/eval py_compile, launcher `bash -n`, and GPU3 dry-run.
- Formal preflight passed on physical GPU3 only with `CUDA_VISIBLE_DEVICES=3`, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`, strict load from `A_4kd08` best-val step `700`, and no missing/unexpected checkpoint keys.
- Preflight metrics: `val_samples=100`, `val_psnr=17.68881486831974`, `val_loss=0.13449759554117918`.
- After preflight, an external GPU3 process was using about `38160MiB`, leaving about `10356MiB`; formal launch is intentionally queued rather than forced.

## Result

- Wait queue later passed the GPU3 gate and launched formal training with `gpu_free_mib=38584`, `min_free_mib=12288`.
- Formal training completed `300` steps with `train_exit_code=0`; peak reserved CUDA memory about `11.2265625GiB`.
- Best EMA validation was step `250`: `val_psnr=18.222453652339333`, `val_loss=0.12469094827771186`.
- CPU standard test on EMA `best_val.pt`: `psnr_mean=17.73016814214326`, `loss_mean=0.12258190430700779`, `total_loss_mean=0.1258132006973028`.
- CPU task diagnostics: `ssim_mean=0.6996025824546814`, `dark_channel_l1_mean=0.1437134078145027`, `saturation_l1_mean=0.05548523813486099`, `edge_magnitude_l1_mean=0.15986251048743724`.
- Conclusion: marginally improves `A_4kd12` on PSNR/loss/SSIM, but does not close the strict PSNR gap to `A_4kd08`.
