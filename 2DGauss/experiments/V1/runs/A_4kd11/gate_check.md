# Gate Check: V1/T01/A_4kd11

Status: completed diagnostic; not promoted over `A_4kd08`.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd11`.
- [x] Full run identifier: `V1/T01/A_4kd11`.
- [x] Run folder: `experiments/V1/runs/A_4kd11`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd08`, `A_4kd09`, and `A_4kd10`.

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
- [x] Best checkpoint selection metric: `val_psnr`.
- [x] Loads `A_4kd08` best-val checkpoint strictly.
- [x] Uses low LR `3e-5`, linear decay to `0.25x`, and `grad_clip_norm=1.0`.
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
- [x] Adds a light saturation guard and slightly stronger dark-channel guard.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe only if
free memory stays above the configured floor and the project self-limits CUDA
memory.

## Result

- Formal preflight passed on physical GPU3 only; strict load from `A_4kd08` best-val step `700`; preflight `val_psnr=17.688662408469433`.
- First formal launch failed before step1 because the new LR schedule code missed `base_lrs` scope; no checkpoint was produced by that failed launch.
- After patching `scripts/train_v1_t01.py`, formal training completed `300` steps with `train_exit_code=0`.
- Best validation was step `200`: `val_psnr=18.456089733874084`, `val_loss=0.11976054381579161`.
- CPU standard test on `best_val.pt`: `psnr_mean=17.36895205643582`, `loss_mean=0.12583659052848817`.
- CPU task diagnostics: `ssim_mean=0.6967320647835732`, `dark_channel_l1_mean=0.13161039672791958`, `saturation_l1_mean=0.0539228655770421`, `edge_l1_mean=0.15448181137442588`.
- Conclusion: useful diagnostic but not promoted. It improves validation/task proxies but loses too much test PSNR relative to `A_4kd08`.
