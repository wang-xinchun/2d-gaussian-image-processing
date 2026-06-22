# Gate Check: V1/T01/A_4kd16

Status: evidence-demoted diagnostic; static checks passed, but formal GPU3 preflight queue was stopped before launch.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd16`.
- [x] Full run identifier: `V1/T01/A_4kd16`.
- [x] Run folder: `experiments/V1/runs/A_4kd16`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd08`, `A_4kd15`, and earlier diagnostics.

## Dataset Check

- [x] Dataset name: `4KDehaze_local_90_5_5_fast_probe_512`.
- [x] Train split expected: `512` paired files.
- [x] Validation split expected: `100` paired files.
- [x] Test split expected: `100` paired files.
- [ ] Formal preflight data load.

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
- [ ] Formal GPU preflight stopped before launch.
- [ ] Formal launch stopped before launch.
- [ ] CPU standard test eval.
- [ ] CPU task-specific diagnostics.

## Model / Loss Check

- [x] Keeps base lightweight `V1/T01` geometry.
- [x] Keeps `A_4kd08` medium detail gate and blended shared demand-control.
- [x] Does not enable reliability gating.
- [x] Keeps no haze guards: `dark_channel_weight=0.0`, `saturation_weight=0.0`.
- [x] Enables PSNR-aligned MSE with `mse_weight=0.25`.
- [x] Enables low-demand identity protection with `low_demand_identity_weight=0.03`.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe only if
free memory stays above the configured floor and the project self-limits CUDA
memory.

## Static Check Result

- TOML parse and key checks passed for template/run-local configs.
- Modified train/eval/helper scripts compile under the project venv.
- Low-demand identity loss random-tensor smoke check passed.
- `launch_tmux.sh` and `wait_and_launch_a_4kd16_gpu_safe.sh` pass `bash -n`.
- Launcher dry-run passed with `TWODGAUSS_MIN_FREE_MIB=0`.
- Formal preflight wait queue was initially started as `2dgauss_wait_a_4kd16_preflight_gpu_safe`, but it was stopped before GPU launch after targeted analysis showed that the watched A15/A08 tail was not a true low-input-demand case. The A16 package is retained as a diagnostic artifact; the active replacement line is `A_4kd17` A08-teacher consistency.

## Demotion Note

- The intended low-demand identity mask used `threshold=0.06`; however, the watched A15-worst high-A08 samples have center-crop input-target L1 around `0.30-0.38` and input PSNR around `8-10dB`.
- Pulling predictions toward the hazy input is therefore unlikely to repair that tail and may be harmful.
- A15/A08 per-sample analysis instead showed that `A15 - A08` PSNR delta is strongly anti-correlated with A08 PSNR (`r=-0.7016`), motivating A17's teacher-output anchoring rather than input-output anchoring.
