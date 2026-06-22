# Gate Check: V1/T01/A_4kd10

Status: prepared; awaiting static checks, formal preflight, and formal launch.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_4kd10`.
- [x] Full run identifier: `V1/T01/A_4kd10`.
- [x] Run folder: `experiments/V1/runs/A_4kd10`.
- [x] Run-local config exists.
- [x] Template config exists.
- [x] This run is independent from `A_4kd08` and `A_4kd09`.

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
- [x] Planned max steps: `800`.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Best checkpoint selection metric: `val_psnr`.
- [x] Only physical GPU `0` or `3` may be used.
- [x] Default launch preference is physical GPU `3`.
- [x] GPU1/GPU2 are forbidden.
- [x] Formal GPU preflight.
- [ ] Formal launch.
- [ ] First validation/checkpoint.

## Model / Loss Check

- [x] Keeps base lightweight `V1/T01` geometry.
- [x] Keeps `A_4kd08` medium detail-preserve gate.
- [x] Keeps `A_4kd08` protection losses.
- [x] Keeps default-off reliability-gate code and enables it only in `A_4kd10`.
- [x] Uses weaker/later reliability than `A_4kd09`.
- [x] Uses `restoration_demand` target only for loss/diagnostics, not forward GT.
- [x] Enables `training.control_diagnostics`.

## Launch Boundary

Use only physical GPU0 or GPU3. Prefer GPU3. Do not kill or displace external
processes. High utilization is acceptable for this low-footprint probe, but
the project must keep its own CUDA memory fraction low and refuse launch below
the configured free-memory floor.

## Preflight Evidence

- Formal preflight log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd10/logs/train_20260611_125645.log`
- Preflight exit code: `0`.
- Physical GPU used: `3`.
- `CUDA_VISIBLE_DEVICES`: `3`.
- CUDA memory fraction: `0.30`.
- Batch input shape: `(2, 3, 512, 512)`.
- Validation samples: `100`.
- Preflight validation:
  - `val_psnr=8.556741222446341`
  - `val_loss=0.39407436169683935`
- Reliability diagnostics:
  - `val_control_guidance_reliability_mean=0.9250248336791992`
  - `val_control_guidance_blend_mean=0.18808600217103957`
  - `val_control_guidance_blend_high_detail_mean=0.16343827188014984`
  - `val_control_guidance_blend_low_detail_mean=0.21279539227485655`
  - `val_control_guidance_blend_detail_corr=-0.849615718126297`

The low preflight PSNR is from the untrained model at `training_progress=1.0`;
the formal training hypothesis remains that this late-soft gate should stay
near `A_4kd08` behavior until the final quarter of training.
