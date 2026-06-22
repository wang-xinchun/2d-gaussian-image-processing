# G_4kd05 Manifest

- Created: `2026-06-21 21:22:35 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd05`
- First smoke run: `G_4kd05_smoke01`
- Status: `evaluated_not_promoted_near_miss; formal training blocked`

## Difference From G_4kd04

`G_4kd04_smoke01` proved step-level rollback can prevent most low/A15-harmful updates from accumulating, but it protected `G_4kd02_smoke01/best_val.pt`, which is already below A37 on low-route and A15-oracle. `G_4kd05` moves the same no-harm concept to an A37-equivalent starting point:

- initialize from `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`;
- keep the G-RSCN version line and trainable prefix separate from old A-line experiments;
- train only `gaussian_route_control`;
- use full `24`-batch validation for the step no-harm guard;
- require low-route, A15-oracle, and low-A15 metrics to stay at or above the initialized A37-equivalent baseline.

## Smoke01

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05/configs/g_4kd05_smoke01_a37_fullval_no_harm_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05/evaluate_smoke01_gpu.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Scope: physical GPU0/3 only, single visible GPU, `max_steps=24`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- Step no-harm guard: enabled every `2` steps, `24` validation batches, restore optimizer state on rollback.
- Trainable prefix: `gaussian_route_control` only.

## Result

- Train exit code: `0`.
- Eval exit codes: `0` for both `best_val` and `final`.
- Peak CUDA reserved: `3.095703125GiB`.
- Guard checks: `12`.
- Guard accepted updates: `5`.
- Guard rollbacks: `7`.
- Best checkpoint: step `0`, exactly A37-aligned baseline.
- Final checkpoint: step `24`.
- Summary: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05_smoke01/metrics/g_4kd05_smoke01_a37_aligned_summary.json`

## Promotion Boundary

`G_4kd05_smoke01` is not promotable, but it is a near miss. Final improves aggregate, low-route, and A15-oracle over A37; it fails only by tiny high-route and A08-oracle margins. The next G-RSCN version should keep the full-validation no-harm guard and add minimal high/A08 lift.
