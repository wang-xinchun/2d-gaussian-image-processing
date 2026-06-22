# G_4kd04 Manifest

- Created: `2026-06-21 20:45:00 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd04`
- First smoke run: `G_4kd04_smoke01`
- Status: `evaluated_not_promoted; formal training blocked`

## Difference From G_4kd03

`G_4kd03_smoke01` proved that a protected validation selector can keep a low/A15-regressing checkpoint from being saved as best, but it did not change the training trajectory. `G_4kd04` moves the low/A15 guard into the update rule:

- after selected optimizer steps, a small router-validation pass checks low-route, A15-oracle, and low-A15 gaps against the initialized baseline;
- if any guard metric regresses below baseline, the model parameters and optimizer state are restored to the pre-step snapshot;
- the G_4kd03 best-selection gate remains active, but it is no longer the only protection;
- Gaussian core and trainable prefix remain unchanged.

## Smoke01

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04/configs/g_4kd04_smoke01_step_no_harm_rollback_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04/evaluate_smoke01_gpu.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02_smoke01/checkpoints/best_val.pt`
- Scope: physical GPU0/3 only, single visible GPU, `max_steps=40`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- Step no-harm guard: enabled every `2` steps, `8` validation batches, restore optimizer state on rollback.
- Trainable prefix: `gaussian_route_control` only.

## Result

- Train exit code: `0`.
- Eval exit codes: `0` for both `best_val` and `final`.
- Peak CUDA reserved: `3.11328125GiB`.
- Guard checks: `20`.
- Guard accepted updates: `4`.
- Guard rollbacks: `16`.
- Best checkpoint: step `0`.
- Final checkpoint: step `40`.
- Summary: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04_smoke01/metrics/g_4kd04_smoke01_a37_aligned_summary.json`

## Promotion Boundary

`G_4kd04_smoke01` is not promotable. It demonstrates that update rollback is active and useful, but it still fails A37 low-route and A15-oracle gates. The next version should use the rollback guard against an A37-equivalent starting point rather than against the already low/A15-below parent checkpoint.
