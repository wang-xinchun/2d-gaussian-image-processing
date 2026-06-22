# G_4kd06 Manifest

- Created: `2026-06-21 21:55:31 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd06`
- First smoke run: `G_4kd06_smoke01`
- Status: `prepared for tiny high/A08 lift continuation; formal training blocked`

## Difference From G_4kd05

`G_4kd05_smoke01` nearly solved the Pareto bottleneck: final aggregate, low-route, and A15-oracle exceed A37, but high-route and A08-oracle remain slightly below A37. `G_4kd06` is a narrow continuation probe, not a broad new recipe:

- initialize from `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05_smoke01/checkpoints/final.pt`;
- keep the 24-batch step no-harm guard;
- keep `gaussian_route_control` as the only trainable prefix;
- reduce LR and steps to avoid spending the recovered low/A15 surplus;
- raise only the small high/A08 anchor pressure needed to close the tiny high/A08 gap.

## Smoke01

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd06/configs/g_4kd06_smoke01_tiny_high_a08_lift_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd06/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd06/evaluate_smoke01_gpu.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05_smoke01/checkpoints/final.pt`
- Scope: physical GPU0/3 only, single visible GPU, `max_steps=16`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- Step no-harm guard: enabled every `2` steps, `24` validation batches, restore optimizer state on rollback.
- Trainable prefix: `gaussian_route_control` only.

## Promotion Boundary

`G_4kd06_smoke01` is promotable only if A37-aligned GPU evaluation improves high-route and A08-oracle over A37 while keeping low-route and A15-oracle at or above A37. Aggregate PSNR alone still cannot promote. Formal long training is blocked until best/final GPU evaluation passes all route/oracle gates.
