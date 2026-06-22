# G_4kd02 Manifest

- Created: `2026-06-21 19:05:00 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd02`
- First smoke run: `G_4kd02_smoke01`
- Status: `smoke01 deployed/evaluated; not promoted; formal training blocked`

## Difference From G_4kd01

`G_4kd01_smoke04` proved paired A08/A15 scalar anchors are live but still interfere through the same final selected output. `G_4kd02` changes the mechanism by supervising the existing Gaussian route-control high/low branches directly:

- high branch restored output is anchored to A08 only on A08-better crops;
- low branch restored output is anchored to A15 only on A15-better crops;
- a small non-target delta budget discourages the low branch from moving on A08-positive crops and the high branch from moving on A15-positive crops.

The Gaussian core remains unchanged.

## Smoke01

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02/configs/g_4kd02_smoke01_branch_isolated_control_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02/evaluate_smoke01_gpu.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/best_val.pt`
- Scope: physical GPU0/3 only, single visible GPU, `max_steps=40`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- Trainable prefix: `gaussian_route_control` only.

## Smoke01 Outcome

- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02/logs/smoke01_20260621_191037.log`
- Best eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02/logs/eval_smoke01_best_val_20260621_191309.log`
- Final eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02/logs/eval_smoke01_final_20260621_191635.log`
- Run dir: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02_smoke01`
- Summary: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd02_smoke01/metrics/g_4kd02_smoke01_a37_aligned_summary.json`
- GPU: physical GPU3 only, `CUDA_VISIBLE_DEVICES=3`, no CPU fallback, no DDP.
- Train exit code: `0`; final step `40`; best step `10`; peak CUDA reserved `3.091796875GiB`.
- Result: branch-level supervision is live and improves high/A08 more clearly than `G_4kd01_smoke04`, but still spends low/A15 surplus, so it is not promotable.

## Promotion Boundary

`G_4kd02_smoke01` is promotable only if it improves high-route and A08-oracle over A37 while keeping low-route and A15-oracle at or above A37. Aggregate PSNR alone cannot promote.
