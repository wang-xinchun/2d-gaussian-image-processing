# G_4kd10 Manifest

- Created: `2026-06-22 00:15:44 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd10`
- First smoke run: `G_4kd10_smoke01`
- Status: `completed; all updates rolled back; formal training blocked`

## Difference From G_4kd09

`G_4kd09_smoke01` proved that a positive-margin four-gate guard protects low-route/A15 but can reject every update; its closest candidate missed only the A08 margin while high-route, low-route, and A15 passed. `G_4kd10` keeps the separate G-RSCN version line and changes only the narrow bottleneck:

- initialize from `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`;
- keep `gaussian_route_control` as the only trainable prefix;
- keep full 24-batch step no-harm guard on high-route, A08-oracle, low-route, A15-oracle, and low-A15 validation metrics;
- relax only the A08 validation margin from G09 `-0.10020664672851562` to `-0.10025`;
- keep high/low/A15 guard margins at G09 levels;
- slightly increase high/A08 pressure while reducing LR: A08 anchor `0.030 -> 0.032`, high branch anchor `0.075 -> 0.080`, LR `3.5e-6 -> 3.2e-6`.

## Smoke01 Artifacts

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10/configs/g_4kd10_smoke01_a37_asymmetric_a08_margin_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10/evaluate_smoke01_gpu.sh`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10/logs/smoke01_20260622_122138.log`
- Guard summary: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd10_smoke01/metrics/g_4kd10_smoke01_guard_summary.json`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Scope: physical GPU0/3 only, single visible GPU, `max_steps=18`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- Step no-harm guard: enabled every `2` steps, `24` validation batches, restore optimizer state on rollback.

## Outcome

Training exited with code `0` on physical GPU3. The no-harm guard ran `9` checks and rolled back all `9` updates. No candidate was accepted, so best/final/latest carry no new weight signal and A37-aligned GPU evaluation was intentionally skipped.

G10 confirmed the G09 diagnosis only partially. Relaxing the A08 margin made a high/A08-positive candidate possible at step `4`, with high-route margin `+0.00030991278754340275` and A08-oracle margin `+0.00005888366699219327`. That same candidate failed low-route, A15-oracle, and low-A15. The closest all-gate candidate was step `18`, but it missed every margin slightly, with worst margin `-0.00016272430419922435`.

## Promotion Boundary

`G_4kd10_smoke01` is not promotable. Aggregate PSNR alone cannot promote, and no accepted weight update exists. Formal long training remains blocked.
