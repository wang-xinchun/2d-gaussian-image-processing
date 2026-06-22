# G_4kd07 Manifest

- Created: `2026-06-21 22:20:07 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd07`
- First smoke run: `G_4kd07_smoke01`
- Status: `completed; all candidate updates rolled back; formal training blocked`

## Difference From G_4kd06

`G_4kd06_smoke01` accepted two updates and preserved aggregate/low/A15, but the A37-aligned test gate still failed high-route and A08-oracle. `G_4kd07` tested whether a short G_4kd05-final continuation could close high/A08 under a stricter five-metric validation rollback guard:

- initialize from `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05_smoke01/checkpoints/final.pt`;
- keep `gaussian_route_control` as the only trainable prefix;
- add full validation gates for high-route, A08-oracle, low-route, A15-oracle, and low-A15;
- keep the run short: `max_steps=12`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`;
- use physical GPU0/3 only, single visible GPU, no DDP.

## Smoke01 Artifacts

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd07/configs/g_4kd07_smoke01_four_gate_high_a08_micro_lift_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd07/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd07/evaluate_smoke01_gpu.sh`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd07/logs/smoke01_20260621_222007.log`
- Guard summary: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd07_smoke01/metrics/g_4kd07_smoke01_guard_summary.json`

## Outcome

Training exited with code `0`. The no-harm guard ran `6` checks and rolled back all `6` candidate updates. Because no update was accepted, best/final/latest carry no new weight signal and A37-aligned GPU evaluation was intentionally skipped.

The best observed low/A15 side at step `12` passed low-route, A15-oracle, and low-A15 margins, but still missed high-route by `-0.00020684136284722203` and A08-oracle by `-0.0000743865966796875`. This showed that G_4kd05-final continuation could not clear high/A08 without a more targeted mechanism.

## Promotion Boundary

`G_4kd07_smoke01` is not promotable. Formal long training remains blocked because no accepted update exists and aggregate PSNR alone cannot promote a G-RSCN run.
