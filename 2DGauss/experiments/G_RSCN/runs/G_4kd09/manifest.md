# G_4kd09 Manifest

- Created: `2026-06-22 00:07:34 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd09`
- First smoke run: `G_4kd09_smoke01`
- Status: `completed; positive-margin guard rejected all updates; formal training blocked`

## Difference From G_4kd08

`G_4kd08_smoke01` passed validation gates but failed A37-aligned test high/A08 gates. `G_4kd09` restarted from A37 and added positive validation margins to test whether validation-side surplus could transfer to test:

- initialize from `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`;
- keep `gaussian_route_control` as the only trainable prefix;
- keep full 24-batch step no-harm guard on high-route, A08-oracle, low-route, A15-oracle, and low-A15;
- raise validation thresholds to positive margins over the G08 non-regression baseline;
- increase high/A08 pressure to A08 anchor `0.030` and high-branch anchor `0.075`;
- keep low/A15 protection at A15 anchor `0.050` and low-branch anchor `0.105`;
- keep the run short: `max_steps=16`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.

## Smoke01 Artifacts

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd09/configs/g_4kd09_smoke01_a37_margin_four_gate_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd09/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd09/evaluate_smoke01_gpu.sh`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd09/logs/smoke01_20260622_000734.log`
- Guard summary: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd09_smoke01/metrics/g_4kd09_smoke01_guard_summary.json`

## Outcome

Training exited with code `0`. The no-harm guard ran `8` checks and rolled back all `8` updates. The closest useful candidate was step `4`: high-route, low-route, A15-oracle, and low-A15 were margin-positive, but A08-oracle missed the G09 A08 margin by `-0.00002738494873047448`.

Because no update was accepted, best/final/latest carry no new weight signal and A37-aligned GPU evaluation was intentionally skipped.

## Promotion Boundary

`G_4kd09_smoke01` is not promotable. It narrowed the bottleneck to the A08 validation margin under strict positive margins, but formal long training remains blocked.
