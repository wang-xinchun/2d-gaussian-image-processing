# G_4kd08 Manifest

- Created: `2026-06-21 22:32:14 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd08`
- First smoke run: `G_4kd08_smoke01`
- Status: `completed; validation gate passed but test gate failed; formal training blocked`

## Difference From G_4kd07

`G_4kd07_smoke01` showed that continuing from G_4kd05 final could not pass high/A08 under the five-metric guard. `G_4kd08` restarted from A37 and tested whether symmetric A37-validation gates could accept a candidate that transfers to the A37-aligned test gate:

- initialize from `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`;
- keep `gaussian_route_control` as the only trainable prefix;
- use five validation gates for high-route, A08-oracle, low-route, A15-oracle, and low-A15;
- use stronger low/A15 protection than G07: A15 anchor `0.055`, low-branch anchor `0.110`;
- keep the run short: `max_steps=12`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.

## Smoke01 Artifacts

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd08/configs/g_4kd08_smoke01_a37_four_gate_symmetric_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd08/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd08/evaluate_smoke01_gpu.sh`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd08/logs/smoke01_20260621_223214.log`
- Best eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd08/logs/eval_smoke01_best_val_20260621_223709.log`
- Final eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd08/logs/eval_smoke01_final_20260621_224103.log`
- Summary: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd08_smoke01/metrics/g_4kd08_smoke01_a37_aligned_summary.json`

## Outcome

Training exited with code `0`. The no-harm guard ran `6` checks, accepted steps `10` and `12`, and rolled back steps `2, 4, 6, 8`. Validation-side five-gate behavior was encouraging, but A37-aligned test evaluation failed:

- best checkpoint PSNR delta vs A37: `-0.000002008497823169364`;
- best high-route delta: `-0.000015728099650044736`;
- best A08-oracle delta: `-0.00009900925367264946`;
- best low-route delta: `+0.000008341377242176984`;
- best A15-oracle delta: `+0.00008062177568141093`.

## Promotion Boundary

`G_4kd08_smoke01` is not promotable. Validation gate pass is insufficient; test-side high-route and A08-oracle must clear A37 before formal long training is allowed.
