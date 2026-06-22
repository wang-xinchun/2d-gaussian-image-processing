# G_4kd04 Result Analysis

- Created: `2026-06-21 20:45:00 CST`
- Updated: `2026-06-21 21:22:35 CST`
- Status: `evaluated_not_promoted`

## Smoke01 Question

`G_4kd04_smoke01` tested whether low/A15 no-harm can be enforced during optimizer updates instead of only during best-checkpoint selection.

## Deployment

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04/configs/g_4kd04_smoke01_step_no_harm_rollback_short.toml`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04/logs/smoke01_20260621_205925.log`
- Best eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04/logs/eval_smoke01_best_val_20260621_210940.log`
- Final eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04/logs/eval_smoke01_final_20260621_211340.log`
- Summary JSON: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd04_smoke01/metrics/g_4kd04_smoke01_a37_aligned_summary.json`

All train/eval jobs used physical GPU3 only, `CUDA_VISIBLE_DEVICES=3`, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`, and no CPU fallback. Train/eval exit codes were all `0`. Peak train CUDA reserved memory was `3.11328125GiB`.

## Step No-Harm Guard

- Guard interval: every `2` optimizer steps.
- Guard validation size: `8` batches.
- Guard metrics: low-route, A15-oracle, and low-A15 router gaps vs initialized baseline.
- Checks: `20`.
- Accepted updates: `4` at steps `4, 6, 8, 10`.
- Rolled back updates: `16`.

The guard is active and not silently bypassed. It prevented most later low-route-harmful updates from accumulating into the final checkpoint.

## A37-Aligned Evaluation

| checkpoint | step | test PSNR | high-route delta vs A37 | low-route delta vs A37 | A08-oracle delta vs A37 | A15-oracle delta vs A37 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_val` | 0 | `17.888740401218694` | `+0.0002355462360844468` | `-0.0017883107046233748` | `+0.0018106951341312072` | `-0.0032425407070728884` |
| `final` | 40 | `17.88861635982521` | `+0.00017073030127777444` | `-0.0019570309529002294` | `+0.0016357684574719311` | `-0.003323235377853284` |

Against `G_4kd03` final, `G_4kd04` final recovered low/A15 substantially:

- test PSNR: `+0.0017128589792072546`;
- low-route gap: `+0.003223806727921802`;
- A15-oracle gap: `+0.005437738767618661`.

It gave back some of the high/A08 movement:

- high-route gap: `-0.0002900252458273439`;
- A08-oracle gap: `-0.002659825989792197`.

## Verdict

`G_4kd04_smoke01` is a useful update-stabilization mechanism, but it is not a breakthrough. The rollback guard keeps the final checkpoint close to the protected parent baseline instead of allowing the larger G_4kd03 final low/A15 slide, but both `best_val` and `final` still fail the A37 low-route and A15-oracle promotion gates.

Do not extend `G_4kd04_smoke01` into formal training. The next probe should start from an A37-equivalent baseline and apply the no-harm guard against that baseline, rather than protecting a parent checkpoint that already trails A37 on low/A15.
