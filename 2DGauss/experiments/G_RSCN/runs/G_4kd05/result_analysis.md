# G_4kd05 Result Analysis

- Created: `2026-06-21 21:22:35 CST`
- Updated: `2026-06-21 21:50:00 CST`
- Status: `evaluated_not_promoted_near_miss`

## Smoke01 Plan

`G_4kd05_smoke01` tests whether the G_4kd04 rollback mechanism can produce a promotable checkpoint when the protected baseline is A37 itself, not a G-line parent already below A37 on low/A15.

Expected signal:

- step-0 router baseline metrics should correspond to the A37-initialized G-RSCN model;
- `step_no_harm_guard_metrics.jsonl` should show full-validation checks every two optimizer steps;
- low-route, A15-oracle, and low-A15 harmful updates should be rolled back;
- useful high/A08 movement must survive both the guard and the best-selection gate;
- final A37-aligned evaluation must improve high/A08 without losing low/A15 to be promotable.

Blocking rule:

- Do not run formal long training until best/final A37-aligned GPU evaluation is recorded here.

## Deployment

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05/configs/g_4kd05_smoke01_a37_fullval_no_harm_short.toml`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05/logs/smoke01_20260621_213558.log`
- Best eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05/logs/eval_smoke01_best_val_20260621_214028.log`
- Final eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05/logs/eval_smoke01_final_20260621_214317.log`
- Summary JSON: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd05_smoke01/metrics/g_4kd05_smoke01_a37_aligned_summary.json`

All train/eval jobs used physical GPU3 only, `CUDA_VISIBLE_DEVICES=3`, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`, and no CPU fallback. Train/eval exit codes were all `0`. Peak train CUDA reserved memory was `3.095703125GiB`.

## Step No-Harm Guard

- Guard interval: every `2` optimizer steps.
- Guard validation size: `24` batches.
- Checks: `12`.
- Accepted updates: `5`, at steps `16, 18, 20, 22, 24`.
- Rolled back updates: `7`.

The first half of training was rejected for low/A15 regression. The later updates passed full-validation no-harm, so the guard is not merely freezing all training.

## A37-Aligned Evaluation

| checkpoint | step | test PSNR delta vs A37 | high-route delta vs A37 | low-route delta vs A37 | A08-oracle delta vs A37 | A15-oracle delta vs A37 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_val` | 0 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `final` | 24 | `+0.00009609345127614688` | `-0.000021236730810156312` | `+0.0001846056939061591` | `-0.00008147229078794838` | `+0.000247353157482616` |

## Verdict

`G_4kd05_smoke01` is the strongest G-RSCN near-miss so far, but it is not promotable. Final improves aggregate PSNR, low-route, and A15-oracle over A37, while high-route and A08-oracle are microscopically below A37. Formal long training remains blocked because promotion requires all four route/oracle gates.

The next probe should keep the A37 full-validation no-harm guard and add only a small high/A08 lift. The target is not another broad loss change; the remaining gap is only `0.000021236730810156312 dB` on high-route and `0.00008147229078794838 dB` on A08-oracle.
