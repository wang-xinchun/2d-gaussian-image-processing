# Result Analysis: V1/T01/A_k03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_k03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `23016`
- Final logged elapsed time: `299.9313` minutes
- Final checkpoint: `experiments/V1/runs/A_k03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_k03/checkpoints/step_023000.pt`
- Training log: `experiments/V1/runs/A_k03/logs/train_20260514_104102.log`
- Train metrics: `experiments/V1/runs/A_k03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_k03/metrics/val_metrics.jsonl`

The run completed by the `wall_clock_minutes=300` cap, not by reaching the
configured `max_steps=24000`. No NaN, data-loader failure, checkpoint failure,
or post-validation OOM occurred during training.

Technical note:

- `step_023000.pt` and `latest.pt` have identical `model` weights.
- `final.pt` is **not** identical to `step_023000.pt` / `latest.pt`.
- This means the accepted official result of `final.pt` comes from the true
  post-step-`23000` endpoint rather than from the last periodic checkpoint.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.12022989988327026
```

The final logged training loss before clean finish was:

```text
step 23010 loss_total = 0.02486291341483593
```

Interpretation:

- The step-cap extension branch remained stable for the full five-hour window.
- Extending the accepted `A_i03` setting did not reintroduce any runtime or
  memory-instability issue.
- The branch therefore answers the intended engineering question positively:
  the `A_i03` line still had room to absorb substantially more optimization.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 1000 | 0.022730 | 28.6590 |
| 2000 | 0.024892 | 28.0299 |
| 3000 | 0.022764 | 28.7470 |
| 4000 | 0.021335 | 29.0432 |
| 5000 | 0.022847 | 28.7280 |
| 6000 | 0.021483 | 29.0172 |
| 7000 | 0.021121 | 29.1544 |
| 8000 | 0.021059 | 29.1719 |
| 9000 | 0.020985 | 29.2204 |
| 10000 | 0.021091 | 29.2011 |
| 11000 | 0.020382 | 29.2845 |
| 12000 | 0.020388 | 29.3024 |
| 13000 | 0.020862 | 29.2321 |
| 14000 | 0.020384 | 29.3773 |
| 15000 | 0.020459 | 29.2851 |
| 16000 | 0.020272 | 29.3262 |
| 17000 | 0.020617 | 29.3501 |
| 18000 | 0.020346 | 29.2710 |
| 19000 | 0.020113 | 29.3206 |
| 20000 | 0.020242 | 29.3531 |
| 21000 | 0.020293 | 29.3498 |
| 22000 | 0.020013 | 29.4266 |
| 23000 | 0.020084 | 29.3495 |

The best training-time validation PSNR is:

```text
step 22000
val_loss = 0.0200129559263587
val_psnr = 29.42655123730821
```

The lowest training-time validation loss is the same point:

```text
step 22000
val_loss = 0.0200129559263587
val_psnr = 29.42655123730821
```

Interpretation:

- Internal full-image validation continues to improve far beyond the original
  `A_i03` endpoint and peaks late in training.
- But this branch still preserves the project-wide lesson that the best
  internal-validation checkpoint is not necessarily the best official-test
  checkpoint.
- Compared with `A_i03`, the same setting under a longer budget produces a
  stronger internal-validation ceiling, which is a useful sign that the line
  had not yet saturated at `12000` steps.

## Post-Hoc Full Evaluation

Post-hoc evaluation was run on:

- full internal validation set: `20` samples, `crop_size=0`
- official test set: `300` samples, `crop_size=0`

Evaluated checkpoints:

- `experiments/V1/runs/A_k03/checkpoints/step_022000.pt`
- `experiments/V1/runs/A_k03/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `step_022000.pt` | full internal val | 20 | 0.020013 | 29.4266 |
| `step_022000.pt` | official test | 300 | 0.031386 | 26.0810 |
| `final.pt` | full internal val | 20 | 0.020395 | 29.2665 |
| `final.pt` | official test | 300 | 0.031135 | 26.2000 |

Interpretation:

- The best internal-validation checkpoint is `step_022000.pt`, but the best
  official-test checkpoint is still the final trained endpoint.
- The official-test gain from `step_022000.pt` to `final.pt` is about
  `+0.1190 dB`, which is much larger than a rounding-level fluctuation.
- This branch therefore sharpens, rather than weakens, the existing project
  rule: official test must remain strictly post-hoc and cannot be replaced by
  the small internal validation slice for branch selection.

## Comparison To Existing UHD Deblur Baselines

Frozen baseline before this run:

```text
A_i03 final.pt official full-image test PSNR = 25.85061787687282
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_i03 Final |
|---|---|---:|---:|
| `A_g03` | `final.pt` | 25.3800849048404 | -0.4705 |
| `A_i03` | `final.pt` | 25.85061787687282 | 0.0000 |
| `A_k03` | `step_022000.pt` | 26.08100467830892 | +0.2304 |
| `A_k03` | `final.pt` | 26.199959289578075 | +0.3493 |

Additional deltas:

```text
A_k03 final.pt vs A_g03 final.pt       = +0.8198743847376759 dB
A_k03 final.pt vs A_i03 final.pt       = +0.3493414127052539 dB
A_k03 final.pt vs A_k03 step_022000.pt = +0.11895461126915663 dB
```

Interpretation:

- The branch gives a clear positive answer to the narrow experimental
  question: simply removing the earlier step cap while preserving the accepted
  `A_i03` setting does further improve same-protocol official performance.
- This is not a marginal bookkeeping update. Relative to the frozen `A_i03`
  milestone, the gain is about `+0.3493 dB`, which is large enough to justify
  baseline replacement and milestone refresh.
- At the same time, the gain is still much smaller than the multi-dB gap to
  strong public references, so this should be read as a strong local
  continuation result rather than as evidence that `T01` has already closed
  the broader architecture gap.

## Relation To The Lightweight 90% Goal

Current working `UHD-Blur` 90% PSNR threshold:

```text
26.5743 dB
```

Comparison:

```text
A_i03 final.pt = 25.85061787687282 dB
A_k03 final.pt = 26.199959289578075 dB
remaining gap to 90% threshold = 0.37434071042192585 dB
```

Interpretation:

- `A_k03` substantially narrows the distance to the lightweight project goal.
- However, it still does **not** cross the current `UHD-Blur` 90% gate.
- The branch therefore becomes the new best lightweight local baseline on the
  deblur line, but it also clarifies that future improvement should move
  beyond “same template, more steps only” and toward a stronger yet still
  lightweight template upgrade.

## Main Conclusion

`V1/T01/A_k03` is accepted as a successful completed medium-budget
`UHD_deblur` experiment, as a positive step-cap-extension follow-up to the
frozen `A_i03/M02` baseline, and as the new official `UHD_deblur` baseline
under the current local protocol.

It proves:

- the accepted `A_i03` setting had not saturated at `12000` steps;
- letting the same branch consume nearly the full five-hour budget improves
  official full-image test performance materially;
- the official-test winner is again the final trained endpoint rather than the
  best internal-validation checkpoint;
- the current local `UHD_deblur` baseline now moves from `25.8506 dB`
  (`A_i03 final.pt`) to `26.2000 dB` (`A_k03 final.pt`).

It does not yet prove:

- that the line has crossed the lightweight `90%` target for `UHD-Blur`;
- that additional near-isomorphic long training alone is the best next move;
- that the broader template-level gap to strong public references has been
  removed.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
300-minute clean finish acceptance: passed
official full-image test acceptance: passed
official baseline replacement: yes
new official UHD_deblur baseline: V1/T01/A_k03 final.pt, PSNR=26.2000 dB
lightweight 90% gate for UHD-Blur: not yet passed
```

## Recommended Next Action

The highest-value next step on the current run line is now:

1. freeze `A_k03 final.pt` as the next milestone package and repoint the
   stable official-baseline link to it;
2. preserve the lesson that final endpoints still beat internal-val winners on
   the official test, even on this stronger long-budget branch;
3. treat `A_k03` as the new deblur comparison anchor while preparing the next
   lightweight template upgrade rather than only replaying another near-
   identical longer-budget run.
