# Result Analysis: V1/T01/A_f03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_f03`
- Dataset: `UHD_deblur`
- Training split: `1864` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `47.6469` minutes
- Final checkpoint: `experiments/V1/runs/A_f03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_f03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_f03/logs/train_20260511_022049.log`
- Train metrics: `experiments/V1/runs/A_f03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_f03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=2500`, not by the
`wall_clock_minutes=58` cap. No NaN, data-loader failure, checkpoint failure,
or post-validation OOM occurred during training.

Important continuity note:

- `A_f03` is the extended-budget follow-up to the accepted `A_e03` baseline.
- It keeps the same edge-guided detail refiner, Laplacian consistency loss,
  `batch_size=20`, full-image internal validation, and cache-safe launch
  discipline.
- The only main experimental variable is the budget extension
  `max_steps=1500 -> 2500`.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.12446232885122299
```

The final logged training loss was:

```text
step 2500 loss_total = 0.04098471254110336
```

Segment means:

| Step range | Mean train loss | Median | Min | Max |
|---|---:|---:|---:|---:|
| 1-250 | 0.041681 | 0.036202 | 0.026213 | 0.124462 |
| 251-500 | 0.036113 | 0.036729 | 0.025049 | 0.048112 |
| 501-1000 | 0.035029 | 0.034987 | 0.022907 | 0.050559 |
| 1001-1500 | 0.036540 | 0.036748 | 0.026692 | 0.047612 |
| 1501-2000 | 0.033994 | 0.034615 | 0.025256 | 0.043575 |
| 2001-2500 | 0.033960 | 0.033841 | 0.023635 | 0.042950 |

Interpretation:

- The run remained numerically stable across the full extended budget.
- The added 1000 steps did not trigger the post-validation cache failure that
  stopped `A_d03`, confirming that the cache-safe runtime discipline scales to
  the longer short-tier run.
- Training loss alone does not show a dramatic late-stage collapse or runaway
  overfit signal.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.022802 | 28.5946 |
| 1000 | 0.023503 | 28.4861 |
| 1500 | 0.023544 | 28.5727 |
| 2000 | 0.025630 | 28.1510 |
| 2500 | 0.023636 | 28.4802 |

The best internal-validation point is still the earliest evaluated checkpoint:

```text
step 500
val_loss = 0.02280208421871066
val_psnr = 28.594590010634533
```

Interpretation:

- Extending the run to `2500` steps did not improve the internal full-image
  validation PSNR beyond the first peak at `step_000500.pt`.
- The late-training full-image validation numbers remain competitive, but they
  suggest that internal-val PSNR alone would have selected an earlier checkpoint
  than the final model.

## Post-Hoc Full Evaluation

Post-hoc evaluation was run on:

- full internal validation set: `20` samples, `crop_size=0`
- official test set: `300` samples, `crop_size=0`

Evaluated checkpoints:

- `experiments/V1/runs/A_f03/checkpoints/step_000500.pt`
- `experiments/V1/runs/A_f03/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `step_000500.pt` | full internal val | 20 | 0.022802 | 28.5946 |
| `step_000500.pt` | official test | 300 | 0.035395 | 25.0888 |
| `final.pt` | full internal val | 20 | 0.023636 | 28.4802 |
| `final.pt` | official test | 300 | 0.034558 | 25.3145 |

Technical note:

- `step_002500.pt` and `final.pt` have identical `model` weights and therefore
  represent the same final model.
- This means the official result of `final.pt` is also the effective result of
  the final periodic checkpoint at step `2500`.

## Comparison To Existing UHD Deblur Baselines

Accepted baselines before this run:

```text
A03 final.pt official full-image test PSNR = 25.1402293880564
A_e03 final.pt official full-image test PSNR = 25.21410079036176
A_a03 final.pt official full-image test PSNR = 25.15848471495631
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_e03 Final |
|---|---|---:|---:|
| `A03` | `final.pt` | 25.1402293880564 | -0.0739 |
| `A_a03` | `final.pt` | 25.15848471495631 | -0.0556 |
| `A_e03` | `final.pt` | 25.21410079036176 | 0.0000 |
| `A_f03` | `step_000500.pt` | 25.088809336119052 | -0.1253 |
| `A_f03` | `final.pt` | 25.314470701891672 | +0.1004 |

Additional deltas:

```text
A_f03 final.pt vs A03 final.pt  = +0.17424131383527097 dB
A_f03 final.pt vs A_a03 final.pt = +0.15598598693536303 dB
A_f03 final.pt vs A_e03 final.pt = +0.10036991152991348 dB
A_f03 final.pt vs A_f03 step_000500.pt = +0.22566136577261986 dB
```

Interpretation:

- `A_f03` resolves the apparent tension between early internal-validation
  preference and official-test performance in an especially clear way.
- The checkpoint that wins internal full-image validation (`step_000500.pt`) is
  actually worse than the current official baseline on the official test split.
- The final model is substantially stronger than `step_000500.pt` on the
  official full-image test, even though it is slightly weaker on internal
  validation PSNR.
- This means the extra budget did produce useful same-protocol generalization,
  but that gain is visible on the official test rather than on the small
  `20`-image internal full-image validation slice.

## Main Conclusion

`V1/T01/A_f03` is accepted as a successful completed high-VRAM
`UHD_deblur` experiment and as the new official `UHD_deblur` baseline under the
current local protocol.

It proves:

- the `A_e03` edge-guided cache-safe branch was not yet exhausted at
  `max_steps=1500`;
- extending the accepted branch to `2500` steps remains stable under the
  `>=80G` deployment constraint and does not reintroduce the `A_d03`
  post-validation OOM failure mode;
- the extended-budget follow-up improves the official full-image test PSNR from
  `25.2141 dB` (`A_e03 final.pt`) to `25.3145 dB` (`A_f03 final.pt`).

It does not yet prove:

- that the project has reached a staged clearly meaningful jump, since the gain
  over `A_e03` is about `+0.1004 dB`, still below the current `+0.3 dB`
  meaningful-improvement guideline.

The conservative project record is therefore:

> `A_f03` completed cleanly at `step=2500`, validated that the accepted
> edge-guided cache-safe branch still benefits from longer short-tier training,
> and improved the official `UHD_deblur` full-image test PSNR from
> `25.2141 dB` (`A_e03 final.pt`) to `25.3145 dB` (`A_f03 final.pt`).

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
high-VRAM gate acceptance: passed
official full-image test acceptance: passed
official baseline replacement: yes
new official UHD_deblur baseline: V1/T01/A_f03 final.pt, PSNR=25.3145 dB
```

## Recommended Next Action

The highest-value next step on the current run line is now:

1. treat `A_f03` as the new comparison anchor for future `UHD_deblur` work;
2. do not use official test to retro-select checkpoints, and explicitly record
   the mismatch between small-slice internal-val PSNR and full official-test
   preference;
3. move the next `UHD_deblur` branch toward training-inference consistency or
   stronger directional/frequency structure, rather than only repeating a longer
   run on the same branch;
4. outside the deblur line, close the still-pending `A_a02` formal analysis.
