# Result Analysis: V1/T01/A_i03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_i03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `12000`
- Final logged elapsed time: `156.1359` minutes
- Final checkpoint: `experiments/V1/runs/A_i03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_i03/checkpoints/step_012000.pt`
- Training log: `experiments/V1/runs/A_i03/logs/train_20260514_022908.log`
- Train metrics: `experiments/V1/runs/A_i03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_i03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=12000`, not by the `wall_clock_minutes=300`
cap. No NaN, data-loader failure, checkpoint failure, or post-validation OOM
occurred during training.

Technical note:

- `step_012000.pt` and `final.pt` have identical `model` weights.
- This means the accepted official result of `final.pt` is also the effective
  result of the true step-`12000` endpoint.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.12022989988327026
```

The final logged training loss was:

```text
step 12000 loss_total = 0.040931668132543564
```

Interpretation:

- The medium-budget consistency branch trained stably for the full planned run.
- The `TLC`-style larger-context supervision path did not introduce optimizer
  instability, even under the stricter `85-93G` VRAM window.
- Relative to the short-run deblur branches, this line had enough budget to
  let the consistency repair actually express itself on the official protocol.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 1000 | 0.022877 | 28.6524 |
| 2000 | 0.025394 | 27.8974 |
| 3000 | 0.022453 | 28.7704 |
| 4000 | 0.021658 | 29.0071 |
| 5000 | 0.022848 | 28.7871 |
| 6000 | 0.021294 | 29.0882 |
| 7000 | 0.021185 | 29.1361 |
| 8000 | 0.021217 | 29.1314 |
| 9000 | 0.020965 | 29.1963 |
| 10000 | 0.021233 | 29.1676 |
| 11000 | 0.020352 | 29.2631 |
| 12000 | 0.020453 | 29.2847 |

The best training-time validation PSNR is:

```text
step 12000
val_loss = 0.02045327741652727
val_psnr = 29.284706775950752
```

The lowest training-time validation loss is:

```text
step 11000
val_loss = 0.020351742208004
val_psnr = 29.263130525757354
```

Interpretation:

- Unlike several earlier deblur branches, this medium-budget consistency run
  does not show a strong conflict between internal full-image validation and
  the final trained model.
- The best-val-loss checkpoint is `step_011000.pt`, but best-val-PSNR is the
  final step, and the two points are very close.

## Post-Hoc Full Evaluation

Post-hoc evaluation was run on:

- full internal validation set: `20` samples, `crop_size=0`
- official test set: `300` samples, `crop_size=0`

Evaluated checkpoints:

- `experiments/V1/runs/A_i03/checkpoints/step_011000.pt`
- `experiments/V1/runs/A_i03/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `step_011000.pt` | full internal val | 20 | 0.020352 | 29.2631 |
| `step_011000.pt` | official test | 300 | 0.032694 | 25.7934 |
| `final.pt` | full internal val | 20 | 0.020453 | 29.2847 |
| `final.pt` | official test | 300 | 0.032553 | 25.8506 |

Interpretation:

- The medium-budget consistency branch clearly improves the official full-image
  result, and the final trained model is the best official-test checkpoint.
- This branch therefore preserves an important project-wide lesson:
  official-test preference still leans toward the fully trained endpoint.
- But unlike `A_f03` / `A_g03`, the internal full-image validation story is
  much more aligned with the official result: `step_011000.pt` is already
  strong on the official test, and `final.pt` improves it further.

## Comparison To Existing UHD Deblur Baselines

Frozen baseline before this run:

```text
A_g03 final.pt official full-image test PSNR = 25.3800849048404
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_g03 Final |
|---|---|---:|---:|
| `A03` | `final.pt` | 25.1402293880564 | -0.2399 |
| `A_e03` | `final.pt` | 25.21410079036176 | -0.1660 |
| `A_f03` | `final.pt` | 25.314470701891672 | -0.0656 |
| `A_g03` | `final.pt` | 25.3800849048404 | 0.0000 |
| `A_i03` | `step_011000.pt` | 25.793384050297114 | +0.4133 |
| `A_i03` | `final.pt` | 25.85061787687282 | +0.4705 |

Additional deltas:

```text
A_i03 final.pt vs A03 final.pt   = +0.7103884888164192 dB
A_i03 final.pt vs A_e03 final.pt = +0.6365170865110581 dB
A_i03 final.pt vs A_f03 final.pt = +0.5361471749811475 dB
A_i03 final.pt vs A_g03 final.pt = +0.47053297203242295 dB
A_i03 final.pt vs A_i03 step_011000.pt = +0.057233826575707525 dB
```

Interpretation:

- This is the first `UHD_deblur` branch in the current line that moves the
  official baseline by nearly `+0.5 dB` in a single step.
- The consistency-repair idea is therefore no longer a speculative direction;
  under a medium budget and a slightly enlarged train split, it becomes the
  strongest validated improvement on the deblur line so far.

## Main Conclusion

`V1/T01/A_i03` is accepted as a successful completed medium-budget
`UHD_deblur` experiment, as the first deployed and validated `TLC`-style
consistency-repair branch after the frozen `A_g03` milestone, and as the new
official `UHD_deblur` baseline under the current local protocol.

It proves:

- exposing larger training context while supervising only the centered region
  can materially reduce the patch-train / full-image-test mismatch on this
  task;
- the `A_g03` directional/frequency mainline and the consistency-repair idea
  are complementary rather than conflicting;
- under the same official full-image protocol, the current local baseline moves
  from `25.3801 dB` (`A_g03 final.pt`) to `25.8506 dB` (`A_i03 final.pt`);
- the medium-budget consistency branch finally reaches the project’s
  long-sought “clearly meaningful” deblur improvement scale.

It does not yet prove:

- that this line has saturated, because the official-test gain is still rising
  slightly from `step_011000.pt` to the final endpoint;
- that the same consistency strategy will transfer equally well to the haze
  task without task-specific adaptation.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
85-93G real-launch window acceptance: passed
official full-image test acceptance: passed
official baseline replacement: yes
new official UHD_deblur baseline: V1/T01/A_i03 final.pt, PSNR=25.8506 dB
```

## Recommended Next Action

The highest-value next step on the current run line is now:

1. treat `A_i03 final.pt` as the new deblur comparison anchor and freeze it as
   the next milestone package;
2. preserve the lesson that the final endpoint still wins the official test,
   but that the consistency branch has greatly narrowed the internal-val /
   official-test mismatch;
3. design the next branch around a careful extension of this consistency line,
   rather than reverting to pre-consistency architecture probes.
