# Result Analysis: V1/T01/A_g03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_g03`
- Dataset: `UHD_deblur`
- Training split: `1864` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `43.6283` minutes
- Final checkpoint: `experiments/V1/runs/A_g03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_g03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_g03/logs/train_20260513_132256.log`
- Train metrics: `experiments/V1/runs/A_g03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_g03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=2500`, not by the `wall_clock_minutes=58`
cap. No NaN, data-loader failure, checkpoint failure, or post-validation OOM
occurred during training.

Important continuity note:

- `A_g03` is the directional/frequency follow-up to the accepted `A_f03`
  baseline.
- It keeps the same edge-guided detail refiner, Laplacian consistency loss,
  full-image internal validation protocol, official-test reservation, and
  cache-safe launch discipline.
- The main mechanism change is `use_directional_frequency_guidance = true`.
- Because the new branch has a slightly larger activation footprint, deployment
  calibration reduced `batch_size` from the inherited design value `20` to
  `18` while still preserving the user's `>=80G` VRAM floor.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.2359021157026291
```

The final logged training loss was:

```text
step 2500 loss_total = 0.03360201418399811
```

Segment means:

| Step range | Mean train loss | Median | Min | Max |
|---|---:|---:|---:|---:|
| 1-250 | 0.047529 | 0.038396 | 0.030417 | 0.235902 |
| 251-500 | 0.036688 | 0.035696 | 0.029740 | 0.046581 |
| 501-1000 | 0.036697 | 0.035282 | 0.026706 | 0.049245 |
| 1001-1500 | 0.034470 | 0.034164 | 0.024104 | 0.047151 |
| 1501-2000 | 0.034805 | 0.034953 | 0.022808 | 0.045845 |
| 2001-2500 | 0.033912 | 0.033646 | 0.024833 | 0.048197 |

Interpretation:

- The run started with a higher loss than `A_f03`, but it stabilized quickly
  and remained in the `0.03x` regime for most of training.
- The added directional/frequency prior did not destabilize optimization.
- The branch stayed comfortably inside the short-tier time budget while
  preserving the high-VRAM deployment requirement.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.022499 | 28.6726 |
| 1000 | 0.022629 | 28.7366 |
| 1500 | 0.022095 | 28.7401 |
| 2000 | 0.024092 | 28.0907 |
| 2500 | 0.022992 | 28.6210 |

The best internal-validation point is:

```text
step 1500
val_loss = 0.02209489792585373
val_psnr = 28.74008243547481
```

Interpretation:

- The small internal full-image validation slice again prefers a mid-training
  checkpoint rather than the final model.
- This means `A_g03` does not remove the checkpoint-selection mismatch already
  exposed by `A_f03`.

## Post-Hoc Full Evaluation

Post-hoc evaluation was run on:

- full internal validation set: `20` samples, `crop_size=0`
- official test set: `300` samples, `crop_size=0`

Evaluated checkpoints:

- `experiments/V1/runs/A_g03/checkpoints/step_001500.pt`
- `experiments/V1/runs/A_g03/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `step_001500.pt` | full internal val | 20 | 0.022095 | 28.7401 |
| `step_001500.pt` | official test | 300 | 0.035130 | 25.2040 |
| `final.pt` | full internal val | 20 | 0.022992 | 28.6210 |
| `final.pt` | official test | 300 | 0.034170 | 25.3801 |

Technical note:

- `step_002500.pt` and `final.pt` have identical `model` weights and therefore
  represent the same final model.
- This means the official result of `final.pt` is also the effective result of
  the true step-`2500` endpoint.

## Comparison To Existing UHD Deblur Baselines

Accepted baseline before this run:

```text
A_f03 final.pt official full-image test PSNR = 25.314470701891672
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_f03 Final |
|---|---|---:|---:|
| `A03` | `final.pt` | 25.1402293880564 | -0.1742 |
| `A_a03` | `final.pt` | 25.15848471495631 | -0.1560 |
| `A_e03` | `final.pt` | 25.21410079036176 | -0.1004 |
| `A_f03` | `final.pt` | 25.314470701891672 | 0.0000 |
| `A_g03` | `step_001500.pt` | 25.204020174253888 | -0.1105 |
| `A_g03` | `final.pt` | 25.3800849048404 | +0.0656 |

Additional deltas:

```text
A_g03 final.pt vs A03 final.pt   = +0.2398555167839973 dB
A_g03 final.pt vs A_e03 final.pt = +0.1659841144786398 dB
A_g03 final.pt vs A_f03 final.pt = +0.06561420294872633 dB
A_g03 final.pt vs A_g03 step_001500.pt = +0.17606473058651062 dB
```

Interpretation:

- `A_g03` gives a real same-protocol official improvement over `A_f03`, but
  the gain is modest.
- The checkpoint that wins internal full-image validation
  (`step_001500.pt`) is weaker than both `A_f03 final.pt` and `A_g03 final.pt`
  on the official full-image test.
- This means the directional/frequency mechanism helps, but its benefit is
  visible only at the final trained model rather than at the internal-val
  winner.

## Main Conclusion

`V1/T01/A_g03` is accepted as a successful completed high-VRAM
`UHD_deblur` experiment, as a positive directional/frequency follow-up to the
accepted `A_f03` branch, and as the new official `UHD_deblur` baseline under
the current local protocol.

It proves:

- structured directional/frequency priors distilled from predicted Gaussian
  parameters can improve the accepted edge-guided deblur branch without
  abandoning the main `lowres -> gaussian -> highres` story;
- the new mechanism remains stable under the `>=80G` deployment constraint
  after recalibration to `batch_size=18`;
- the new branch improves the official full-image test PSNR from
  `25.3145 dB` (`A_f03 final.pt`) to `25.3801 dB` (`A_g03 final.pt`).

It does not yet prove:

- a clearly meaningful jump, since the gain over `A_f03` is only about
  `+0.0656 dB`, and the total gain over the original `A03 final.pt` baseline
  is still below `+0.3 dB`.

The conservative project record is therefore:

> `A_g03` completed cleanly at `step=2500`, validated that Gaussian-derived
> directional/frequency guidance can improve the official `UHD_deblur`
> full-image test result, and raised the current local baseline from
> `25.3145 dB` (`A_f03 final.pt`) to `25.3801 dB` (`A_g03 final.pt`), while
> leaving the larger breakthrough still open.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
high-VRAM gate acceptance: passed
official full-image test acceptance: passed
official baseline replacement: yes
new official UHD_deblur baseline: V1/T01/A_g03 final.pt, PSNR=25.3801 dB
```

## Recommended Next Action

The highest-value next step on the current run line is now:

1. treat `A_g03` as the new comparison anchor for future `UHD_deblur` work;
2. explicitly preserve the lesson that the small internal full-image validation
   slice still does not reliably rank the official-test winner;
3. move the next `UHD_deblur` branch toward train-inference consistency or a
   stronger upstream low-resolution representation change, rather than replaying
   another near-identical directional/frequency branch;
4. outside the deblur line, continue filling out the formal metric package
   beyond PSNR, especially `SSIM` and `LPIPS` on accepted branches.
