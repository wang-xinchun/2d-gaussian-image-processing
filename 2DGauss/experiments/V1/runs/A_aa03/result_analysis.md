# Result Analysis: V1/T01/A_aa03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_aa03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `21256`
- Final logged elapsed time: `299.9259` minutes
- Final checkpoint: `experiments/V1/runs/A_aa03/checkpoints/final.pt`
- Latest periodic checkpoint:
  `experiments/V1/runs/A_aa03/checkpoints/step_021000.pt`
- Training log:
  `experiments/V1/runs/A_aa03/logs/train_20260516_161430.log`
- Train metrics:
  `experiments/V1/runs/A_aa03/metrics/train_metrics.jsonl`
- Validation metrics:
  `experiments/V1/runs/A_aa03/metrics/val_metrics.jsonl`
- Official test metrics:
  `experiments/V1/runs/A_aa03/metrics/official_test_final.json`

The run completed by the configured `300`-minute cap, not by reaching
`max_steps=24000`. No NaN, data-loader failure, checkpoint failure, or
post-validation OOM occurred during training.

Technical note:

- `step_021000.pt` and `latest.pt` have identical `model` weights.
- `final.pt` is **not** identical to `step_021000.pt` / `latest.pt`.
- This means the accepted official result must be attributed to the true
  post-step-`21256` endpoint rather than to the last periodic checkpoint.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.18244826793670654
```

The final logged training loss before clean finish was:

```text
step 21250 loss_total = 0.027953719720244408
```

Interpretation:

- The promoted detail-hint line remains runtime-stable for the full
  five-hour budget.
- No new engineering instability appears despite the near-94G real startup
  envelope and the added refinement hint.
- So the question answered by this run is quality-side, not runtime-side.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 1000 | 0.024787 | 28.5406 |
| 2000 | 0.024570 | 28.6234 |
| 3000 | 0.023065 | 28.9705 |
| 4000 | 0.023058 | 28.9045 |
| 5000 | 0.024395 | 28.4705 |
| 6000 | 0.022393 | 29.0950 |
| 7000 | 0.021923 | 29.2144 |
| 8000 | 0.022326 | 29.1428 |
| 9000 | 0.021939 | 29.2448 |
| 10000 | 0.022951 | 28.9813 |
| 11000 | 0.021436 | 29.3148 |
| 12000 | 0.021701 | 29.2957 |
| 13000 | 0.021426 | 29.3134 |
| 14000 | 0.021408 | 29.2519 |
| 15000 | 0.021332 | 29.3715 |
| 16000 | 0.021552 | 29.3040 |
| 17000 | 0.021071 | 29.3760 |
| 18000 | 0.021745 | 29.1201 |
| 19000 | 0.020909 | 29.4790 |
| 20000 | 0.021335 | 29.2859 |
| 21000 | 0.020933 | 29.4789 |

The best training-time validation PSNR is:

```text
step 19000
val_loss = 0.02090863501653075
val_psnr = 29.47901527270407
```

The lowest training-time validation loss is the same point:

```text
step 19000
val_loss = 0.02090863501653075
val_psnr = 29.47901527270407
```

Interpretation:

- Internal validation is strong and clearly above the short-run regime.
- But even the best internal-validation point does not imply the branch beats
  the accepted `M04/A_y03` line on official test.
- The branch therefore confirms that the hint is trainable at medium scale,
  but not that it is the right medium-scale direction.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_aa03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.033309 | 26.0850 |

Interpretation:

- The promoted detail-hint line remains above the older medium anchor `A_p03`.
- But it fails the actual promotion question against frozen `M04/A_y03`.
- The branch is also below `A_x03`, so it does not even preserve the earlier
  best non-milestone medium result.

## Comparison To Existing UHD Deblur Anchors

Relevant same-task same-protocol references:

```text
A_y03 final.pt official full-image test PSNR = 26.272695829299355
A_x03 final.pt official full-image test PSNR = 26.207660130733675
A_p03 final.pt official full-image test PSNR = 26.008706971730295
A_z03 final.pt official full-image test PSNR = 25.385710820365173
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_aa03 Final |
|---|---|---:|---:|
| `A_y03` | `final.pt` | 26.2727 | +0.1877 |
| `A_x03` | `final.pt` | 26.2077 | +0.1227 |
| `A_aa03` | `final.pt` | 26.0850 | 0.0000 |
| `A_p03` | `final.pt` | 26.0087 | -0.0763 |
| `A_z03` | `final.pt` | 25.3857 | -0.6993 |

Additional deltas:

```text
A_aa03 final.pt vs A_y03 final.pt = -0.18771254321788788 dB
A_aa03 final.pt vs A_x03 final.pt = -0.12267684465220796 dB
A_aa03 final.pt vs A_p03 final.pt = +0.07627631435117266 dB
A_aa03 final.pt vs A_z03 final.pt = +0.6992724657162945 dB
```

Interpretation:

- Promoting the hint to medium scale does preserve a large gain over the
  short-run source `A_z03`, so the branch is not a failed training run.
- But the only comparison that matters for acceptance is against frozen
  `M04/A_y03`, and there the branch is clearly worse.
- The correct lesson is therefore not “detail-hint is noise,” but
  “detail-hint alone is insufficient as a medium-budget replacement
  mechanism.”

## Main Conclusion

`V1/T01/A_aa03` is accepted as a successfully completed medium-budget
promotion probe, but **not** as a successful mainline promotion.

It proves:

- the `A_z03` decoder-detail-aware hint is runtime-safe and medium-budget
  trainable;
- the branch remains better than the older `A_p03` line;
- the hint is not merely a short-run artifact in the sense of collapsing
  below all earlier medium results.

It does not prove:

- that the hint improves on frozen `M04/A_y03`;
- that the hint improves on `A_x03`;
- that the current accepted medium-line mechanism should be replaced.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
300-minute clean finish acceptance: passed
official full-image test acceptance: passed
beat frozen M04/A_y03: no
beat A_x03: no
remain above A_p03: yes
recommended next action: retire this hint as a standalone medium-line promotion mechanism and return to new lightweight structural redesign above M04/A_y03
```
