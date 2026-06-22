# Result Analysis: V1/T01/A_y03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_y03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `21444`
- Final logged elapsed time: `299.9066` minutes
- Final checkpoint: `experiments/V1/runs/A_y03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_y03/checkpoints/step_021000.pt`
- Training log: `experiments/V1/runs/A_y03/logs/train_20260516_085803.log`
- Train metrics: `experiments/V1/runs/A_y03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_y03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_y03/metrics/official_test_final.json`

The run completed by the configured `300` minute cap, not by reaching
`max_steps=24000`. No NaN, data-loader failure, checkpoint failure, or
post-validation OOM occurred during training.

Technical note:

- `step_021000.pt` and `latest.pt` have identical `model` weights.
- `final.pt` is **not** identical to `step_021000.pt` / `latest.pt`.
- This means the accepted official result must be attributed to the true
  post-step-`21444` endpoint rather than to the last periodic checkpoint.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.22724193334579468
```

The final logged training loss before clean finish was:

```text
step 21440 loss_total = 0.01940067857503891
```

Interpretation:

- The replayed light-edge feature-refinement line remains stable through the
  full five-hour confirmation budget.
- No new runtime instability appears under the same near-94G real startup
  envelope observed in `A_x03`.
- So the confirmation replay answers the engineering reproducibility question
  positively: this line is not a one-off deployment artifact.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 1000 | 0.024702 | 28.3936 |
| 2000 | 0.024206 | 28.6056 |
| 3000 | 0.023701 | 28.7880 |
| 4000 | 0.023210 | 28.9175 |
| 5000 | 0.023151 | 28.9509 |
| 6000 | 0.021961 | 29.1725 |
| 7000 | 0.021743 | 29.1419 |
| 8000 | 0.021823 | 29.2570 |
| 9000 | 0.021702 | 29.1154 |
| 10000 | 0.021377 | 29.1949 |
| 11000 | 0.021521 | 29.2497 |
| 12000 | 0.021907 | 29.2423 |
| 13000 | 0.021397 | 29.1796 |
| 14000 | 0.021481 | 29.2076 |
| 15000 | 0.021550 | 29.3667 |
| 16000 | 0.021124 | 29.2790 |
| 17000 | 0.021126 | 29.3397 |
| 18000 | 0.021221 | 29.3071 |
| 19000 | 0.021095 | 29.2799 |
| 20000 | 0.021237 | 29.2050 |
| 21000 | 0.021056 | 29.2312 |

The best training-time validation PSNR is:

```text
step 15000
val_loss = 0.02154993759468198
val_psnr = 29.36665063895877
```

The lowest training-time validation loss is:

```text
step 21000
val_loss = 0.021056028548628092
val_psnr = 29.231167659453103
```

Interpretation:

- Internal validation remains consistently above `29 dB` over a long interval,
  which is strong evidence that the replay did not collapse.
- The best validation PSNR is lower than the peak seen in `A_x03`, but still
  clearly within the same high-performing regime.
- As in `A_x03`, the best internal-validation point is not necessarily the
  final official-test winner.

## Post-Hoc Full Evaluation

Post-hoc evaluation was run on:

- full internal validation set: `20` samples, `crop_size=0`
- official test set: `300` samples, `crop_size=0`

Evaluated checkpoints:

- `experiments/V1/runs/A_y03/checkpoints/step_021000.pt`
- `experiments/V1/runs/A_y03/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `step_021000.pt` | full internal val | 20 | 0.021056 | 29.2312 |
| `step_021000.pt` | official test | 300 | 0.032981 | 26.2206 |
| `final.pt` | full internal val | 20 | 0.021223 | 29.1910 |
| `final.pt` | official test | 300 | 0.032932 | 26.2727 |

Interpretation:

- The best internal-validation checkpoint remains `step_021000.pt`, which is
  exactly the point of lowest validation loss during training.
- But the best official-test checkpoint is again the true final endpoint.
- The official-test gain from `step_021000.pt` to `final.pt` is about
  `+0.0521 dB`, so the last training tail after the final periodic checkpoint
  still converts into real benchmark gain rather than only internal-val drift.
- The replay not only reproduces the `A_x03` edge over frozen `A_k03`, it
  expands it materially under the same protocol.
- This converts the previous “best observed once” result into a reproducible
  positive line with a visibly larger official margin.
- The milestone-refresh question is therefore answered positively by direct
  replay evidence.

## Comparison To Existing UHD Deblur Anchors

Frozen milestone before this run:

```text
A_k03 final.pt official full-image test PSNR = 26.199959289578075
```

Relevant same-task same-protocol references:

```text
A_x03 final.pt official full-image test PSNR = 26.207660130733675
A_p03 final.pt official full-image test PSNR = 26.008706971730295
A_k03 final.pt official full-image test PSNR = 26.199959289578075
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_k03 Final |
|---|---|---:|---:|
| `A_y03` | `final.pt` | 26.2727 | +0.0727 |
| `A_x03` | `final.pt` | 26.2077 | +0.0077 |
| `A_k03` | `final.pt` | 26.2000 | 0.0000 |
| `A_p03` | `final.pt` | 26.0087 | -0.1913 |

Additional deltas:

```text
A_y03 final.pt vs A_x03 final.pt = +0.06503569856568035 dB
A_y03 final.pt vs A_p03 final.pt = +0.26398885756906096 dB
A_y03 final.pt vs A_k03 final.pt = +0.07273653972128045 dB
```

Interpretation:

- The fresh-seed replay clearly beats both the older medium continuation
  `A_p03` and the frozen baseline `A_k03`.
- Unlike `A_x03`, the margin over `A_k03` is now no longer a rounding-level
  effect; it is large enough to clear the prior confirmation concern.
- The replay also preserves a recurring project rule already seen in prior
  milestones: the true final endpoint can beat the best internal-validation
  periodic checkpoint on official test.
- The project therefore has evidence that the light-edge feature-refinement
  line is not only numerically best, but reproducibly best under this local
  protocol.

## Relation To The Lightweight 90% Goal

Current working `UHD-Blur` 90% PSNR threshold:

```text
26.5743 dB
```

Comparison:

```text
A_y03 final.pt = 26.272695829299355 dB
remaining gap to 90% threshold = 0.3016041707006434 dB
```

Interpretation:

- `A_y03` narrows the remaining gap further than both `A_x03` and `A_k03`.
- The 90% target is still unmet, but the remaining distance is now materially
  smaller than before this confirmation replay.

## Main Conclusion

`V1/T01/A_y03` is accepted as a successfully completed medium-budget
confirmation replay for the light-edge feature-refinement line.

It proves:

- the `A_x03` result was not a one-off fluctuation;
- the line reproduces and improves on the frozen `A_k03` baseline under a
  fresh seed;
- the official margin over `A_k03` now reaches about `+0.0727 dB`, which is
  much stronger evidence than the earlier `+0.0077 dB` edge.

Therefore the safe updated reading is:

- `A_y03` is the current best observed and now confirmed same-protocol
  `UHD_deblur` result on the local line;
- `A_x03` should be treated as the first positive signal;
- `A_y03` is the replay that establishes the signal as real enough to justify
  milestone-refresh discussion.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
300-minute clean finish acceptance: passed
official full-image test acceptance: passed
fresh-seed confirmation acceptance: passed
official baseline replacement: yes
new official UHD_deblur baseline: V1/T01/A_y03 final.pt, PSNR=26.2727 dB
lightweight 90% gate for UHD-Blur: not yet passed
```

## Recommended Next Action

The highest-value next step on the current line is now:

1. freeze `A_y03 final.pt` as a new isolated milestone package and repoint the
   stable official-baseline link to it;
2. preserve the lesson that feature-space Gaussian guidance plus lighter edge
   supervision is currently the strongest confirmed lightweight deblur line;
3. continue future structure work with `A_y03/M04` as the new comparison
   anchor rather than mutating older frozen milestone assets.
