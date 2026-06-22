# Result Analysis: V1/T01/A_x03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_x03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `21441`
- Final logged elapsed time: `299.9925` minutes
- Final checkpoint: `experiments/V1/runs/A_x03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_x03/checkpoints/step_021000.pt`
- Training log: `experiments/V1/runs/A_x03/logs/train_20260516_020341.log`
- Train metrics: `experiments/V1/runs/A_x03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_x03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_x03/metrics/official_test_final.json`

The run completed by the configured `300` minute cap, not by reaching
`max_steps=24000`. No NaN, data-loader failure, checkpoint failure, or
post-validation OOM occurred during training.

Technical note:

- `step_021000.pt` and `latest.pt` have identical `model` weights.
- `final.pt` is **not** identical to `step_021000.pt` / `latest.pt`.
- This means the accepted official result must be attributed to the true
  post-step-`21441` endpoint rather than to the last periodic checkpoint.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.23233415186405182
```

The final logged training loss before clean finish was:

```text
step 21440 loss_total = 0.03385142609477043
```

Interpretation:

- The promoted light-edge feature-refinement line remains stable for the full
  five-hour window.
- No new runtime instability appears despite the feature-refinement path and
  the near-94G real startup envelope.
- So the engineering question is answered positively: the `A_u03` structure is
  not only short-run safe, but medium-budget trainable.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 2000 | 0.024844 | 28.4105 |
| 3000 | 0.024225 | 28.6890 |
| 4000 | 0.023497 | 28.9818 |
| 5000 | 0.023477 | 28.9061 |
| 6000 | 0.022090 | 29.2207 |
| 7000 | 0.022188 | 29.2301 |
| 8000 | 0.021538 | 29.3033 |
| 9000 | 0.021595 | 29.3672 |
| 10000 | 0.021586 | 29.2945 |
| 11000 | 0.021250 | 29.2793 |
| 12000 | 0.021065 | 29.3655 |
| 13000 | 0.021200 | 29.2577 |
| 14000 | 0.021344 | 29.3433 |
| 15000 | 0.021155 | 29.3060 |
| 16000 | 0.021233 | 29.3122 |
| 17000 | 0.021065 | 29.4230 |
| 18000 | 0.021205 | 29.2470 |
| 19000 | 0.020834 | 29.5140 |
| 20000 | 0.020901 | 29.3394 |
| 21000 | 0.020847 | 29.4099 |

The best training-time validation PSNR is:

```text
step 19000
val_loss = 0.020833698753267526
val_psnr = 29.51399547894033
```

The lowest training-time validation loss is the same point:

```text
step 19000
val_loss = 0.020833698753267526
val_psnr = 29.51399547894033
```

Interpretation:

- Internal validation is stronger than both `A_p03` and `A_k03` on its best
  point.
- The line peaks meaningfully before the final training endpoint, so this run
  continues the project-wide pattern that the best internal-validation
  checkpoint is not necessarily the final official-test winner.
- The promoted `A_u03` structure therefore has real medium-budget headroom,
  not just a short-run bump.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_x03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.033038 | 26.2077 |

Interpretation:

- The light-edge feature-refinement line scales beyond its short-run quality
  signal and surpasses the previous medium continuation `A_p03`.
- It also edges above the frozen `A_k03` baseline on same-protocol official
  test, but only by a very small margin.
- So this is a real new best result numerically, yet not a decisive
  step-change.

## Comparison To Existing UHD Deblur Anchors

Frozen milestone before this run:

```text
A_k03 final.pt official full-image test PSNR = 26.199959289578075
```

Relevant same-task same-protocol references:

```text
A_u03 final.pt official full-image test PSNR = 25.310064081965912
A_v03 final.pt official full-image test PSNR = 25.271115184810476
A_t03 final.pt official full-image test PSNR = 25.294446219726133
A_p03 final.pt official full-image test PSNR = 26.008706971730295
A_k03 final.pt official full-image test PSNR = 26.199959289578075
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_k03 Final |
|---|---|---:|---:|
| `A_x03` | `final.pt` | 26.2077 | +0.0077 |
| `A_k03` | `final.pt` | 26.2000 | 0.0000 |
| `A_p03` | `final.pt` | 26.0087 | -0.1913 |
| `A_u03` | `final.pt` | 25.3101 | -0.8899 |
| `A_t03` | `final.pt` | 25.2944 | -0.9055 |
| `A_v03` | `final.pt` | 25.2711 | -0.9288 |

Additional deltas:

```text
A_x03 final.pt vs A_u03 final.pt = +0.8975960487677632 dB
A_x03 final.pt vs A_v03 final.pt = +0.9365449459231989 dB
A_x03 final.pt vs A_t03 final.pt = +0.9132139110075421 dB
A_x03 final.pt vs A_p03 final.pt = +0.19895315900338062 dB
A_x03 final.pt vs A_k03 final.pt = +0.007700841155600102 dB
```

Interpretation:

- The five-hour promotion converts the `A_u03` short-run clue into a
  meaningful medium-budget gain over the same family and over `A_p03`.
- However, the margin over frozen `A_k03` is only about `+0.0077 dB`, which is
  too small to claim a strong milestone-level breakthrough with confidence.
- The safe reading is therefore: `A_x03` is the current best observed number,
  but not yet a robust enough separation to justify immediate milestone
  replacement without further confirmation.

## Relation To The Lightweight 90% Goal

Current working `UHD-Blur` 90% PSNR threshold:

```text
26.5743 dB
```

Comparison:

```text
A_x03 final.pt = 26.207660130733675 dB
remaining gap to 90% threshold = 0.366639869266323 dB
```

Interpretation:

- `A_x03` slightly narrows the remaining gap beyond `A_k03`.
- But the improvement is small, and the 90% target is still clearly unmet.
- So the run is best treated as a positive continuation result, not as proof
  that the template gap has been solved.

## Main Conclusion

`V1/T01/A_x03` is accepted as a successfully completed medium-budget
`UHD_deblur` promotion run for the light-edge feature-refinement line.

It proves:

- the `A_u03` structure scales cleanly into a full five-hour budget;
- the line delivers a clear medium-budget gain over `A_p03`;
- the official result now reaches a new project-best observed number on the
  deblur line: `26.2077 dB`.

It does not yet prove:

- that the improvement over frozen `A_k03` is large enough to count as a
  meaningful milestone-refresh result;
- that the project should immediately replace the frozen milestone based on a
  `+0.0077 dB` edge alone;
- that the remaining lightweight 90% gap can be closed by more near-isomorphic
  long-budget replay.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
300-minute clean finish acceptance: passed
official full-image test acceptance: passed
best observed UHD_deblur result so far: yes
automatic milestone replacement: not yet recommended
recommended next action: treat A_x03 as the current best long-run result, but require at least one confirming result or clearer margin before replacing frozen M03/A_k03
```
