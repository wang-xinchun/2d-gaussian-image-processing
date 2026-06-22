# Result Analysis: V1/T01/A_t03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_t03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `35.6375` minutes
- Final checkpoint: `experiments/V1/runs/A_t03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_t03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_t03/logs/train_20260515_215528.log`
- Train metrics: `experiments/V1/runs/A_t03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_t03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_t03/metrics/official_test_final.json`

The run completed by reaching `max_steps=2500`, not by the `58`-minute
wall-clock cap. No NaN, data-loader failure, checkpoint failure, or
post-validation OOM occurred during training.

Technical note:

- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model`
  weights.
- Their serialized checkpoint files differ because of non-model payload, but
  the effective trained endpoint is the same.
- This means the accepted official result of `final.pt` is also the effective
  result of the true step-`2500` endpoint.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.1410551816225052
```

The final logged training loss before clean finish was:

```text
step 2500 loss_total = 0.02422221191227436
```

Interpretation:

- The combined branch is fully trainable and stable under the tightened
  `6x728` geometry.
- So the branch answers the engineering question positively: combining
  mechanism-side feature refinement with loss-side edge emphasis does not
  destabilize the short run.
- The remaining question is whether that combination creates a real official
  quality gain rather than only a mild ranking shuffle.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.024566 | 28.6587 |
| 1000 | 0.026263 | 28.4214 |
| 1500 | 0.025458 | 28.5868 |
| 2000 | 0.024413 | 28.6852 |
| 2500 | 0.024678 | 28.7877 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.024677901435643433
val_psnr = 28.78770197583349
```

Interpretation:

- `A_t03` gives a healthy and fairly monotonic late-stage internal signal.
- It finishes above `A_r03` on the internal side, which supports mild
  complementarity between feature refinement and edge-weighted supervision.
- But the internal endpoint still remains below the promoted short anchor
  `A_n03` (`28.9612 dB`), so the branch does not enter official testing as an
  obvious validation winner.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_t03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.037726 | 25.2944 |

Interpretation:

- This is the strongest result inside the post-`A_q03` short family.
- It does edge above `A_n03`, so the combined branch is not a negative result.
- But the gain over `A_n03` is only about `+0.0025 dB`, far below the
  project's meaningful-gain threshold, so it should be treated as a slight
  positive nudge rather than a decisive promotion win.

## Comparison To Existing UHD Deblur Anchors

Frozen milestone before this run:

```text
A_k03 final.pt official full-image test PSNR = 26.199959289578075
```

Relevant same-task same-protocol references:

```text
A_m03 final.pt official full-image test PSNR = 25.269007130575343
A_n03 final.pt official full-image test PSNR = 25.291929097577473
A_o03 final.pt official full-image test PSNR = 25.280874529877735
A_p03 final.pt official full-image test PSNR = 26.008706971730295
A_q03 final.pt official full-image test PSNR = 25.18255409763688
A_r03 final.pt official full-image test PSNR = 25.286687395113344
A_s03 final.pt official full-image test PSNR = 25.25673342402521
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_k03 Final |
|---|---|---:|---:|
| `A_k03` | `final.pt` | 26.2000 | 0.0000 |
| `A_p03` | `final.pt` | 26.0087 | -0.1913 |
| `A_t03` | `final.pt` | 25.2944 | -0.9055 |
| `A_n03` | `final.pt` | 25.2919 | -0.9080 |
| `A_r03` | `final.pt` | 25.2867 | -0.9133 |
| `A_o03` | `final.pt` | 25.2809 | -0.9191 |
| `A_m03` | `final.pt` | 25.2690 | -0.9310 |
| `A_s03` | `final.pt` | 25.2567 | -0.9432 |
| `A_q03` | `final.pt` | 25.1826 | -1.0174 |

Additional deltas:

```text
A_t03 final.pt vs A_m03 final.pt = +0.025439089150790323 dB
A_t03 final.pt vs A_n03 final.pt = +0.0025171221486601504 dB
A_t03 final.pt vs A_o03 final.pt = +0.013571689848397917 dB
A_t03 final.pt vs A_p03 final.pt = -0.7142607520041617 dB
A_t03 final.pt vs A_q03 final.pt = +0.11189212208925192 dB
A_t03 final.pt vs A_r03 final.pt = +0.007758824612789511 dB
A_t03 final.pt vs A_s03 final.pt = +0.03771279570092251 dB
A_t03 final.pt vs A_k03 final.pt = -0.9055130698519417 dB
```

Interpretation:

- The combination branch is indeed the best short probe in this post-`A_q03`
  family.
- The family ordering now supports the idea that feature refinement is the more
  useful ingredient, while edge-weighted supervision is at most a small helper.
- But the margin over `A_n03` is tiny enough that it does not justify claiming
  a meaningful new short anchor or skipping structural redesign.
- Relative to `A_p03` and frozen `M03/A_k03`, the absolute gap remains large.

## Main Conclusion

`V1/T01/A_t03` is accepted as a successfully completed short combined probe
with a **slight positive but not decisive** quality result.

It proves:

- the combined feature-refinement plus edge-weighted route is runtime-safe and
  trainable under the tightened `6x728` geometry;
- the combination is slightly stronger than `A_r03`, `A_s03`, and `A_n03` on
  same-protocol official testing;
- the post-`A_q03` short family does not collapse when the two interventions
  are combined.

It does not prove:

- that the gain over `A_n03` is large enough to count as a meaningful
  short-run breakthrough;
- that this branch should replace `A_n03` as a clear new promotion anchor;
- that the current family closes the remaining absolute gap to `A_p03` or
  frozen `A_k03`.

What it suggests instead is:

- the combination branch is the best short candidate if the project wants one
  more medium-budget confirmation run from this family;
- however, the tiny official margin means this family is close to exhausted as
  a source of short-budget gains, and future larger improvements likely still
  require structural redesign rather than replaying nearby short probes.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official baseline replacement: no
best short result inside this family: yes
recommended next action: treat A_t03 as the best short candidate only if a confirming medium run is desired; otherwise return to structural redesign
```
