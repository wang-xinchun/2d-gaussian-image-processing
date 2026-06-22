# Result Analysis: V1/T01/A_ae03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ae03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `36.1353` minutes
- Final checkpoint: `experiments/V1/runs/A_ae03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_ae03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_ae03/logs/train_20260518_133133.log`
- Train metrics: `experiments/V1/runs/A_ae03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_ae03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_ae03/metrics/official_test_final.json`

The run completed by reaching `max_steps=2500`, not by the `58`-minute wall-clock cap. No NaN, checkpoint failure, dataloader failure, or runtime OOM occurred during training.

Technical note:

- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model` weights.
- Their serialized files may differ because of non-model payload, but the effective trained endpoint is the same.
- Therefore the accepted official result of `final.pt` is also the true result of the actual `step=2500` endpoint.

## Training Behavior

The branch showed a steady and fairly flat internal-validation trajectory across the short run:

- `step=500`: `val_psnr=28.5609`
- `step=1000`: `val_psnr=28.3061`
- `step=1500`: `val_psnr=28.6193`
- `step=2000`: `val_psnr=28.6578`
- `step=2500`: `val_psnr=28.7005`

Interpretation:

- Shared-prior confidence gating did not create a dramatic live spike, but it remained stable and improved again by the endpoint.
- The run did not collapse or become erratic after the shared-prior change.
- This makes it a clean short probe, but not yet a decisive promotion winner.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image `crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.025060 | 28.5609 |
| 1000 | 0.024883 | 28.3061 |
| 1500 | 0.024046 | 28.6193 |
| 2000 | 0.024453 | 28.6578 |
| 2500 | 0.024563 | 28.7005 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.024562807008624076
val_psnr = 28.70049307976054
```

The lowest training-time validation loss is:

```text
step 1500
val_loss = 0.02404609238728881
val_psnr = 28.61931127206347
```

Interpretation:

- The branch ends at its best validation PSNR, even though the loss minimum is slightly earlier.
- The validation profile is healthy, but it does not by itself prove a stronger official result than the current short leader.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_ae03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.037085 | 25.2533 |

Interpretation:

- `A_ae03` improves over `A_ac03` and `A_ad03` on the official test.
- But it still does not beat `A_z03`, and it remains below the older medium line `M04/A_y03`.
- So it is a stronger local short probe, not yet a promotion candidate.

## Comparison To Existing UHD Deblur Anchors

Relevant same-task same-protocol references:

```text
A_ac03 final.pt official full-image test PSNR = 25.147392029064648
A_ad03 final.pt official full-image test PSNR = 25.200779442877334
A_z03 final.pt official full-image test PSNR = 25.385710820365173
A_y03 final.pt official full-image test PSNR = 26.272695829299355
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_ae03 Final |
|---|---|---:|---:|
| `A_ae03` | `final.pt` | 25.2533 | 0.0000 |
| `A_ad03` | `final.pt` | 25.2008 | -0.0525 |
| `A_ac03` | `final.pt` | 25.1474 | -0.1059 |
| `A_z03` | `final.pt` | 25.3857 | +0.1324 |
| `A_y03` | `final.pt` | 26.2727 | +1.0194 |

Additional deltas:

```text
A_ae03 final.pt vs A_ad03 final.pt = +0.05249473309750391 dB
A_ae03 final.pt vs A_ac03 final.pt = +0.10588214691018931 dB
A_ae03 final.pt vs A_z03 final.pt = -0.1324366443903348 dB
A_ae03 final.pt vs A_y03 final.pt = -1.019421653324517 dB
```

Interpretation:

- `A_ae03` is the best result among the three short probes `A_ac03/A_ad03/A_ae03`.
- It is still below `A_z03`, so the short-run leader remains unchanged.
- It also remains well below the frozen medium milestone `M04/A_y03`.

## Main Conclusion

`V1/T01/A_ae03` is accepted as a successfully completed short probe with a positive local gain over the previous two redesigns, but not a new leader.

It proves:

- shared-prior confidence gating is runtime-safe under the short-tier `6x720` envelope;
- this branch improves over both `A_ac03` and `A_ad03` on the official test;
- the shared-control-surface direction is still worth keeping alive.

It does not prove:

- that the current shared-prior confidence gate is enough to beat `A_z03`;
- that the absolute gap to `M04/A_y03` is closed;
- that this branch should be promoted to medium training as-is.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official full-image test acceptance: passed
best completed post-M04 short probe so far: no
best result inside the A_ac03/A_ad03/A_ae03 family: yes
medium-promotion candidate: no
recommended next action: keep refining the shared control surface with another lightweight redesign rather than replaying this exact gate
```
