# Result Analysis: V1/T01/A_u03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_u03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `35.6395` minutes
- Final checkpoint: `experiments/V1/runs/A_u03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_u03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_u03/logs/train_20260515_233126.log`
- Train metrics: `experiments/V1/runs/A_u03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_u03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_u03/metrics/official_test_final.json`

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
step 1 loss_total = 0.13822488486766815
```

The final logged training loss before clean finish was:

```text
step 2500 loss_total = 0.023787854239344597
```

Interpretation:

- The lighter-edge branch is fully trainable and stable under the tightened
  `6x728` geometry.
- Reducing the edge-weighted term does not destabilize the feature-refinement
  route.
- The remaining question is not engineering safety, but whether lighter
  supervision pressure improves the real official endpoint.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.023968 | 28.6578 |
| 1000 | 0.025330 | 28.4825 |
| 1500 | 0.024483 | 28.6105 |
| 2000 | 0.024021 | 28.6601 |
| 2500 | 0.023983 | 28.7799 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.02398321758955717
val_psnr = 28.779889882694157
```

Interpretation:

- `A_u03` tracks the same general internal shape as `A_t03`.
- Its internal endpoint is slightly below `A_t03` (`28.7877`) and still below
  `A_n03` (`28.9612`).
- So if we looked only at internal validation, `A_u03` would appear mostly
  like a near-tie continuation rather than an obvious new winner.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_u03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.036647 | 25.3101 |

Interpretation:

- This is the strongest official result so far inside the post-`A_t03`
  rebalance family.
- Unlike the tiny `A_t03` over `A_n03` gap, `A_u03` improves both anchors by
  a somewhat clearer amount.
- The gain is still small in absolute terms, but it is no longer just a
  rounding-level nudge.

## Comparison To Existing UHD Deblur Anchors

Frozen milestone before this run:

```text
A_k03 final.pt official full-image test PSNR = 26.199959289578075
```

Relevant same-task same-protocol references:

```text
A_n03 final.pt official full-image test PSNR = 25.291929097577473
A_p03 final.pt official full-image test PSNR = 26.008706971730295
A_r03 final.pt official full-image test PSNR = 25.286687395113344
A_s03 final.pt official full-image test PSNR = 25.25673342402521
A_t03 final.pt official full-image test PSNR = 25.294446219726133
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_k03 Final |
|---|---|---:|---:|
| `A_k03` | `final.pt` | 26.2000 | 0.0000 |
| `A_p03` | `final.pt` | 26.0087 | -0.1913 |
| `A_u03` | `final.pt` | 25.3101 | -0.8899 |
| `A_t03` | `final.pt` | 25.2944 | -0.9055 |
| `A_n03` | `final.pt` | 25.2919 | -0.9080 |
| `A_r03` | `final.pt` | 25.2867 | -0.9133 |
| `A_s03` | `final.pt` | 25.2567 | -0.9432 |
```

Additional deltas:

```text
A_u03 final.pt vs A_t03 final.pt = +0.015617862239778664 dB
A_u03 final.pt vs A_n03 final.pt = +0.018134984388439264 dB
A_u03 final.pt vs A_r03 final.pt = +0.023376686852568625 dB
A_u03 final.pt vs A_s03 final.pt = +0.05333065794070163 dB
A_u03 final.pt vs A_p03 final.pt = -0.6986428897643832 dB
A_u03 final.pt vs A_k03 final.pt = -0.8898952076121631 dB
```

Interpretation:

- `A_u03` is the clearest official positive signal so far in this rebalance
  family.
- The result supports a very specific lesson: in the current feature-refinement
  branch, the original `A_t03` edge-weighted term was probably a little too
  strong.
- But the absolute gap to `A_p03` and frozen `A_k03` remains large, so this is
  still an incremental gain, not a milestone-level jump.

## Main Conclusion

`V1/T01/A_u03` is accepted as a successfully completed short rebalance probe
with a **small but real positive** official result.

It proves:

- the lighter-edge variant is runtime-safe and trainable under the same
  tightened `6x728` geometry;
- reducing `edge_weighted_reconstruction_strength` from `0.03` to `0.02`
  improves official test over both `A_t03` and `A_n03`;
- the best signal in this family now points toward slightly less supervision
  pressure rather than more.

It does not prove:

- that the family has closed the large absolute gap to `A_p03` or `A_k03`;
- that `A_u03` is already strong enough to skip the remaining `A_v03` and
  `A_w03` rebalance probes;
- that internal validation alone would have reliably predicted the official
  improvement.

What it suggests instead is:

- the current highest-confidence family leader is now `A_u03`;
- the most plausible next comparison question is whether stronger mechanism
  pressure (`A_v03`) or coordinated rebalance (`A_w03`) can beat this new
  lighter-edge leader;
- if neither remaining probe clears `A_u03`, then the family answer is likely
  “keep feature refinement, but relax the edge-weighted term.”

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official baseline replacement: no
best completed rebalance-family result so far: yes
recommended next action: continue serially with A_v03 and A_w03, but treat A_u03 as the current family leader to beat
```
