# Result Analysis: V1/T01/A_ab03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ab03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `36.1174` minutes
- Final checkpoint: `experiments/V1/runs/A_ab03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_ab03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_ab03/logs/train_20260517_234834.log`
- Train metrics: `experiments/V1/runs/A_ab03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_ab03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_ab03/metrics/official_test_final.json`

The run completed by reaching `max_steps=2500`, not by the `58`-minute
wall-clock cap. No NaN, data-loader failure, checkpoint failure, or
runtime OOM occurred during training.

Technical note:

- `step_002500.pt`, `latest.pt`, and `final.pt` have identical `model`
  weights.
- Their serialized checkpoint files differ because of non-model payload, but
  the effective trained endpoint is the same.
- This means the accepted official result of `final.pt` is also the effective
  result of the true step-`2500` endpoint.

## Training Behavior

The branch showed a moderately uneven but overall stable short-run curve:

- `step=500`: `val_psnr=28.5939`
- `step=1000`: `val_psnr=28.5592`
- `step=1500`: `val_psnr=28.5976`
- `step=2000`: `val_psnr=28.2654`
- `step=2500`: `val_psnr=28.7948`

Interpretation:

- The edge-aware prior did not collapse training.
- Internal validation remained within the usual short-run band.
- But the live curve never produced a clearly stronger signal than the
  existing short positive `A_z03`.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.024854 | 28.5939 |
| 1000 | 0.024693 | 28.5592 |
| 1500 | 0.023999 | 28.5976 |
| 2000 | 0.025755 | 28.2654 |
| 2500 | 0.024028 | 28.7948 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.024027574341744184
val_psnr = 28.79484848823599
```

The lowest training-time validation loss is:

```text
step 1500
val_loss = 0.023998875450342894
val_psnr = 28.59761568679224
```

Interpretation:

- Unlike `A_z03`, the best internal validation PSNR here is not especially
  strong relative to the nearby short family.
- The final endpoint does recover into the best PSNR point of the run, but
  the magnitude of that recovery is not large enough to imply a likely short
  promotion win.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_ab03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.037290 | 25.1723 |

Interpretation:

- `A_ab03` is a cleanly completed short probe, but it does **not** convert
  the new edge-aware prior into a positive same-protocol benchmark result.
- The branch is runtime-safe and structurally interpretable, but it is not a
  winning short continuation above the current post-`M04` leader set.

## Comparison To Existing UHD Deblur Anchors

Relevant same-task same-protocol references:

```text
A_z03 final.pt official full-image test PSNR = 25.385710820365173
A_u03 final.pt official full-image test PSNR = 25.310064081965912
A_t03 final.pt official full-image test PSNR = 25.294446219726133
A_n03 final.pt official full-image test PSNR = 25.291929097577473
A_r03 final.pt official full-image test PSNR = 25.286687395113344
A_v03 final.pt official full-image test PSNR = 25.271115184810476
A_s03 final.pt official full-image test PSNR = 25.25673342402521
A_p03 final.pt official full-image test PSNR = 26.008706971730295
A_k03 final.pt official full-image test PSNR = 26.199959289578075
A_y03 final.pt official full-image test PSNR = 26.272695829299355
A_aa03 final.pt official full-image test PSNR = 26.084983286081467
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_ab03 Final |
|---|---|---:|---:|
| `A_ab03` | `final.pt` | 25.1723 | 0.0000 |
| `A_z03` | `final.pt` | 25.3857 | +0.2134 |
| `A_u03` | `final.pt` | 25.3101 | +0.1378 |
| `A_t03` | `final.pt` | 25.2944 | +0.1221 |
| `A_n03` | `final.pt` | 25.2919 | +0.1196 |
| `A_r03` | `final.pt` | 25.2867 | +0.1144 |
| `A_v03` | `final.pt` | 25.2711 | +0.0988 |
| `A_s03` | `final.pt` | 25.2567 | +0.0844 |
| `A_p03` | `final.pt` | 26.0087 | +0.8364 |
| `A_aa03` | `final.pt` | 26.0850 | +0.9127 |
| `A_k03` | `final.pt` | 26.2000 | +1.0276 |
| `A_y03` | `final.pt` | 26.2727 | +1.1004 |

Additional deltas:

```text
A_ab03 final.pt vs A_z03 final.pt = -0.2133989259367084 dB
A_ab03 final.pt vs A_u03 final.pt = -0.13775218753744767 dB
A_ab03 final.pt vs A_t03 final.pt = -0.12213432529766877 dB
A_ab03 final.pt vs A_n03 final.pt = -0.11961720314900859 dB
A_ab03 final.pt vs A_r03 final.pt = -0.11437550068487923 dB
A_ab03 final.pt vs A_v03 final.pt = -0.09880329038201197 dB
A_ab03 final.pt vs A_s03 final.pt = -0.08442152959674587 dB
A_ab03 final.pt vs A_p03 final.pt = -0.8363950773018303 dB
A_ab03 final.pt vs A_k03 final.pt = -1.0276473951496108 dB
A_ab03 final.pt vs A_y03 final.pt = -1.1003839348708908 dB
```

Interpretation:

- `A_ab03` is clearly below `A_z03`, which means the first edge-aware
  feature-refinement redesign does not beat the current post-`M04` short
  leader.
- It is also below the older short anchors `A_u03/A_t03/A_n03`, so this is
  not just a near-tie loss.
- Therefore the correct conclusion is not "edge structure matters and should
  be promoted," but "this direct edge-aware prior formulation is not the right
  lightweight upgrade under the current envelope."

## Main Conclusion

`V1/T01/A_ab03` is accepted as a successfully completed short probe with a
negative official result.

It proves:

- the edge-aware prior formulation is runtime-safe under the same practical
  VRAM envelope;
- the branch can train cleanly and close at the true short-run endpoint;
- direct explicit edge-aware prior injection into feature refinement is not,
  by itself, a stronger short continuation than the existing `A_z03` line.

It does not prove:

- that edge information is useless globally;
- that feature-space Gaussian refinement is the wrong family;
- that the post-`A_aa03` redesign space is exhausted.

What it does prove is narrower and more useful:

- this particular direct edge-aware-prior implementation should **not** be the
  next medium-budget promotion candidate.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official full-image test acceptance: passed
best completed post-M04 short probe so far: no
medium-promotion candidate: no
recommended next action: retire A_ab03 as a negative structural probe and move to the next prepared orthogonal short direction
```
