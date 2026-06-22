# Result Analysis: V1/T01/A_s03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_s03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `33.7827` minutes
- Final checkpoint: `experiments/V1/runs/A_s03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_s03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_s03/logs/train_20260515_190958.log`
- Train metrics: `experiments/V1/runs/A_s03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_s03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_s03/metrics/official_test_final.json`

The run completed by reaching `max_steps=2500`, not by the `58`-minute
wall-clock cap. No NaN, data-loader failure, checkpoint failure, or
post-validation OOM occurred.

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
step 1 loss_total = 0.12526938319206238
```

The final logged training loss before clean finish was:

```text
step 2500 loss_total = 0.047565411776304245
```

Interpretation:

- The supervision-only branch is fully deployable under the accepted `6x784`
  envelope and does not introduce runtime instability.
- So the branch answers the engineering question positively: adding a small
  edge-weighted reconstruction term is safe.
- The real remaining question is quality transfer to the official protocol,
  not trainability.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.026683 | 28.4094 |
| 1000 | 0.025461 | 28.3611 |
| 1500 | 0.025152 | 28.4896 |
| 2000 | 0.025699 | 28.5492 |
| 2500 | 0.024284 | 28.8699 |

The best training-time validation loss is:

```text
step 2500
val_loss = 0.024284375831484795
val_psnr = 28.869936170894352
```

The best training-time validation PSNR is the same point:

```text
step 2500
val_loss = 0.024284375831484795
val_psnr = 28.869936170894352
```

Interpretation:

- `A_s03` gives a healthy short-run validation signal and finishes stronger
  than its early checkpoints.
- This confirms that the loss-side change is not simply destabilizing the
  optimization.
- But the validation peak still does not exceed the promoted short anchor
  `A_n03` (`28.9612 dB`), so even before official testing it does not look
  like an obvious structural winner.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_s03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.037836 | 25.2567 |

Interpretation:

- The same-protocol official result is weaker than the current decoder-guidance
  short band, despite the positive internal validation trajectory.
- So this is not a “validation and official both rise together” success case.
- The edge-weighted supervision term may help the internal slice emphasize
  details, but that benefit does not transfer cleanly to the official full test
  split in this form.

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
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_k03 Final |
|---|---|---:|---:|
| `A_k03` | `final.pt` | 26.2000 | 0.0000 |
| `A_p03` | `final.pt` | 26.0087 | -0.1913 |
| `A_n03` | `final.pt` | 25.2919 | -0.9080 |
| `A_o03` | `final.pt` | 25.2809 | -0.9191 |
| `A_m03` | `final.pt` | 25.2690 | -0.9310 |
| `A_s03` | `final.pt` | 25.2567 | -0.9432 |
| `A_q03` | `final.pt` | 25.1826 | -1.0174 |

Additional deltas:

```text
A_s03 final.pt vs A_m03 final.pt = -0.012273706550132601 dB
A_s03 final.pt vs A_n03 final.pt = -0.03519567355226272 dB
A_s03 final.pt vs A_o03 final.pt = -0.024141105852525158 dB
A_s03 final.pt vs A_p03 final.pt = -0.7519735477050844 dB
A_s03 final.pt vs A_q03 final.pt = +0.0741793263883288 dB
A_s03 final.pt vs A_k03 final.pt = -0.9432258655528649 dB
```

Interpretation:

- This is a mild negative relative to the decoder-guidance short family, not a
  catastrophic collapse.
- But it is still clearly below `A_n03`, which means the supervision-only idea
  does not become the new preferred short anchor.
- Relative to `A_q03`, `A_s03` is better, so the loss-side change is less bad
  than the earlier direct terminal residual control path.
- Relative to `A_p03` and frozen `M03/A_k03`, the gap remains large enough that
  `A_s03` does not justify direct long-budget promotion.

## Main Conclusion

`V1/T01/A_s03` is accepted as a successfully completed short supervision-only
probe, but it is a **negative-quality result** relative to the current deblur
promotion anchors.

It proves:

- the edge-weighted reconstruction term is runtime-safe and trainable under the
  accepted `6x784` short-probe geometry;
- the branch can produce a healthy internal validation curve;
- the supervision-only route is still directionally stronger than the failed
  direct terminal-residual implementation `A_q03`.

It does not prove:

- that a pure loss-side change is sufficient to solve the current deblur gap;
- that `A_s03` should replace `A_n03` as the next short anchor;
- that this branch deserves direct medium/long-budget promotion.

What it suggests instead is:

- the main bottleneck is still more structural than loss-only;
- if the post-`A_q03` family is to continue, the next higher-value branch should
  shift toward feature-space refinement (`A_r03`) or the combined
  feature-refinement plus supervision route (`A_t03`), rather than replaying
  supervision-only longer.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official baseline replacement: no
preferred next promotion anchor: stay with A_n03-derived mainline, do not promote A_s03 directly
```

## Recommended Next Action

The highest-value next step on the current run line is now:

1. record `A_s03` as a clean but negative supervision-only probe;
2. preserve the lesson that loss-side pressure alone is not enough;
3. move the next short deployment decision to `A_r03` or `A_t03`, both under
   the already accepted `6x728` geometry.
