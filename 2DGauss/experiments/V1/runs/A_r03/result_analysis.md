# Result Analysis: V1/T01/A_r03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_r03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `35.6173` minutes
- Final checkpoint: `experiments/V1/runs/A_r03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_r03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_r03/logs/train_20260515_202545.log`
- Train metrics: `experiments/V1/runs/A_r03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_r03/metrics/val_metrics.jsonl`
- Official test metrics: `experiments/V1/runs/A_r03/metrics/official_test_final.json`

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
step 1 loss_total = 0.13256430625915527
```

The final logged training loss before clean finish was:

```text
step 2500 loss_total = 0.022078227251768112
```

Interpretation:

- The feature-space refinement branch is fully trainable and stable once the
  geometry is tightened to `6x728`.
- So the branch answers the engineering question positively: moving the
  high-resolution intervention into feature space is feasible without runtime
  collapse.
- The remaining question is whether that cleaner mechanism translates into a
  meaningful same-protocol official gain.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.023650 | 28.5712 |
| 1000 | 0.025957 | 28.2523 |
| 1500 | 0.023383 | 28.6170 |
| 2000 | 0.022729 | 28.6553 |
| 2500 | 0.022803 | 28.7519 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.022803147602826358
val_psnr = 28.751903874074475
```

Interpretation:

- `A_r03` produces a healthy short-run internal signal and finishes strongest
  at the endpoint.
- It clearly beats `A_s03` on official potential indicators from the internal
  side, which supports the idea that feature-space intervention is stronger
  than supervision-only intervention.
- But the internal peak still remains below the promoted short anchor
  `A_n03` (`28.9612 dB`), so the branch does not enter official evaluation as
  an obvious new winner.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_r03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.034657 | 25.2867 |

Interpretation:

- This is a mild but real improvement over `A_s03` and `A_q03`.
- It also edges above `A_m03` and `A_o03`.
- However, it still falls just short of `A_n03`, so feature refinement alone
  does not yet become the new short promotion anchor.

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
A_s03 final.pt official full-image test PSNR = 25.25673342402521
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_k03 Final |
|---|---|---:|---:|
| `A_k03` | `final.pt` | 26.2000 | 0.0000 |
| `A_p03` | `final.pt` | 26.0087 | -0.1913 |
| `A_n03` | `final.pt` | 25.2919 | -0.9080 |
| `A_r03` | `final.pt` | 25.2867 | -0.9133 |
| `A_o03` | `final.pt` | 25.2809 | -0.9191 |
| `A_m03` | `final.pt` | 25.2690 | -0.9310 |
| `A_s03` | `final.pt` | 25.2567 | -0.9432 |
| `A_q03` | `final.pt` | 25.1826 | -1.0174 |

Additional deltas:

```text
A_r03 final.pt vs A_m03 final.pt = +0.017680264538000756 dB
A_r03 final.pt vs A_n03 final.pt = -0.005241702464129361 dB
A_r03 final.pt vs A_o03 final.pt = +0.0058128652356082 dB
A_r03 final.pt vs A_p03 final.pt = -0.722019576616951 dB
A_r03 final.pt vs A_q03 final.pt = +0.10413329747646216 dB
A_r03 final.pt vs A_s03 final.pt = +0.029953971088133358 dB
A_r03 final.pt vs A_k03 final.pt = -0.9132718944647316 dB
```

Interpretation:

- This is the strongest member so far inside the post-`A_q03` short family.
- But the margin over `A_n03` is still slightly negative, so it is best
  described as a near-tie rather than a promotion win.
- The branch supports the structural hypothesis more than `A_s03` did, because
  feature-space refinement performs better than supervision-only, but not yet
  strongly enough to settle the family on its own.

## Main Conclusion

`V1/T01/A_r03` is accepted as a successfully completed short feature-refinement
probe with a **near-neutral quality result** relative to the current short
anchor.

It proves:

- the feature-space refinement mechanism is runtime-safe and trainable under
  the tightened `6x728` geometry;
- moving the intervention from terminal RGB space to feature space is better
  than the negative `A_q03` path;
- the feature-space route is also stronger than supervision-only `A_s03`.

It does not prove:

- that feature refinement alone is enough to replace `A_n03` as the next short
  anchor;
- that this branch deserves direct medium/long-budget promotion without another
  corroborating signal;
- that the remaining deblur gap is now solved.

What it suggests instead is:

- the main bottleneck really is more structural than loss-only;
- the most informative next short deployment is now `A_t03`, because the
  combination branch is the remaining way to test whether the small positive
  effect of feature-space refinement becomes clearer when paired with
  detail-aware supervision.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
short-probe completion: passed
official baseline replacement: no
preferred next action: deploy A_t03 before any longer-budget promotion
```

## Recommended Next Action

The highest-value next step on the current run line is now:

1. record `A_r03` as a clean near-tie structural probe;
2. preserve the lesson that feature-space refinement is stronger than
   supervision-only, but not yet decisive by itself;
3. move the next short deployment to `A_t03` under the already prepared
   `6x728` geometry.
