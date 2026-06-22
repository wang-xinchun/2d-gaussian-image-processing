# Result Analysis: V1/T01/A_n03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_n03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `33.7058` minutes
- Final checkpoint: `experiments/V1/runs/A_n03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_n03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_n03/logs/train_20260515_serial.log`
- Train metrics: `experiments/V1/runs/A_n03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_n03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=2500`, not by the `58`-minute
wall-clock cap. No NaN, data-loader failure, checkpoint failure, or
post-validation OOM occurred.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.12209546566009521
```

The final logged training loss before clean finish was:

```text
step 2500 loss_total = 0.044029928743839264
```

Interpretation:

- The default-strength decoder-guidance path is deployable under the same
  clean `6x784` geometry used by `A_m03`.
- Relative to `A_m03`, this branch shows a more uneven mid-run trajectory, but
  it remains fully stable and reaches the endpoint cleanly.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.025130 | 28.3583 |
| 1000 | 0.022991 | 28.4925 |
| 1500 | 0.023225 | 28.4493 |
| 2000 | 0.023739 | 28.5384 |
| 2500 | 0.021695 | 28.9612 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.021694760490208864
val_psnr = 28.961212847591696
```

Interpretation:

- `A_n03` starts weaker than `A_m03` at the first validation point.
- It partially recovers by `step 1000`, softens again around `1500`, then
  finishes with a strong endpoint jump.
- Under the short-scout family gate, `28.9612 dB` is a clear strong-keep
  result and currently exceeds `A_m03`'s best short-run validation peak.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_n03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.034500 | 25.2919 |

Interpretation:

- This official-test result is used as corroborative evidence only.
- The architectural keep/drop and long-run nomination logic for this probe
  family still primarily follows the short-run full-image validation signal.

## Comparison To A_m03 Short Signal

Relevant short-probe comparison:

| Run | Step 500 | Step 1000 | Step 1500 | Step 2000 | Step 2500 |
|---|---:|---:|---:|---:|---:|
| `A_m03` | 28.6168 | 28.3975 | 28.5168 | 28.4718 | 28.8486 |
| `A_n03` | 28.3583 | 28.4925 | 28.4493 | 28.5384 | 28.9612 |

Interpretation:

- `A_n03` does not dominate uniformly across the short run.
- It trails `A_m03` at `500` and `1500`, but leads at `1000`, `2000`, and the
  final endpoint.
- At the current stage, the strongest evidence in favor of `A_n03` is that its
  endpoint beats `A_m03` by about `+0.1126 dB` on the short-run validation
  gate.

## Relation To The Probe-Family Gate

Family gate thresholds:

```text
best full-image val PSNR >= 28.70 dB: keep candidate
best full-image val PSNR >= 28.80 dB: strong keep candidate
best full-image val PSNR < 28.60 dB: reject
```

Observed `A_n03` short-probe result:

```text
best training-time full-image val PSNR = 28.9612 dB
official test corroboration (final.pt) = 25.2919 dB
```

Interpretation:

- `A_n03` clearly passes the keep gate.
- It also clears the strong-keep threshold by a comfortable margin.
- At the level of training-time full-image validation alone, it currently
  ranks above `A_m03`.

## Main Conclusion

`V1/T01/A_n03` is accepted as a successful completed short screening branch in
the `A_m03/A_n03/A_o03` family.

It proves:

- the default-strength `0.15` decoder-guidance setting remains runtime-safe
  under the accepted `6x784` short-probe geometry;
- the branch can finish with a stronger short-run full-image validation signal
  than the low-strength `A_m03` probe;
- the default-strength branch remains a serious candidate for promotion into
  the later `5`-hour window.

It does not yet prove:

- that it should outrank the forthcoming `A_o03` high-fill scout without the
  near-`A_k03` geometry answer;
- that it should be promoted directly into the `5`-hour window without the
  near-`A_k03` geometry answer from `A_o03`.
