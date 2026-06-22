# Result Analysis: V1/T01/A_m03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_m03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `33.7139` minutes
- Final checkpoint: `experiments/V1/runs/A_m03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_m03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_m03/logs/train_20260515_004700.log`
- Train metrics: `experiments/V1/runs/A_m03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_m03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=2500`, not by the `58`-minute
wall-clock cap. No NaN, data-loader failure, checkpoint failure, or
post-validation OOM occurred.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.12410826981067657
```

The final logged training loss before clean finish was:

```text
step 2500 loss_total = 0.043891847133636475
```

Interpretation:

- The new decoder-guidance path is runtime-safe under a clean serial launch.
- The conservative low-strength setting does not destabilize the accepted
  `A_k03/M03` recipe.
- The branch therefore passes the first architectural question: this new
  mechanism is not immediately disqualified on engineering grounds.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.022702 | 28.6168 |
| 1000 | 0.023304 | 28.3975 |
| 1500 | 0.023073 | 28.5168 |
| 2000 | 0.024288 | 28.4718 |
| 2500 | 0.022700 | 28.8486 |

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.022699878364801408
val_psnr = 28.84861604749448
```

Interpretation:

- The branch does not look strongest in the middle of the run.
- Instead, it behaves like an endpoint-favoring short probe: the final stage
  recovers clearly above the earlier checkpoints and becomes the best
  validation point.
- Under the short-scout family gate, `28.8486 dB` crosses the strong-keep
  band `>= 28.80 dB`.

## Post-Hoc Official Evaluation

Post-hoc evaluation was run on:

- official test set: `300` samples, `crop_size=0`

Evaluated checkpoint:

- `experiments/V1/runs/A_m03/checkpoints/final.pt`

Result:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `final.pt` | official test | 300 | 0.034553 | 25.2690 |

Interpretation:

- This official-test result is only corroborative for the short-probe family,
  not the formal promotion gate.
- Even so, it is useful evidence that the low-strength decoder-guidance branch
  is not a catastrophic transfer failure on the full official protocol.

## Comparison To Existing UHD Deblur References

Current frozen official baseline:

```text
A_k03 final.pt official full-image test PSNR = 26.199959289578075
```

Relevant local references:

| Run | Checkpoint | Official Test PSNR | Delta vs A_k03 Final |
|---|---|---:|---:|
| `A_j03` | `final.pt` | 25.196463325894427 | -1.0035 |
| `A_g03` | `final.pt` | 25.3800849048404 | -0.8199 |
| `A_k03` | `final.pt` | 26.199959289578075 | 0.0000 |
| `A_m03` | `final.pt` | 25.269007130575343 | -0.9309 |

Additional deltas:

```text
A_m03 final.pt vs A_j03 final.pt = +0.07254380468091637 dB
A_m03 final.pt vs A_g03 final.pt = -0.11107777426505635 dB
A_m03 final.pt vs A_k03 final.pt = -0.9309521590027318 dB
```

Interpretation:

- The corroborative official result is still far below the frozen `A_k03/M03`
  baseline, so `A_m03` is not evidence for immediate baseline replacement.
- But the short-probe purpose here is narrower than baseline replacement:
  relative to the clearly negative `A_j03` reference, `A_m03` does not look
  like a dead architectural branch.

## Relation To The Probe-Family Gate

Family gate thresholds:

```text
best full-image val PSNR >= 28.70 dB: keep candidate
best full-image val PSNR >= 28.80 dB: strong keep candidate
best full-image val PSNR < 28.60 dB: reject
```

Observed `A_m03` short-probe result:

```text
best training-time full-image val PSNR = 28.8486 dB
official test corroboration (final.pt) = 25.2690 dB
```

Interpretation:

- `A_m03` clearly passes the keep gate.
- It also clears the strong-keep threshold.
- This means the low-strength decoder-guidance branch deserves direct
  comparison against `A_n03` and `A_o03` before the next `5`-hour run is
  chosen.

## Main Conclusion

`V1/T01/A_m03` is accepted as a successful completed short screening branch in
the `A_m03/A_n03/A_o03` family.

It proves:

- the new Gaussian-guided decoder modulation is deployable under the required
  `85-93G` runtime window at `batch=6`, `context=784`;
- a conservative `strength=0.10` setting already gives a strong keep-level
  short-run validation signal;
- the branch is viable enough to remain in contention for the later `5`-hour
  promotion stage.

It does not prove:

- that low-strength decoder guidance is already better than the accepted
  `A_k03/M03` baseline;
- that it should be promoted ahead of the default-strength or high-fill scouts
  without direct comparison.
