# Result Analysis: V1/T01/A03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A03`
- Dataset: `UHD_deblur`
- Training split: `1864` pairs from `train`
- Internal validation split: `100` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2000`
- Final logged elapsed time: `17.7587` minutes
- Final checkpoint: `experiments/V1/runs/A03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A03/checkpoints/step_002000.pt`
- Training log: `experiments/V1/runs/A03/logs/train_20260506_011651.log`
- Train metrics: `experiments/V1/runs/A03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=2000`, not by the `wall_clock_minutes=58`
cap. This means the selected `UHD_deblur` short-run envelope was valid and
comfortably inside the short tier.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.10514473170042038
```

The final logged training loss was:

```text
step 2000 loss_total = 0.0343700647354126
```

Segment means:

| Step range | Mean train loss | Median | Min | Max |
|---|---:|---:|---:|---:|
| 1-250 | 0.039439 | 0.036458 | 0.025956 | 0.105145 |
| 251-500 | 0.038933 | 0.037823 | 0.028253 | 0.054582 |
| 501-1000 | 0.037611 | 0.037178 | 0.025049 | 0.054368 |
| 1001-1500 | 0.035843 | 0.034959 | 0.026417 | 0.053884 |
| 1501-2000 | 0.035876 | 0.034900 | 0.023643 | 0.051872 |

Interpretation:

- The model learned quickly in the first few hundred steps.
- After the early drop, train loss remained stable in the `0.03x` to low `0.04x`
  range.
- No NaN, OOM, or divergence was observed.

## Training-Time Validation

Training-time validation used `100` batches from the internal validation split,
so unlike `A01`, it is already a meaningful checkpoint-selection signal.

Best training-time validation point:

```text
step 500
val_loss = 0.025427820812910795
val_psnr = 29.00563267209822
```

Last training-time validation point:

```text
step 2000
val_loss = 0.025943749761208893
val_psnr = 28.296640982009375
```

Validation trajectory summary:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 250 | 0.025622 | 28.8894 |
| 500 | 0.025428 | 29.0056 |
| 750 | 0.025552 | 28.9018 |
| 1000 | 0.025580 | 28.8240 |
| 1250 | 0.025747 | 28.6652 |
| 1500 | 0.025771 | 28.6687 |
| 1750 | 0.025558 | 28.5842 |
| 2000 | 0.025944 | 28.2966 |

Interpretation:

- The internal validation curve peaked early at `step 500`.
- After that, validation stayed generally strong but drifted downward.
- This is a clear sign that `UHD_deblur` under the current `V1/T01` setup is
  learnable, but the best short-run checkpoint is likely earlier than the final
  checkpoint if one only looks at internal validation.

## Post-Hoc Full Evaluation

Post-hoc evaluation was run on:

- full internal validation set: `100` samples
- official test set: `300` samples

Evaluated checkpoints:

- `experiments/V1/runs/A03/checkpoints/step_000500.pt`
- `experiments/V1/runs/A03/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `step_000500.pt` | full internal val | 100 | 0.025428 | 29.0056 |
| `step_000500.pt` | official test | 300 | 0.035098 | 25.0978 |
| `final.pt` | full internal val | 100 | 0.025944 | 28.2966 |
| `final.pt` | official test | 300 | 0.035112 | 25.1402 |

Interpretation:

- The post-hoc full validation set favors `step_000500.pt`.
- The official test set slightly favors `final.pt`.
- The official test PSNR gap is small: `final.pt` is ahead of `step_000500.pt`
  by about `0.0424 dB`.
- This means `A03` does not collapse late in training, but the internal
  validation and official test preferences are not perfectly aligned.

## Main Conclusion

`V1/T01/A03` is a successful first `UHD_deblur` deployment.

It proves:

- the current `V1/T01` line transfers from low-light and dehazing to UHD
  deblurring;
- the `UHD_deblur` data-loading envelope with `batch_size=16` and
  `num_workers=16` is valid for short training;
- the run reaches stable train/validation/checkpoint milestones without runtime
  failure;
- the project now has a reproducible first `UHD_deblur` baseline.

From a reporting perspective:

- if the selection rule is **internal validation**, `step_000500.pt` is the best
  checkpoint;
- if the selection rule is **official test after training**, `final.pt` is
  marginally better.

For the project baseline record, the most conservative summary is:

> `A03` successfully established the first `UHD_deblur` baseline, with official
> test PSNR `25.1402 dB` at `final.pt`, while the best full internal validation
> checkpoint was `step_000500.pt` at `29.0056 dB` internal-val PSNR.

## Recommended Next Action

The next most valuable `UHD_deblur` step is not another blind replay of `A03`,
but a branch run such as `A_b03` or equivalent, designed explicitly to improve
over the newly established baseline.

Recommended priorities:

- use `A03` as the first official `UHD_deblur` baseline;
- choose one checkpoint-selection policy and keep it stable in the next branch;
- target same-protocol improvement of about `+0.5 dB` on the preferred
  evaluation split, consistent with the current stage metric policy.
