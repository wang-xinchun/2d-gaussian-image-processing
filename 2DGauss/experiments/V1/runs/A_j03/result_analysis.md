# Result Analysis: V1/T01/A_j03

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_j03`
- Dataset: `UHD_deblur`
- Training split: `1924` pairs from `train`
- Internal validation split: `20` pairs from `train`
- Official test split: `300` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2500`
- Final logged elapsed time: `35.0091` minutes
- Final checkpoint: `experiments/V1/runs/A_j03/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_j03/checkpoints/step_002500.pt`
- Training log: `experiments/V1/runs/A_j03/logs/train_20260514_091412.log`
- Queue log: `experiments/V1/runs/A_j03/logs/queue_20260514_091412.log`
- Train metrics: `experiments/V1/runs/A_j03/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_j03/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=2500`, not by the `wall_clock_minutes=58`
cap. The initial deployment was blocked by a workspace-external Spark process,
but the queued launcher later acquired a clean GPU, passed gating, and
completed training normally.

Technical note:

- `step_002500.pt` and `final.pt` have identical `model` weights.
- This means the accepted official result of `final.pt` is also the effective
  result of the true step-`2500` endpoint.

## Training Behavior

The branch trained stably after queued launch and did not show NaN, checkpoint
failure, or post-validation OOM.

Interpretation:

- Enlarging consistency-training context to `864` while keeping the accepted
  `A_i03` data split and mechanisms fixed is deployable under a high-VRAM
  envelope.
- However, short-budget stability alone is not enough; this branch must be
  judged by whether it gives a useful official-test signal for the next long
  run.

## Training-Time Full-Image Validation

Training-time validation used the first `20` train pairs with full-image
`crop_size=0`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.023142 | 28.5388 |
| 1000 | 0.022495 | 28.6247 |
| 1500 | 0.024006 | 28.2505 |
| 2000 | 0.023736 | 28.4924 |
| 2500 | 0.023385 | 28.6460 |

The best training-time validation loss is:

```text
step 1000
val_loss = 0.022494712937623264
val_psnr = 28.624749595707772
```

The best training-time validation PSNR is:

```text
step 2500
val_loss = 0.02338494062423706
val_psnr = 28.646022120991763
```

Interpretation:

- The branch again shows mild checkpoint-selection tension:
  best validation loss appears earlier, while best validation PSNR is the final
  endpoint.
- The internal-validation scale is also weaker than `A_i03`, despite using the
  same effective train split.

## Post-Hoc Full Evaluation

Post-hoc evaluation was run on:

- full internal validation set: `20` samples, `crop_size=0`
- official test set: `300` samples, `crop_size=0`

Evaluated checkpoints:

- `experiments/V1/runs/A_j03/checkpoints/step_001000.pt`
- `experiments/V1/runs/A_j03/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `step_001000.pt` | full internal val | 20 | 0.022495 | 28.6247 |
| `step_001000.pt` | official test | 300 | 0.035382 | 25.0181 |
| `final.pt` | full internal val | 20 | 0.023385 | 28.6460 |
| `final.pt` | official test | 300 | 0.035085 | 25.1965 |

Interpretation:

- The final checkpoint is clearly better than the internal-val-loss winner on
  the official test, so the project-wide lesson about endpoint preference still
  holds on this branch.
- But the branch as a whole is weak on the official protocol: even its final
  endpoint does not recover to the older `A03` baseline level, let alone the
  frozen `A_i03/M02` anchor.

## Comparison To Existing UHD Deblur Baselines

Frozen milestone before this run:

```text
A_i03 final.pt official full-image test PSNR = 25.85061787687282
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_i03 Final |
|---|---|---:|---:|
| `A03` | `final.pt` | 25.1402293880564 | -0.7104 |
| `A_g03` | `final.pt` | 25.3800849048404 | -0.4705 |
| `A_i03` | `final.pt` | 25.85061787687282 | 0.0000 |
| `A_j03` | `step_001000.pt` | 25.01813963817044 | -0.8325 |
| `A_j03` | `final.pt` | 25.196463325894427 | -0.6542 |

Additional deltas:

```text
A_j03 final.pt vs A_j03 step_001000.pt = +0.17832368772398704 dB
A_j03 final.pt vs A03 final.pt         = +0.05623393783802855 dB
A_j03 final.pt vs A_g03 final.pt       = -0.18362157894597252 dB
A_j03 final.pt vs A_i03 final.pt       = -0.6541545509783944 dB
```

Interpretation:

- This short-context-enlargement probe does **not** justify promoting the
  `864`-context setting directly into the next long run.
- Relative to the accepted `A_i03/M02` anchor, the short-budget result drops by
  about `0.65 dB`, which is far too large to treat as a merely noisy short-run
  fluctuation.
- The branch is still a useful negative result: it shows that reallocating
  memory toward a larger context under a short budget is not automatically a
  better next move than staying closer to `A_i03`.

## Main Conclusion

`V1/T01/A_j03` is accepted as a successfully completed short screening
experiment, but not as a baseline-replacement candidate and not as the new
preferred anchor for the next long-budget run.

It proves:

- the post-`A_i03` consistency line remains trainable even with a larger
  `864x864` training context under a short high-VRAM envelope;
- final endpoints still beat earlier internal-val-loss winners on the official
  test;
- simply reallocating more of the short-budget VRAM envelope toward a larger
  training context is **not** sufficient to match the accepted `A_i03/M02`
  baseline.

It does not prove:

- that larger context is useless in general under longer budgets;
- that `A_i03` is fully saturated.

What it does strongly suggest is:

- the next long-budget branch should stay much closer to the accepted
  `A_i03/M02` setting than to `A_j03`'s `864`-context short probe.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
queued deployment completion: passed
short-probe screening value: passed
official baseline replacement: no
preferred next long-run anchor: stay with A_i03/M02, not A_j03
```

## Recommended Next Action

The highest-value next step on the current run line is now:

1. record `A_j03` as a useful negative short probe rather than a winner;
2. keep `A_i03/M02` as the comparison anchor for the next long-budget branch;
3. if we extend the line further, prefer a longer-budget variant closer to
   `A_i03` rather than promoting `864` context directly from this short result.
