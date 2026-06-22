# Result Analysis: V1/T01/A01

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A01`
- Dataset: `UHD-LL_down4`
- Training split: 1900 pairs from `training_set`
- Internal validation split: 100 pairs from `training_set`
- Official test split: 150 pairs from `testing_set`

## Completion Status

- Training finished successfully.
- Final step: `4300`
- Final logged elapsed time: `59.3975` minutes
- Final checkpoint: `experiments/V1/runs/A01/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A01/checkpoints/step_004200.pt`
- Training log: `experiments/V1/runs/A01/logs/train_20260425_004245.log`
- Train metrics: `experiments/V1/runs/A01/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A01/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=4300`, not by the `wall_clock_minutes=62` cap. It landed slightly under 60 minutes. For future runs that must be strictly at least 1 hour, use a larger step budget such as `4500` or raise the wall-clock cap while keeping the same memory setting.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.4426628053188324
```

The final logged training loss was:

```text
step 4300 loss_total = 0.035297393798828125
```

Segment means:

| Step range | Mean train loss | Median | Min | Max |
|---|---:|---:|---:|---:|
| 1-500 | 0.064310 | 0.054046 | 0.038143 | 0.442663 |
| 501-1000 | 0.046167 | 0.045236 | 0.033931 | 0.058058 |
| 1001-2000 | 0.043693 | 0.042882 | 0.032557 | 0.063388 |
| 2001-3000 | 0.041631 | 0.041411 | 0.030552 | 0.060780 |
| 3001-4300 | 0.040303 | 0.040339 | 0.027001 | 0.057145 |

Interpretation:

- The model learned quickly in the first few hundred steps.
- After about `1000` steps, train loss continued to improve but at a slower rate.
- There was no observed NaN, OOM, or divergence.

## Training-Time Validation

Training-time validation used only `10` batches from the internal validation set, so it is useful for trend monitoring but not enough as the final selection metric by itself.

Best training-time validation point:

```text
step 1800
val_loss = 0.04320543296635151
val_psnr = 27.57744468782796
```

Last training-time validation point:

```text
step 4200
val_loss = 0.05001976136118173
val_psnr = 26.687948379945322
```

Interpretation:

- The small validation subset was noisy.
- Step `1800` looked best during training-time validation.
- Later checkpoints did not collapse, but validation fluctuated.

## Post-Hoc Full Evaluation

Because training-time validation used only 10 validation samples, a post-hoc CPU evaluation was run on:

- full internal validation set: 100 samples
- official test set: 150 samples

Evaluated checkpoints:

- `experiments/V1/runs/A01/checkpoints/step_001800.pt`
- `experiments/V1/runs/A01/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Loss mean | PSNR mean |
|---|---|---:|---:|---:|
| `step_001800.pt` | full internal val | 100 | 0.045259 | 27.3418 |
| `step_001800.pt` | official test | 150 | 0.063895 | 24.8208 |
| `final.pt` | full internal val | 100 | 0.045027 | 27.5009 |
| `final.pt` | official test | 150 | 0.056583 | 25.7391 |

Interpretation:

- The post-hoc full validation set favors `final.pt`, not `step_001800.pt`.
- The official test set also favors `final.pt`.
- Continuing from step `1800` to `4300` improved official test PSNR by about `0.9183 dB`.
- Official test loss improved from `0.063895` to `0.056583`.

## Main Conclusion

`V1/T01/A01` is a successful first deployment.

It proves:

- the `V1` model trains end-to-end on real paired image data;
- the `T01` training template works;
- full-image `UHD-LL_down4` training with `batch_size=28` fits the target memory range;
- checkpointing and metrics logging work;
- the final checkpoint improves both full internal validation and official test metrics compared with the early best training-time validation checkpoint.

The result is not yet a paper-quality model. It is a positive feasibility signal that justifies continuing to a better short-run or medium-run experiment.

## Recommended Next Action

Prepare `V1/T01/A02` as a second short run or short-to-medium run with:

- same split policy;
- same batch size `28`;
- full-image training;
- larger or full validation during training rather than only `10` validation samples;
- checkpoint selection based on full internal validation;
- slightly longer duration or explicit `4500+` step budget if the requirement is at least 1 hour.

The most important correction before A02 is validation protocol: A01's 10-sample training-time validation was too noisy for reliable checkpoint selection.
