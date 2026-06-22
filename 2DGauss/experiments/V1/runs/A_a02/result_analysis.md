# Result Analysis: V1/T01/A_a02

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_a02`
- Dataset: `UHD_haze`
- Training split: `2190` pairs from `train`
- Internal validation split: `100` pairs from `train`
- Official test split: local `231` pairs from `test`

## Completion Status

- Training finished successfully.
- Final step: `2000`
- Final logged elapsed time: `30.8456` minutes
- Final checkpoint: `experiments/V1/runs/A_a02/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_a02/checkpoints/step_002000.pt`
- Training log: `experiments/V1/runs/A_a02/logs/train_20260425_175801.log`
- Train metrics: `experiments/V1/runs/A_a02/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_a02/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=2000`, not by the
`wall_clock_minutes=58` cap. It therefore established a valid short-run loop
under the IO-balanced configuration rather than merely surviving the startup
window.

Important continuity note:

- `A_a02` is the data-loader-balanced derivative of the aborted `A02`
  deployment.
- `A02` was not a model-failure case; it was stopped because the original
  high-batch configuration could not reach meaningful validation/checkpoint
  milestones inside the short-run cap under the current 4K JPEG pipeline.
- `A_a02` changes runtime balance, not task identity: it keeps the same
  `UHD_haze` single-task objective while reducing batch size enough to recover a
  usable train/validation/checkpoint loop.

## Evaluation Protocol Note

This run uses a patch-based `UHD_haze` protocol rather than the later
`UHD_deblur` full-image official-test protocol.

Current local official-test record for `A_a02` therefore means:

- split: local `UHD_haze/test` with `231` pairs
- crop size: `512`
- evaluation mode: center-crop patch evaluation on the held-out official test
  split

This is the protocol already implied by the run gate and existing post-hoc test
artifacts. It should not be confused with the full-image `crop_size=0`
evaluation later adopted as the official comparison protocol on the deblurring
line.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.23698098957538605
```

The final logged training loss was:

```text
step 2000 loss_total = 0.12147729843854904
```

Segment means:

| Step range | Mean train loss | Median | Min | Max |
|---|---:|---:|---:|---:|
| 1-250 | 0.156135 | 0.147293 | 0.117912 | 0.236981 |
| 251-500 | 0.144088 | 0.143568 | 0.111579 | 0.184188 |
| 501-1000 | 0.128322 | 0.126060 | 0.095134 | 0.172797 |
| 1001-1500 | 0.125982 | 0.124700 | 0.079453 | 0.181013 |
| 1501-2000 | 0.113388 | 0.111416 | 0.085369 | 0.181406 |

Interpretation:

- The run shows a clear downward training trend across the full short budget.
- There is no sign of NaN, divergence, OOM, or checkpoint failure after the
  deployment was corrected from `A02` to `A_a02`.
- The model is trainable on `UHD_haze`; the dominant issue on this task is the
  input pipeline, not an inability to optimize at all.

## Training-Time Validation

Training-time validation used `100` held-out train pairs with `crop_size=512`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 250 | 0.131501 | 16.9850 |
| 500 | 0.118170 | 18.3842 |
| 750 | 0.127005 | 17.3555 |
| 1000 | 0.095784 | 19.7586 |
| 1250 | 0.110673 | 18.8070 |
| 1500 | 0.094231 | 20.0701 |
| 1750 | 0.095190 | 19.9418 |
| 2000 | 0.098073 | 19.9608 |

The best internal-validation point is:

```text
step 1500
val_loss = 0.09423118511214852
val_psnr = 20.07013735128858
```

Interpretation:

- `A_a02` crosses the stage target of pushing stable validation PSNR to about
  `20 dB`.
- Validation is not perfectly monotonic, but the best point appears late rather
  than in the first few hundred steps.
- This is useful evidence that the branch is learning meaningful haze-removal
  behavior under the IO-balanced deployment.

## Post-Hoc Evaluation

Post-hoc evaluation is now complete for:

- full internal validation set: `100` samples, `crop_size=512`
- local official test set: `231` samples, `crop_size=512`

Evaluated checkpoints:

- `experiments/V1/runs/A_a02/checkpoints/step_000500.pt`
- `experiments/V1/runs/A_a02/checkpoints/step_001500.pt`
- `experiments/V1/runs/A_a02/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Crop | Loss mean | PSNR mean |
|---|---|---:|---:|---:|---:|
| `step_001500.pt` | full internal val | 100 | 512 | 0.094231 | 20.0701 |
| `final.pt` | full internal val | 100 | 512 | 0.098073 | 19.9608 |
| `step_000500.pt` | official test | 231 | 512 | 0.131646 | 17.3755 |
| `step_001500.pt` | official test | 231 | 512 | 0.103245 | 19.1197 |
| `final.pt` | official test | 231 | 512 | 0.101156 | 19.4444 |

Interpretation:

- The internal-validation-selected checkpoint is `step_001500.pt`.
- The official test split still prefers `final.pt` over the internal-val winner.
- The gap is meaningful rather than negligible:

```text
step_001500.pt vs step_000500.pt official test = +1.7443 dB
final.pt vs step_000500.pt official test      = +2.0689 dB
final.pt vs step_001500.pt official test      = +0.3246 dB
```

- So this run already shows a smaller version of the checkpoint-selection
  tension later seen more clearly on the `UHD_deblur` line: the best internal
  validation checkpoint is not automatically the official-test winner.

## Main Conclusion

`V1/T01/A_a02` is accepted as a successful completed `UHD_haze` short probe and
as the first usable official-test reference point on this task under the
current local patch-based protocol.

It proves:

- `V1/T01` transfers beyond low-light enhancement and can learn on real
  `UHD_haze` paired restoration data;
- the original `A02` failure was primarily a throughput/configuration problem,
  not a proof that the model cannot train on `UHD_haze`;
- the IO-balanced derivative reaches a complete short-run loop and produces a
  positive official-test signal.

It does not yet prove:

- that the current `UHD_haze` configuration is efficient, since GPU occupancy is
  intentionally low relative to the earlier high-batch attempt;
- that the current patch-based `UHD_haze` line is already strong enough for a
  paper-level headline result;
- that the present task bottleneck is solved, because the data pipeline remains
  the main blocker to scaling this line further.

The conservative project record is therefore:

> `A_a02` completed cleanly at `step=2000` as an IO-balanced recovery from the
> throughput-invalid `A02` deployment, reached about `20.07 dB` on internal
> validation, and established a first local `UHD_haze` official-test reference
> of `19.4444 dB` (`final.pt`) under the current `231`-pair `crop_size=512`
> patch-evaluation protocol.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
cross-task trainability signal: passed
official test evaluation: completed
current UHD_haze result class: valid probe / early official reference
paper-level strength: not yet
```

## Recommended Next Action

The highest-value next step on the `UHD_haze` line is not a blind replay of the
same configuration. It is:

1. treat `A_a02 final.pt PSNR=19.4444` as the current local reference on the
   existing patch-based protocol;
2. prioritize data-pipeline optimization first, such as patch cache,
   LMDB/WebDataset-style storage, or another way to avoid repeated 4K JPEG
   decode before cropping;
3. after the pipeline is fixed, recalibrate batch size to recover high GPU
   utilization and then test whether `UHD_haze` official test PSNR can move past
   the staged `20.5 dB` region.
