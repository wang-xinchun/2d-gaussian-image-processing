# Result Analysis: V1/T01/A_b02

## Run Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_b02`
- Dataset: `UHD_haze`
- Training split: cache-backed `17520` fixed patches derived from `2190` effective
  source train pairs
- Internal validation split: cache-backed `100` center-crop patches derived from
  the fixed held-out `100` train pairs
- Official test split: cache-backed `231` center-crop patches derived from the
  local `UHD_haze/test` split

## Completion Status

- Training finished successfully.
- Final step: `3200`
- Final logged elapsed time: `41.6782` minutes
- Final checkpoint: `experiments/V1/runs/A_b02/checkpoints/final.pt`
- Latest periodic checkpoint: `experiments/V1/runs/A_b02/checkpoints/step_003000.pt`
- Training log: `experiments/V1/runs/A_b02/logs/train_20260511_121007.log`
- Train metrics: `experiments/V1/runs/A_b02/metrics/train_metrics.jsonl`
- Validation metrics: `experiments/V1/runs/A_b02/metrics/val_metrics.jsonl`

The run completed by reaching `max_steps=3200`, not by the `58` minute
wall-clock cap. No NaN, no DataLoader stall, no checkpoint failure, and no
runtime abort occurred during the cache-backed deployment.

Important continuity note:

- `A_b02` is not a model-change branch.
- It keeps the same `V1/T01` model and loss mix as `A02/A_a02`.
- Its sole main experimental variable is the IO repair: replace repeated online
  4K JPEG decode-plus-crop with a deterministic offline patch cache.
- Before launch, `A_a02 final.pt` was re-evaluated against the cache-backed
  validation and official-test splits and reproduced the accepted existing
  values exactly, so the cache can be treated as an IO-path replacement rather
  than a protocol change.

## Cache Build Summary

Cache root:

```text
/root/autodl-tmp/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511
```

Build result:

```text
train cached pairs: 17520
val cached pairs: 100
test cached pairs: 231
train cache bytes: 8013721157
val cache bytes: 54896772
test cache bytes: 99658204
disk usage: about 7.7G
```

Protocol-equivalence check:

```text
A_a02 final.pt cache-backed full-val: loss_mean=0.09807305697351694, psnr_mean=19.960832028478485
A_a02 final.pt cache-backed official test: loss_mean=0.10115578951257648, psnr_mean=19.444372918691123
```

These numbers exactly match the accepted `A_a02` records, so later comparisons
between `A_a02` and `A_b02` remain same-protocol comparisons.

## Training Behavior

The first logged training loss was:

```text
step 1 loss_total = 0.27591672539711
```

The final logged training loss was:

```text
step 3200 loss_total = 0.087982177734375
```

Key runtime observation:

- original `A02` reached only `step=30` by `elapsed=5.2063` minutes and was
  stopped because validation/checkpoint milestones were unreachable under the
  short-tier cap;
- cache-backed `A_b02` reached `step=300` by `elapsed=3.9242` minutes while
  holding about `93.3 GiB` of GPU memory;
- it then continued cleanly to `step=3200` in `41.6782` minutes.

Interpretation:

- The IO repair worked as intended.
- `A_b02` is the first fair high-VRAM `UHD_haze` short run under the current
  local patch protocol.
- The engineering question behind `A02` is now resolved: the previous failure
  was indeed a data-pipeline failure, not a memory-fit or model-fit failure.

## Training-Time Validation

Training-time validation used the cache-backed `100`-pair held-out train slice
with `crop_size=512`.

Validation trajectory:

| Step | Val loss | Val PSNR |
|---|---:|---:|
| 500 | 0.107602 | 18.6372 |
| 1000 | 0.097742 | 19.8660 |
| 1500 | 0.106415 | 18.8808 |
| 2000 | 0.075071 | 21.6290 |
| 2500 | 0.075165 | 21.8647 |
| 3000 | 0.086566 | 20.7548 |

Checkpoint-selection facts:

```text
best validation PSNR checkpoint: step_002500.pt, PSNR=21.864663560497057
lowest validation loss checkpoint: step_002000.pt, loss=0.0750709313340485
```

Interpretation:

- Relative to `A_a02`, the cache-backed high-VRAM branch substantially improves
  internal validation PSNR.
- The run crosses `21.5 dB` on internal validation and peaks near `21.86 dB`,
  well above the earlier `A_a02` internal-val reference `20.0701 dB`.
- So the IO repair unlocks stronger optimization behavior on the internal
  validation slice.

## Post-Hoc Evaluation

Post-hoc evaluation is complete for:

- full internal validation set: `100` samples, `crop_size=512`
- local official test set: `231` samples, `crop_size=512`

Evaluated checkpoints:

- `experiments/V1/runs/A_b02/checkpoints/step_002000.pt`
- `experiments/V1/runs/A_b02/checkpoints/step_002500.pt`
- `experiments/V1/runs/A_b02/checkpoints/final.pt`

Results:

| Checkpoint | Split | Samples | Crop | Loss mean | PSNR mean |
|---|---|---:|---:|---:|---:|
| `step_002000.pt` | full internal val | 100 | 512 | 0.075071 | 21.6290 |
| `step_002000.pt` | official test | 231 | 512 | 0.104952 | 18.9624 |
| `step_002500.pt` | full internal val | 100 | 512 | 0.075165 | 21.8647 |
| `step_002500.pt` | official test | 231 | 512 | 0.102023 | 19.3968 |
| `final.pt` | full internal val | 100 | 512 | 0.086377 | 20.7964 |
| `final.pt` | official test | 231 | 512 | 0.100904 | 19.4395 |

Interpretation:

- Internal validation strongly prefers the late mid-run checkpoints
  `step_002000.pt` and `step_002500.pt`.
- Official test still prefers the final checkpoint, but only narrowly:

```text
final.pt vs step_002500.pt official test = +0.042684653433173736 dB
step_002500.pt vs step_002000.pt official test = +0.4344248770888193 dB
```

- This means the same checkpoint-selection tension still exists on the haze
  line even after the IO fix:
  the internal-val best checkpoint is not automatically the official-test
  winner.

## Comparison To Existing UHD Haze Reference

Accepted same-protocol reference before this run:

```text
A_a02 final.pt official test PSNR = 19.444372918691123
```

Same-task same-protocol comparison:

| Run | Checkpoint | Official Test PSNR | Delta vs A_a02 Final |
|---|---|---:|---:|
| `A_a02` | `final.pt` | 19.444372918691123 | 0.0000 |
| `A_b02` | `step_002000.pt` | 18.962357169960864 | -0.4820 |
| `A_b02` | `step_002500.pt` | 19.396782047049683 | -0.0476 |
| `A_b02` | `final.pt` | 19.439466700482857 | -0.0049 |

Interpretation:

- `A_b02` completely solves the engineering throughput problem and materially
  improves internal validation quality, but it does **not** produce a meaningful
  same-protocol official-test gain over `A_a02`.
- The best official-test checkpoint is `A_b02 final.pt`, but it remains about
  `-0.0049 dB` below `A_a02 final.pt`, which is effectively flat and certainly
  below the project's meaningful-gain threshold.
- So the IO repair is necessary, but not sufficient by itself to move the
  `UHD_haze` official-test result into the next stage.

## Main Conclusion

`V1/T01/A_b02` is accepted as a successful completed high-VRAM `UHD_haze`
experiment and as a successful engineering fix for the task's confirmed
throughput bottleneck. It is **not** accepted as a new official `UHD_haze`
result baseline replacing `A_a02 final.pt`.

It proves:

- the `UHD_haze` 4K JPEG / DataLoader bottleneck can be resolved cleanly with a
  deterministic patch cache while preserving the current local official-test
  protocol;
- the original `A02` high-VRAM envelope becomes genuinely trainable once the IO
  path is repaired;
- cache-backed training can sustain `>=80G` VRAM and complete a full short-tier
  train/validation/checkpoint loop cleanly.

It does not yet prove:

- that IO repair alone lifts the official `UHD_haze` result above the existing
  local reference `A_a02 final.pt`;
- that the haze line has entered the staged stronger-goal region around
  `20.5 dB` official test;
- that the remaining gap on this task is still dominated by engineering rather
  than by modeling or protocol limitations.

The conservative project record is therefore:

> `A_b02` completed cleanly at `step=3200` as the first cache-backed high-VRAM
> `UHD_haze` short run, resolved the original `A02` throughput failure, pushed
> internal validation as high as `21.8647 dB`, but did not improve the current
> same-protocol official-test reference beyond `A_a02 final.pt PSNR=19.4444 dB`.

## Acceptance Decision

Acceptance status:

```text
runtime/checkpoint acceptance: passed
cache-backed IO-fix acceptance: passed
high-VRAM gate acceptance: passed
official test evaluation: completed
official UHD_haze baseline replacement: no
current same-protocol UHD_haze reference remains: V1/T01/A_a02 final.pt, PSNR=19.4444 dB
```

## Recommended Next Action

The highest-value next step on the `UHD_haze` line is no longer “fix the IO”
because that step is now complete. The next question should move upward in the
stack:

1. keep the cache-backed data path as the default `UHD_haze` training substrate
   for future high-VRAM runs;
2. preserve the experimental lesson that internal validation improved strongly
   while official test stayed flat, so future branch selection still cannot rely
   on internal-val PSNR alone;
3. design the next haze branch around actual modeling or train-test-consistency
   changes rather than another pure IO replay.
