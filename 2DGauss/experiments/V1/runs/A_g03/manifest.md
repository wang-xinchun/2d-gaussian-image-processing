# Run Manifest: V1/T01/A_g03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_g03`
- Full identifier: `V1/T01/A_g03`

## Current Status

Training completed and formally accepted.

## Purpose

`A_g03` is the first post-`A_f03` `UHD_deblur` branch design that changes model
behavior instead of only extending budget.

`A_f03` proved three things:

1. the accepted edge-guided cache-safe branch still benefits from longer
   short-tier training;
2. the gain appears on official full-image test rather than on the small
   `20`-image internal full-image validation slice;
3. the resulting improvement is real but modest (`+0.1004 dB` over `A_e03`),
   so simply extending budget again is not the highest-value next move.

Therefore `A_g03` targets the next bottleneck identified in the project
strategy: stronger directional/frequency structure for `UHD_deblur`.

## New Mechanism

The new code keeps the current `A_f03` mainline intact:

- low-resolution compact modeling;
- Gaussian parameter prediction;
- Gaussian-rendered high-resolution modulation;
- high-resolution edge-guided detail refinement.

On top of that, it adds a default-off directional-frequency guidance path:

- derive low-resolution orientation/frequency priors from the already predicted
  Gaussian parameters;
- summarize dominant direction, frequency intensity, texture gain, coherence,
  and spectral decay into a compact multi-channel prior;
- upsample that prior to high resolution;
- inject it into the edge-guided refiner so high-resolution detail recovery is
  no longer conditioned only on generic edge magnitude and scalar attention.

This is intentionally aligned with the project strategy rather than a backbone
replacement:

- `Stripformer`-style insight: blur is directional;
- `FFTformer`-style insight: restoration depends on selecting which frequencies
  to amplify or suppress;
- project constraint: keep the `lowres -> gaussian -> highres` story as the
  main contribution.

## Why This Branch After A_f03

`A_f03` already answers the obvious budget question. It does not answer whether
the current deblur ceiling comes from weak directional/frequency guidance inside
the high-resolution refinement path.

`A_g03` is designed to answer that question cleanly by changing only one
high-level idea:

> Should the edge-guided high-resolution branch also receive structured
> direction/frequency priors distilled from Gaussian parameters?

This is more targeted than:

- another capacity-only replay;
- another longer-budget replay;
- replacing the backbone with a generic deblurring architecture.

## Planned Protocol

Kept from `A_f03`:

- dataset: `UHD_deblur`;
- train split: after the first `100` train pairs;
- internal validation: first `20` train pairs;
- official test: `300` full-image pairs, post-hoc only;
- training crop: random `512x512`;
- internal validation crop: full image, `crop_size=0`;
- batch target: `20`;
- high-VRAM requirement: keep `>=80G` at deployment time;
- cache-safe runtime discipline;
- edge-guided refiner and Laplacian loss;
- short-tier wall-clock cap `58` minutes.

Changed from `A_f03`:

- `use_directional_frequency_guidance = true`
- calibrated `batch_size = 18` instead of the inherited design value `20`

Not changed:

- no scheduler changes;
- no dataset changes;
- no official-test peeking.

## Files

- Template config: `experiments/V1/templates/T01/configs/a_g03_short_uhddeblur_directional_frequency.toml`
- Run-local config: `experiments/V1/runs/A_g03/configs/a_g03_short_uhddeblur_directional_frequency.toml`
- Gate check: `experiments/V1/runs/A_g03/gate_check.md`
- Manifest: `experiments/V1/runs/A_g03/manifest.md`
- Launch script: `experiments/V1/runs/A_g03/launch_tmux.sh`

## Gate Completion

Static checks:

```text
scripts/train_v1_t01.py: py_compile ok
scripts/smoke_calibrate_v1_t01.py: py_compile ok
scripts/evaluate_v1_t01.py: py_compile ok
template config full_id: V1/T01/A_g03
run-local config full_id: V1/T01/A_g03
use_edge_guidance: true
use_directional_frequency_guidance: true
batch_size: 18
launch_tmux.sh: bash -n ok
CPU compatibility check: accepted old path still runs with directional_prior=None
```

Calibration summary:

```text
batch_size=19: oom, max_memory_reserved_gb=94.1035
batch_size=18: ok, max_memory_reserved_gb=93.0137
avg_total_step_seconds=1.2536
estimated_steps_60min=2871
```

Formal preflight:

```text
preflight: ok
device: cuda
train_samples: 18
val_samples: 20
batch_input_shape: (18, 3, 512, 512)
batch_target_shape: (18, 3, 512, 512)
total_params: 246888
loss_total: 0.2592295
val_loss: 0.1859989
val_psnr: 13.6522
```

## Working Hypothesis

If `A_g03` helps, the official full-image test should improve because the
high-resolution detail branch receives more task-specific directional/frequency
guidance than the current scalar-attention-only modulation path.

If it does not help, that is still useful evidence that the current deblur
ceiling is more likely dominated by train-inference consistency or by a weaker
upstream low-resolution representation issue rather than by missing direction
structure inside the edge refiner.

## Launch Approval

```text
approved_at: 2026-05-13 10:43 UTC
authorization_basis: user requested deployment after the model-adjustment turn
```

## Launch Status

```text
session: 2dgauss_a_g03_uhddeblur_directional_frequency
config: experiments/V1/templates/T01/configs/a_g03_short_uhddeblur_directional_frequency.toml
log: experiments/V1/runs/A_g03/logs/train_20260513_132256.log
launch_env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Startup-window observations:

```text
step 1 loss_total=0.2359021
step 110 loss_total=0.0451925
step 250 checkpoint written: step_000250.pt and latest.pt
step 500 full-image validation: val_loss=0.0224994, val_psnr=28.6726
post-validation continuation confirmed
status: startup window passed; training later finished cleanly at step 2500
```

## Final Runtime Status

`A_g03` finished by reaching `max_steps=2500`.

```text
final_step: 2500
final_elapsed_minutes: 43.62825961907705
final_train_loss_total: 0.03360201418399811
best_training_time_val_step: 1500
best_training_time_val_loss: 0.02209489792585373
best_training_time_val_psnr: 28.74008243547481
final_validation_step: 2500
final_val_batches: 20
final_val_loss: 0.02299201423302293
final_val_psnr: 28.62097594450625
tmux_session_after_finish: closed
training_process_after_finish: none
gpu_memory_after_finish: 0 MiB used
```

## Acceptance Evaluation Summary

Training/runtime acceptance passed. `A_g03` preserved the `>=80G` deployment
gate after recalibration to `batch_size=18`, crossed all validation/checkpoint
milestones cleanly, and finished the full `2500`-step short-tier run without
OOM, NaN, checkpoint write failure, or data-loader failure.

Post-hoc full-image evaluation:

```text
step_001500.pt full-val: loss_mean=0.02209489792585373, psnr_mean=28.74008243547481
step_001500.pt official test: loss_mean=0.03513034270765881, psnr_mean=25.204020174253888
final.pt full-val: loss_mean=0.02299201423302293, psnr_mean=28.62097594450625
final.pt official test: loss_mean=0.034169728457927706, psnr_mean=25.3800849048404
```

Comparison to the previous official baseline:

```text
A_f03 final.pt official full-image test PSNR = 25.314470701891672
A_g03 final.pt official full-image test PSNR = 25.3800849048404
delta = +0.06561420294872633
```

Additional continuity reference:

```text
A03 final.pt official full-image test PSNR = 25.1402293880564
delta vs A03 = +0.2398555167839973
```

Technical note:

```text
step_002500.pt and final.pt have identical model weights; the accepted official result is therefore also the effective step-2500 result.
```

Conclusion:

```text
A_g03 is accepted as a completed high-VRAM directional-frequency UHD_deblur experiment and becomes the new official UHD_deblur baseline, although the gain over A_f03 remains modest.
```

Result analysis:

```text
experiments/V1/runs/A_g03/result_analysis.md
```
