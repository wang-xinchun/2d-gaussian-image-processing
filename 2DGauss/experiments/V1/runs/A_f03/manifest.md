# Run Manifest: V1/T01/A_f03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_f03`
- Full identifier: `V1/T01/A_f03`

## Current Status

Training completed and formally accepted.

## Purpose

`A_f03` is the next `UHD_deblur` branch after the accepted `A_e03` official
baseline.

`A_e03` established that the edge-guided cache-safe branch is both stable and
slightly better than the old `A03` baseline:

```text
A_e03 final.pt official full-image test PSNR = 25.21410079036176
delta vs A03 final.pt = +0.07387140230535955
```

However, `A_e03` stopped because it reached `max_steps=1500`, not because it
hit the short-tier wall-clock cap. It finished in only `27.6250` minutes and
its full-image internal validation PSNR improved again at the final checkpoint.

`A_f03` therefore keeps the accepted `A_e03` mechanism and runtime hygiene
fixed, and changes only one main variable:

```text
max_steps: 1500 -> 2500
```

The goal is to test whether the edge-guided branch was under-trained inside the
same short-tier budget.

## Protocol

Kept from `A_e03`:

- dataset: `UHD_deblur`;
- training crop: random `512x512`;
- internal validation: full-image `crop_size=0`;
- validation limit: first `20` train pairs;
- validation interval: `500`;
- checkpoint interval: `250`;
- batch size: `20`;
- `>=80G` VRAM gate;
- edge-guided detail refiner and Laplacian consistency loss;
- cache-safe launch discipline.

Changed from `A_e03`:

- `max_steps=1500` -> `max_steps=2500`;
- `seed=20260509` -> `seed=20260511`.

## Files

- Template config: `experiments/V1/templates/T01/configs/a_f03_short_uhddeblur_edge_guided_extended_budget.toml`
- Run-local config: `experiments/V1/runs/A_f03/configs/a_f03_short_uhddeblur_edge_guided_extended_budget.toml`
- Gate check: `experiments/V1/runs/A_f03/gate_check.md`
- Manifest: `experiments/V1/runs/A_f03/manifest.md`

## Required Before Launch

1. Launch with the template config path, not the run-local config path.
2. Observe the startup window and confirm that the extended-budget branch stays
   healthy past the first checkpoint and toward the first validation point.

## Launch Authorization

This turn is authorized by the user's deployment request plus the explicit
instruction to run both lines without waiting for further confirmation.

## Gate Completion

Static checks passed:

```text
scripts/train_v1_t01.py: py_compile ok
scripts/smoke_calibrate_v1_t01.py: py_compile ok
scripts/evaluate_v1_t01.py: py_compile ok
template config full_id: V1/T01/A_f03
run-local config full_id: V1/T01/A_f03
batch_size: 20
max_steps: 2500
val_crop_size: 0
launch_tmux.sh: bash -n ok
```

Data count check:

```text
train/input_new images: 1964
train/gt_new images: 1964
test/input300 images: 300
test/gt300 images: 300
```

Smoke calibration passed under the actual launch environment:

```text
batch_size: 20
steps: 20
avg_total_step_seconds: 1.3712702706456183
estimated_steps_55min: 2406
estimated_steps_60min: 2625
max_memory_reserved_gb: 94.126953125
```

Formal preflight passed:

```text
device: cuda
train_samples: 20
val_samples: 20
total_params: 231528
loss_total: 0.11356815695762634
loss_laplacian: 0.02811802737414837
val_loss: 0.07758434377610683
val_psnr: 20.31078866964017
```

## Launch Status

```text
session: 2dgauss_a_f03_uhddeblur_edge_guided_step2500
config: experiments/V1/templates/T01/configs/a_f03_short_uhddeblur_edge_guided_extended_budget.toml
log: experiments/V1/runs/A_f03/logs/train_20260511_022049.log
launch_env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Startup-window observations:

```text
step 1 loss_total=0.1244623
step 10 loss_total=0.0545635
step 20 loss_total=0.0510086
step 250 checkpoint written: step_000250.pt and latest.pt
step 500 full-image validation: val_loss=0.0228021, val_psnr=28.5946
step 500 checkpoint written: step_000500.pt and latest.pt
latest observed train step: 550
latest observed train elapsed_minutes: 10.5659
latest observed train loss_total: 0.0312628
post-validation continuation: confirmed
observed GPU memory: about 97.1GB used
status: startup window passed; training continuing
```

## Final Runtime Status

`A_f03` finished by reaching `max_steps=2500`.

```text
final_step: 2500
final_elapsed_minutes: 47.64691624244054
final_train_loss_total: 0.04098471254110336
final_validation_step: 2500
final_val_batches: 20
final_val_loss: 0.02363556819036603
final_val_psnr: 28.48022637157565
tmux_session_after_finish: closed
training_process_after_finish: none
gpu_memory_after_finish: 0 MiB used
```

## Acceptance Evaluation Summary

Training/runtime acceptance passed. `A_f03` preserved the `>=80G` deployment
gate, crossed the historical `A_d03` post-validation failure point, and
finished the full `2500`-step extended-budget run without OOM, NaN, checkpoint
write failure, or data-loader failure.

Post-hoc full-image evaluation:

```text
step_000500.pt full-val: loss_mean=0.02280208421871066, psnr_mean=28.594590010634533
step_000500.pt official test: loss_mean=0.03539453802475085, psnr_mean=25.088809336119052
final.pt full-val: loss_mean=0.02363556819036603, psnr_mean=28.48022637157565
final.pt official test: loss_mean=0.03455845245160163, psnr_mean=25.314470701891672
```

Comparison to the previous official baseline:

```text
A_e03 final.pt official full-image test PSNR = 25.21410079036176
A_f03 final.pt official full-image test PSNR = 25.314470701891672
delta = +0.10036991152991348
```

Additional continuity reference:

```text
A03 final.pt official full-image test PSNR = 25.1402293880564
delta vs A03 = +0.17424131383527097
```

Technical note:

```text
step_002500.pt and final.pt have identical model weights; the final official result is therefore also the effective step-2500 result.
```

Conclusion:

```text
A_f03 is accepted as a completed high-VRAM extended-budget edge-guided experiment and becomes the new official UHD_deblur baseline, although the gain over A_e03 remains modest rather than a staged meaningful jump.
```

Result analysis:

```text
experiments/V1/runs/A_f03/result_analysis.md
```
