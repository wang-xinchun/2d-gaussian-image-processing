# Run Manifest: V1/T01/A01

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A01`
- Full identifier: `V1/T01/A01`

## Purpose

First short-training deployment for the `V1` architecture. The goal is to verify the trainability of the current model, data pipeline, logging, checkpointing, and validation loop on a known-good paired dataset before moving to medium or long training.

## Dataset

- Dataset: `UHD-LL_down4`
- Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/training_set/input`
- Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/training_set/gt`
- Internal validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/training_set/input`
- Internal validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/training_set/gt`
- Internal validation selection: first `100` paired samples from `training_set`
- Effective training selection: remaining `1900` paired samples from `training_set`
- Final test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/testing_set/input`
- Final test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/testing_set/gt`
- Dataset acceptance reference: `docs/dataset_organization_status_20260424.md`

## Planned Runtime

- Tier: short
- Duration limit: 62 minutes
- Max steps: 4300
- Batch size: 28
- Training crop: full `540x960` images (`train_crop_size=0`)
- Validation crop: 512

## Files

- Gate check: `gate_check.md`
- Launch config: `../../templates/T01/configs/a01_short_uhdll_down4.toml`
- Training script: `scripts/train_v1_t01.py`
- Archived template snapshot: `template_snapshot/`

## Preflight Result

Initial preflight command succeeded on 2026-04-24 11:35:49 CST, before the split-policy correction.

After 2026-04-24 20:04:03 CST, A01 uses an internal validation subset from `training_set` and reserves `testing_set` for final evaluation.

Corrected preflight command succeeded on 2026-04-24 20:04:03 CST.

```text
device: cuda
train_samples: 4
val_samples: 100
batch_input_shape: (2, 3, 256, 256)
batch_target_shape: (2, 3, 256, 256)
total_params: 70223
loss_total: 0.39183712005615234
loss_charbonnier: 0.39097803831100464
loss_gradient: 0.017181899398565292
val_batches: 10
val_loss: 0.35530078411102295
val_psnr: 8.285547616220203
effective_train_samples: 1900
internal_validation_samples: 100
final_test_samples: 150
train_val_overlap: 0
```

```text
device: cuda
train_samples: 4
val_samples: 20
batch_input_shape: (2, 3, 256, 256)
batch_target_shape: (2, 3, 256, 256)
total_params: 70223
loss_total: 0.39047640562057495
loss_charbonnier: 0.38959842920303345
loss_gradient: 0.017559509724378586
val_batches: 10
val_loss: 0.33160065710544584
val_psnr: 9.02392835011534
```

## Launch Command

```bash
python3 scripts/train_v1_t01.py --config experiments/V1/templates/T01/configs/a01_short_uhdll_down4.toml
```

## Launch Status

Launched on 2026-04-25 after the duration adjustment to `62` minutes / `4300` max steps.

Startup stability check on 2026-04-25 01:00:34 CST:

```text
main process: PID 49860
gpu_memory_used: 92315 MiB
gpu_utilization: 100%
current_observed_step: about 1170
current_observed_elapsed_minutes: about 16.18
recent_train_loss_total_range: about 0.035 to 0.052
latest_validation: step 1000, val_loss 0.05686534568667412, val_psnr 24.843871388781743
checkpoints_present: step_000200.pt, step_000400.pt, step_000600.pt, step_000800.pt, step_001000.pt, latest.pt
status: running and stable in startup window
```

Per user instruction, long continuous monitoring is not required after confirming the early deployment window is stable.

## Deployment Smoke Calibration

Deployment smoke was completed on 2026-04-25 00:37:09 CST.

Chosen first-deployment setting:

```text
batch_size: 28
train_crop_size: 0
max_steps: 4300
wall_clock_minutes: 62
val_interval: 200
checkpoint_interval: 200
```

Stable smoke result:

```text
batch_shape: (28, 3, 540, 960)
steps: 20
max_memory_reserved_gb: 89.453125
max_memory_allocated_gb: 84.5402135848999
avg_step_seconds: 0.8511253251694143
estimated_steps_55min: 3877
estimated_steps_60min: 4229
```

Launch adjustment:

```text
The user requested about 1 hour with duration no shorter than 1 hour if possible.
The deployment cap was adjusted from 58 minutes / 4000 steps to 62 minutes / 4300 steps before launch.
```

Formal configured preflight also succeeded:

```text
device: cuda
train_samples: 28
val_samples: 100
batch_input_shape: (28, 3, 540, 960)
batch_target_shape: (28, 3, 540, 960)
total_params: 70223
loss_total: 0.44990119338035583
loss_charbonnier: 0.4483032524585724
loss_gradient: 0.03195899724960327
val_batches: 10
val_loss: 0.354649218916893
val_psnr: 8.261483954411192
```

## Preflight Command

```bash
python3 scripts/train_v1_t01.py --config experiments/V1/templates/T01/configs/a01_short_uhdll_down4.toml --preflight
```

## Final Result Summary

Training finished successfully:

```text
final_step: 4300
final_elapsed_minutes: 59.39754991531372
final_checkpoint: experiments/V1/runs/A01/checkpoints/final.pt
latest_periodic_checkpoint: experiments/V1/runs/A01/checkpoints/step_004200.pt
```

Best training-time validation point, using the 10-batch validation loop:

```text
step: 1800
val_loss: 0.04320543296635151
val_psnr: 27.57744468782796
```

Post-hoc full evaluation:

```text
step_001800 full-val: loss 0.045258832015097145, psnr 27.34180851910698
step_001800 official-test: loss 0.06389497933288416, psnr 24.820797851889395
final full-val: loss 0.04502652919851244, psnr 27.500874817761318
final official-test: loss 0.05658330567181111, psnr 25.739053405072088
```

Result analysis document:

```text
experiments/V1/runs/A01/result_analysis.md
```
