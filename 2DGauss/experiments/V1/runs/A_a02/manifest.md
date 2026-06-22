# Run Manifest: V1/T01/A_a02

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_a02`
- Full identifier: `V1/T01/A_a02`

## Purpose

Data-loader-balanced derivative of the A02 `UHD_haze` short probe. The original `A02` startup window showed stable training but unacceptable end-to-end step time because the first smoke calibration measured GPU compute only and missed 4K JPEG decoding / DataLoader time. `A_a02` uses smaller batch size and more DataLoader workers so the run can reach validation and checkpoint milestones inside the short-training cap.

## Dataset

- Dataset: `UHD_haze`
- Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/input`
- Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/gt`
- Internal validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/input`
- Internal validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/gt`
- Internal validation selection: first `100` paired samples from `train`
- Effective training selection: remaining `2190` paired samples from `train`
- Final test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/test/input`
- Final test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/test/gt`
- Local final test口径: `231` pairs
- Dataset acceptance reference: `docs/unpacked_dataset_acceptance_20260425.md`
- Strategy reference: `docs/next_experiment_dataset_strategy_20260425.md`

## Planned Runtime

- Tier: short
- Duration limit: `58` minutes
- Max steps: `2000`
- Batch size: `16`
- DataLoader workers: `8`
- Training crop: `512x512` random patches
- Validation crop: `512x512` center crops
- Validation interval: `250`
- Checkpoint interval: `250`
- Validation batches: `100`

## Data-Inclusive Smoke Calibration

Smoke calibration with the updated data-inclusive timer:

```text
batch 16, workers 8: ok, avg_total_step_seconds 1.5391, fetch 1.2558, compute 0.2833, reserved 25.844G, estimated_steps_60min 2339
batch 24, workers 8: ok, avg_total_step_seconds 2.3162, fetch 1.8592, compute 0.4571, reserved 38.752G, estimated_steps_60min 1554
batch 32, workers 8: ok, avg_total_step_seconds 2.9822, fetch 2.4309, compute 0.5512, reserved 51.668G, estimated_steps_60min 1207
```

Chosen deployment setting:

```text
batch_size: 16
num_workers: 8
max_steps: 2000
wall_clock_minutes: 58
val_interval: 250
checkpoint_interval: 250
```

## Files

- Gate check: `gate_check.md`
- Launch config: `../../templates/T01/configs/a_a02_short_uhdhaze_patch_iobalanced.toml`
- Training script: `scripts/train_v1_t01.py`
- Archived template snapshot: `template_snapshot/`

## Preflight

Formal preflight succeeded with the final `A_a02` configuration:

```text
device: cuda
train_samples: 16
val_samples: 100
batch_input_shape: (16, 3, 512, 512)
batch_target_shape: (16, 3, 512, 512)
total_params: 70223
loss_total: 0.3110075294971466
loss_charbonnier: 0.3095058798789978
loss_gradient: 0.030032876878976822
val_batches: 100
val_loss: 0.2923806319385767
val_psnr: 9.867607934865733
```

## Launch Command

```bash
bash experiments/V1/runs/A_a02/launch_tmux.sh
```

## Launch Status

Launched in tmux session `2dgauss_a_a02_uhdhaze` on `2026-04-25 17:58:01 CST`.

Startup-window observation through the first validation/checkpoint:

```text
step 1: elapsed 0.1431 min, loss_total 0.23698
step 90: elapsed 1.3178 min, loss_total 0.17770
step 250: elapsed 3.7246 min, loss_total 0.11791
step 250 validation: val_batches 100, val_loss 0.131501, val_psnr 16.9850
step 400: elapsed 6.0608 min, loss_total 0.13523
gpu_memory_used: about 27173 MiB
checkpoints_present: step_000250.pt, latest.pt
```

Current status:

```text
tmux session: online
training process: running
train log: experiments/V1/runs/A_a02/logs/train_20260425_175801.log
status: stable in startup window; train/validation/checkpoint loop confirmed
```

## Final Runtime Status

`A_a02` finished by reaching `max_steps=2000`, not by the `58` minute wall-clock cap.

```text
final_step: 2000
final_elapsed_minutes: 30.84555850028992
final_train_loss_total: 0.12147729843854904
final_validation_step: 2000
final_val_batches: 100
final_val_loss: 0.09807305697351694
final_val_psnr: 19.960832028478485
final_checkpoint: experiments/V1/runs/A_a02/checkpoints/final.pt
latest_checkpoint: experiments/V1/runs/A_a02/checkpoints/latest.pt
last_periodic_checkpoint: experiments/V1/runs/A_a02/checkpoints/step_002000.pt
tmux_session_after_finish: closed
```

## Post-Hoc Evaluation Summary

Completed post-hoc records under the current local `UHD_haze` patch-based
official-test protocol (`crop_size=512`, local `231`-pair test split):

```text
step_001500.pt full-val: loss_mean=0.09423118511214852, psnr_mean=20.07013735128858
final.pt full-val: loss_mean=0.09807305697351694, psnr_mean=19.960832028478485
step_000500.pt official test: loss_mean=0.1316460735483087, psnr_mean=17.375494248965794
step_001500.pt official test: loss_mean=0.10324496093702007, psnr_mean=19.119746433850633
final.pt official test: loss_mean=0.10115578951257648, psnr_mean=19.444372918691123
```

Current checkpoint-selection conclusion:

```text
best internal-validation checkpoint: step_001500.pt
best official-test checkpoint: final.pt
official-test delta final.pt vs step_001500.pt: +0.32462648484048984 dB
```

Result analysis:

```text
experiments/V1/runs/A_a02/result_analysis.md
```

## Runtime Efficiency Note

User raised a concern that the GPU memory usage is too low. This is correct for
`A_a02`: the observed memory usage is about `27173 MiB`, much lower than the
original A02 batch-56 attempt. The reason is that A02 on full 4K JPEG sources is
currently data-loading bound. The original A02 used about `93295 MiB` but reached
only step `30` by `5.2063` minutes and was unlikely to reach its first planned
validation/checkpoint before the short-run cap.

`A_a02` therefore intentionally prioritizes a valid short-run loop over high
VRAM occupancy. It has already reached multiple validation/checkpoint points.
This should be treated as a temporary probe configuration, not as the final
efficient UHD-Haze training setup.

Next efficiency direction:

```text
Build a data-pipeline optimized derivative, such as a pre-cropped patch cache,
LMDB/WebDataset-style storage, or another loader that avoids repeatedly decoding
large 4K JPEGs before cropping. After that, recalibrate batch size to recover
high GPU memory utilization without losing validation/checkpoint throughput.
```
