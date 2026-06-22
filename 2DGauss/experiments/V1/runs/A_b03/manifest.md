# Run Manifest: V1/T01/A_b03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_b03`
- Full identifier: `V1/T01/A_b03`

## Current Status

Deployment gate passed on `2026-05-07 23:08:19 CST`; launch is approved.

## Purpose

`A_b03` targets the current `UHD_deblur` problem exposed by `A03/A_a03`:
training-time checkpoint selection has used `512x512` validation crops, while
the official baseline is measured on full-image `crop_size=0` evaluation.

This run keeps the `A_a03` capacity/Gaussian-expression settings but changes
the training-time internal validation protocol to full-image validation. The
goal is not to create a new architecture, but to test whether a more aligned
validation signal gives better short-run checkpoint evidence.

## Baselines

Current official baseline:

```text
V1/T01/A03 final.pt official full-image test PSNR = 25.1402 dB
```

Latest capacity branch:

```text
V1/T01/A_a03 step_000500.pt full-val crop512 PSNR = 29.0861 dB
V1/T01/A_a03 final.pt diagnostic test crop512 PSNR = 24.0163 dB
```

`A_a03` official full-image test is still pending, so `A_b03` must not use the
official test split during training.

## Experimental Question

Can `V1/T01` get a more reliable checkpoint-selection signal on `UHD_deblur` by
using full-image internal validation during short training, while keeping the
same capacity setting as `A_a03`?

## Controlled Variables

Kept from `A_a03`:

- dataset: `UHD_deblur`;
- train split: `train/input_new -> train/gt_new`, offset `100`;
- model capacity: `width=48`, `highres_blocks=3`, `lowres_blocks=6`,
  `decoder_blocks=3`, `affine_rank=6`, `gaussian_atoms=6`,
  `max_frequency=6.0`, `max_texture_gain=1.2`;
- loss mix: Charbonnier `1.0` plus gradient `0.05`;
- optimizer: AdamW lr `0.0002`, weight decay `0.0001`;
- training patch: random `512x512`;
- short tier, `max_steps=2000`, batch size `16`;
- official test reservation.

Changed from `A_a03`:

- validation crop: `512` center crop -> full-image `crop_size=0`;
- validation size: first `100` pairs -> first `20` pairs to keep short-run
  validation cost controlled;
- validation interval: every `250` steps -> every `500` steps.

## Dataset Protocol

- Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`
- Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`
- Internal validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`
- Internal validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`
- Internal validation selection: first `20` train pairs
- Effective training selection: train pairs after first `100`
- Official test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/input300`
- Official test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300`
- Official test count: `300` image pairs

The official test split remains reserved for post-training evaluation.

## Runtime Envelope

```text
tier = short
wall_clock_minutes = 58
max_steps = 2000
batch_size = 16
num_workers = 8
train_crop_size = 512
val_crop_size = 0
val_limit = 20
val_interval = 500
checkpoint_interval = 250
max_val_batches = 20
amp = false
```

## Checkpoint Policy

During training, select candidates by full-image internal validation only.

After training:

1. identify the best full-image internal-validation checkpoint;
2. also evaluate `final.pt`;
3. run official full-image test post-hoc for the selected checkpoint and
   `final.pt`;
4. compare against `A03 final.pt` and pending `A_a03` official full-image
   results under the same `crop_size=0` protocol.

## Files

- Template config: `experiments/V1/templates/T01/configs/a_b03_short_uhddeblur_fullval.toml`
- Run-local config: `experiments/V1/runs/A_b03/configs/a_b03_short_uhddeblur_fullval.toml`
- Gate check: `experiments/V1/runs/A_b03/gate_check.md`
- Manifest: `experiments/V1/runs/A_b03/manifest.md`

## Required Before Launch

1. Syntax-check scripts and TOML identity.
2. Run data-inclusive smoke calibration on real `UHD_deblur` batches.
3. Run formal preflight with full-image validation.
4. Update gate and manifest with measured results.
5. Create and check launch script.
6. Start tmux only after all gate items pass.

## Gate Results

Syntax and identity checks passed:

```text
scripts/train_v1_t01.py: py_compile ok
scripts/smoke_calibrate_v1_t01.py: py_compile ok
scripts/evaluate_v1_t01.py: py_compile ok
template config full_id: V1/T01/A_b03
run-local config full_id: V1/T01/A_b03
val_crop_size: 0
val_limit: 20
val_interval: 500
max_val_batches: 20
launch_tmux.sh: bash -n ok
```

Dataset counts:

```text
train/input_new images: 1964
train/gt_new images: 1964
test/input300 images: 300
test/gt300 images: 300
```

Data-inclusive training smoke:

```text
batch 16, workers 8, crop 512, steps 20: ok
avg_total_step_seconds: 1.3369811685290187
avg_fetch_seconds: 0.8848452201578766
avg_compute_seconds: 0.4521338018588722
estimated_steps_55min: 2468
estimated_steps_60min: 2692
max_memory_allocated_gb: 50.373448848724365
max_memory_reserved_gb: 51.759765625
last_loss: 0.048051316291093826
```

Formal preflight with full-image internal validation:

```text
preflight: ok
device: cuda
train_samples: 16
val_samples: 20
batch_input_shape: (16, 3, 512, 512)
batch_target_shape: (16, 3, 512, 512)
total_params: 189717
loss_total: 0.19018182158470154
loss_charbonnier: 0.18850398063659668
loss_gradient: 0.033556777983903885
val_batches: 20
val_loss: 0.11466717235743999
val_psnr: 17.524048632373642
```

## Launch Command

```bash
bash experiments/V1/runs/A_b03/launch_tmux.sh
```

## Launch Status

Launched in tmux session `2dgauss_a_b03_uhddeblur_fullval` on
`2026-05-07 23:09:23 CST`.

```text
session: 2dgauss_a_b03_uhddeblur_fullval
train log: experiments/V1/runs/A_b03/logs/train_20260507_230923.log
config: experiments/V1/templates/T01/configs/a_b03_short_uhddeblur_fullval.toml
run_dir: experiments/V1/runs/A_b03
```

Startup-window observation through the first full-image validation:

```text
step 1: elapsed 0.1811 min, loss_total 0.19248
step 10: elapsed 0.3002 min, loss_total 0.06647
step 250: elapsed 4.7671 min, loss_total 0.02833
step 250 checkpoints: step_000250.pt, latest.pt
step 500: elapsed 9.3184 min, loss_total 0.03747
step 500 full-image validation: val_batches 20, val_loss 0.0213020, val_psnr 28.7915
step 500 checkpoint: step_000500.pt
latest observed train step: 530
tmux status: online
gpu_memory_used: about 62.8 GiB
```

Startup verdict:

```text
passed
```
