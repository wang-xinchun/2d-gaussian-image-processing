# Gate Check: V1/T01/A_a03

Status: deployment gate passed; eligible for launch.

This file was first created as a design-only gate. On 2026-05-07 CST the user
explicitly requested deployment and reminded that the version number must be
handled carefully. The launch target is `V1/T01/A_a03`; it must not overwrite
`A03`.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock target: capped at `58` minutes.
- [x] Purpose: test whether a modest capacity and Gaussian-expression increase
  can improve over the accepted `A03` `UHD_deblur` baseline under the same data
  and evaluation protocol.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_a03`.
- [x] Full run identifier: `V1/T01/A_a03`.
- [x] Run folder: `experiments/V1/runs/A_a03`.
- [x] Exact training-template snapshot is configured to be saved under
  `template_snapshot/` at launch.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Dataset root: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur`.
- [x] Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`.
- [x] Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`.
- [x] Validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`.
- [x] Validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`.
- [x] Validation selection: first `100` paired samples from `train` in deterministic filename order.
- [x] Effective training selection: remaining `1864` paired samples from `train`.
- [x] Final test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/input300`.
- [x] Final test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300`.
- [x] Accepted inventory reference: `docs/unpacked_dataset_acceptance_20260425.md`.
- [x] Strategy reference: `docs/next_experiment_dataset_strategy_20260425.md`.
- [x] Metric-target reference: `docs/stage_metric_targets_20260505.md`.
- [x] Expected image counts: train input/gt `1964/1964`, official test image pairs `300/300`.
- [x] `test/gt300` contains extra non-image metadata files; the dataloader filters by image extension, so the effective test image count remains `300`.
- [x] Official `test` split is reserved for final post-training evaluation and is not used for training-time validation.
- [x] Image resolution target: accepted as `3840x2160`; current design uses `512x512` random training patches and `512x512` validation center crops.

## Contamination Check

- [x] Config source: `experiments/V1/templates/T01/configs/a_a03_short_uhddeblur_capacity.toml`.
- [x] Dataset paths intentionally match `A03` for same-protocol comparison.
- [x] Split policy intentionally matches `A03`.
- [x] Model architecture parameters intentionally differ from `A03`:
  `width=48`, `highres_blocks=3`, `lowres_blocks=6`, `decoder_blocks=3`,
  `affine_rank=6`, `gaussian_atoms=6`, `max_frequency=6.0`,
  `max_texture_gain=1.2`.
- [x] Loss terms remain unchanged from `A03`: Charbonnier weight `1.0`,
  gradient loss weight `0.05`.
- [x] Optimizer remains AdamW, lr `0.0002`, weight decay `0.0001`.
- [x] Evaluation protocol is explicit: RGB tensor range `[0,1]`, PSNR on
  clamped output, validation center crop.
- [x] Pretrained weight path: empty; current `A_a03` design is for fresh initialization.
- [x] Resume checkpoint path: empty; current `A_a03` design is not a resume.
- [x] Output directory is tied to the new run `experiments/V1/runs/A_a03`.
- [x] Random seed is explicit: `20260507`.

## Runtime Check

- [x] Batch size target: `16`.
- [x] Gradient accumulation: `1`.
- [x] Patch policy: `512x512` random training patches.
- [x] Max steps target: `2000`.
- [x] Planned duration cap: `58` minutes.
- [x] Validation interval target: every `250` steps.
- [x] Checkpoint interval target: every `250` steps.
- [x] Validation batches target: `100`.
- [x] DataLoader workers target: `16`.
- [x] Device plan: `auto`, CUDA if available.
- [x] AMP target: disabled for conservatism and comparability.
- [x] Data-inclusive smoke calibration on `UHD_deblur` has been run for the
  `A_a03` capacity setting.
- [x] Expected memory risk has been verified against the selected runtime target.
- [x] Expected end-to-end throughput has been verified against the selected
  runtime target.

Smoke calibration results:

```text
batch 16, workers 16, steps 5: ok, avg_total_step_seconds 2.2338,
fetch 1.7088, compute 0.5251, reserved 51.7598G, estimated_steps_60min 1611

batch 16, workers 16, steps 20: ok, avg_total_step_seconds 0.9747,
fetch 0.5232, compute 0.4514, reserved 51.7598G, estimated_steps_60min 3693
```

The 20-step smoke is treated as the more representative runtime estimate after
DataLoader workers are warm. The selected `max_steps=2000` and `58` minute cap
remain appropriate for a short run.

## Launch Decision

- [x] Gate file has been created.
- [x] Config file and run manifest have been prepared.
- [x] Syntax check of the new config has been run.
- [x] Data-inclusive smoke calibration has passed on `UHD_deblur`.
- [x] Formal preflight has passed on the final `A_a03` config.
- [x] Launch command and runtime owner have been written.
- [x] User approval for launch has been received after design; deployment was
  requested in the next turn.
- [x] `A_a03` is approved to launch.

Formal preflight result:

```text
preflight: ok
device: cuda
train_samples: 16
val_samples: 100
batch_input_shape: (16, 3, 512, 512)
batch_target_shape: (16, 3, 512, 512)
total_params: 189717
loss_total: 0.19018182158470154
loss_charbonnier: 0.18850398063659668
loss_gradient: 0.033556777983903885
val_batches: 100
val_loss: 0.10965484395623207
val_psnr: 17.758774463709766
```
