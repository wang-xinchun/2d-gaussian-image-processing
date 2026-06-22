# Gate Check: V1/T01/A03

Status: deployment gate passed; eligible for launch.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock target: capped at `58` minutes.
- [x] Purpose: establish the first reproducible `UHD_deblur` baseline under `V1/T01` before any medium training or branch tuning.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A03`.
- [x] Full run identifier: `V1/T01/A03`.
- [x] Run folder: `experiments/V1/runs/A03`.
- [x] The exact training-template snapshot is configured to be saved under `template_snapshot/` at launch.

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

- [x] Config source: `experiments/V1/templates/T01/configs/a03_short_uhddeblur_patch.toml`.
- [x] Dataset paths point to `UHD_deblur`, not stale `UHD_haze` or `UHD-LL_down4`.
- [x] Model architecture parameters match current V1 defaults for comparability with `A01` and `A_a02`.
- [x] Loss terms remain conservative: Charbonnier weight `1.0`, gradient loss weight `0.05`.
- [x] Optimizer remains AdamW, lr `0.0002`, weight decay `0.0001`.
- [x] Evaluation protocol is explicit: RGB tensor range `[0,1]`, PSNR on clamped output, validation center crop.
- [x] Pretrained weight path: empty; current `A03` design is for fresh initialization.
- [x] Resume checkpoint path: empty; current `A03` design is not a resume.
- [x] Output directory is tied to the new run `experiments/V1/runs/A03`.
- [x] Random seed is explicit: `20260506`.

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
- [x] Data-inclusive smoke calibration on `UHD_deblur` has been run.
- [x] Expected memory risk has been verified against the selected runtime target.
- [x] Expected end-to-end throughput has been verified against the selected runtime target.

## Launch Decision

- [x] Gate file has been created.
- [x] Any future exception must be written into `manifest.md`.
- [x] Syntax check of the new config has been run.
- [x] Data-inclusive smoke calibration has passed on `UHD_deblur`.
- [x] Formal preflight has passed on the final `A03` config.
- [x] Launch command and runtime owner have been written.
- [x] User approval for launch has been requested after the above checks.
- [x] A03 is approved to launch.
