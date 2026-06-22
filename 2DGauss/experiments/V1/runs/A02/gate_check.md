# Gate Check: V1/T01/A02

Status: launched, then stopped during startup window due to data-loading throughput bottleneck.

## Training Duration Tier

- [x] Tier: short training.
- [x] Planned wall-clock duration: capped at `58` minutes.
- [x] Purpose: test whether the existing `V1/T01` paired-restoration template transfers from low-light enhancement to the newly accepted `UHD_haze` UHD dehazing dataset.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A02`.
- [x] Full run identifier: `V1/T01/A02`.
- [x] Run folder: `experiments/V1/runs/A02`.
- [x] The exact training-template snapshot is configured to be saved under `template_snapshot/` at launch.

## Dataset Check

- [x] Dataset name: `UHD_haze`.
- [x] Dataset root: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze`.
- [x] Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/input`.
- [x] Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/gt`.
- [x] Validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/input`.
- [x] Validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/train/gt`.
- [x] Validation selection: first `100` paired samples from `train` in deterministic filename order.
- [x] Effective training selection: remaining `2190` paired samples from `train`.
- [x] Final test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/test/input`.
- [x] Final test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_haze/test/gt`.
- [x] Accepted inventory reference: `docs/unpacked_dataset_acceptance_20260425.md`.
- [x] Strategy reference: `docs/next_experiment_dataset_strategy_20260425.md`.
- [x] Expected source counts: train input/gt `2290/2290`, local test input/gt `231/231`.
- [x] Local `UHD_haze` official test口径 is `231` pairs, not the paper inventory's `230`.
- [x] Official `test` split is not used for training-time validation; it is reserved for final post-training evaluation.
- [x] Image resolution: accepted as `3840x2160`; A02 uses `512x512` random training patches and `512x512` validation center crops.

## Contamination Check

- [x] Config source: `experiments/V1/templates/T01/configs/a02_short_uhdhaze_patch.toml`.
- [x] Dataset paths point to `UHD_haze`, not stale `UHD-LL_down4`.
- [x] Model architecture parameters match current V1 defaults.
- [x] Loss terms: Charbonnier weight `1.0`, gradient loss weight `0.05`.
- [x] Optimizer: AdamW, lr `0.0002`, weight decay `0.0001`.
- [x] Evaluation protocol: RGB tensor range `[0,1]`, PSNR on clamped output, validation center crop.
- [x] Pretrained weight path: empty; A02 starts from random initialization.
- [x] Resume checkpoint path: empty; A02 is a fresh short run.
- [x] Output directory is tied to the new run `experiments/V1/runs/A02`.
- [x] Random seed: `20260425`.

## Runtime Check

- [x] Batch size: `56`.
- [x] Gradient accumulation: `1`.
- [x] Patch policy: `512x512` random training patches.
- [x] Max steps: `3400`.
- [x] Planned duration cap: `58` minutes.
- [x] Validation interval: every `500` steps.
- [x] Checkpoint interval: every `500` steps.
- [x] Device plan: `auto`, CUDA if available.
- [x] AMP: disabled for A02 conservatism.
- [x] Expected memory: smoke-calibrated peak reserved memory about `90.418G` at batch size `56`.

## Launch Decision

- [x] Gate file created.
- [x] Syntax checks passed.
- [x] Formal preflight passed.
- [x] Deployment smoke calibration passed.
- [x] Final batch size and max steps written back to config and gate.
- [x] A02 was eligible for short-training launch after final gate completion.
- [x] A02 was launched.
- [x] A02 was intentionally stopped after startup-window observation showed it would not reach the first planned validation/checkpoint before the wall-clock cap.
