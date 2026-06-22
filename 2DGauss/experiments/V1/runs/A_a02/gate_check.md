# Gate Check: V1/T01/A_a02

Status: deployment gate passed; eligible for launch.

## Training Duration Tier

- [x] Tier: short training.
- [x] Planned wall-clock duration: capped at `58` minutes.
- [x] Purpose: data-loader-balanced derivative of A02 for the `UHD_haze` single-task short probe.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_a02`.
- [x] Full run identifier: `V1/T01/A_a02`.
- [x] Run folder: `experiments/V1/runs/A_a02`.
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
- [x] Image resolution: accepted as `3840x2160`; A_a02 uses `512x512` random training patches and `512x512` validation center crops.

## Contamination Check

- [x] Config source: `experiments/V1/templates/T01/configs/a_a02_short_uhdhaze_patch_iobalanced.toml`.
- [x] Dataset paths point to `UHD_haze`, not stale `UHD-LL_down4`.
- [x] Model architecture parameters match current V1 defaults.
- [x] Loss terms: Charbonnier weight `1.0`, gradient loss weight `0.05`.
- [x] Optimizer: AdamW, lr `0.0002`, weight decay `0.0001`.
- [x] Evaluation protocol: RGB tensor range `[0,1]`, PSNR on clamped output, validation center crop.
- [x] Pretrained weight path: empty; A_a02 starts from random initialization.
- [x] Resume checkpoint path: empty; A_a02 is a fresh derivative run, not a resume of stopped A02.
- [x] Output directory is tied to the new run `experiments/V1/runs/A_a02`.
- [x] Random seed: `20260425`.

## Runtime Check

- [x] Batch size: `16`.
- [x] Gradient accumulation: `1`.
- [x] Patch policy: `512x512` random training patches.
- [x] Max steps: `2000`.
- [x] Planned duration cap: `58` minutes.
- [x] Validation interval: every `250` steps.
- [x] Checkpoint interval: every `250` steps.
- [x] Validation batches: `100`.
- [x] DataLoader workers: `8`.
- [x] Device plan: `auto`, CUDA if available.
- [x] AMP: disabled for conservatism.
- [x] Expected memory: data-inclusive smoke reserved about `25.844G` at batch size `16`.
- [x] Expected end-to-end speed: data-inclusive smoke average about `1.539s/step`, estimated about `2144` steps in 55 minutes before validation overhead.

## Launch Decision

- [x] Gate file created.
- [x] Syntax checks passed.
- [x] Data-inclusive smoke calibration passed.
- [x] Formal preflight passed.
- [x] A_a02 is eligible for launch after formal preflight completion.
