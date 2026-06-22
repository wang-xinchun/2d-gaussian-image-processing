# Gate Check: V1/T01/A01

Status: deployment smoke passed; eligible for first short-training deployment after final user approval.

## Training Duration Tier

- [x] Tier: short training.
- [x] Planned wall-clock duration: about 1 hour, capped at 62 minutes.
- [x] Purpose: verify the first V1 training loop on a validated paired dataset before any medium or long training.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A01`.
- [x] Full run identifier: `V1/T01/A01`.
- [x] Run folder: `experiments/V1/runs/A01`.
- [x] The exact training-template snapshot is configured to be saved under `template_snapshot/` at launch.

## Dataset Check

- [x] Dataset name: `UHD-LL_down4`.
- [x] Dataset root: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4`.
- [x] Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/training_set/input`.
- [x] Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/training_set/gt`.
- [x] Validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/training_set/input`.
- [x] Validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/training_set/gt`.
- [x] Validation selection: first `100` paired samples from `training_set` in deterministic filename order.
- [x] Training selection: remaining `1900` paired samples from `training_set`.
- [x] Final test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/testing_set/input`.
- [x] Final test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4/testing_set/gt`.
- [x] Accepted inventory reference: `docs/dataset_organization_status_20260424.md`.
- [x] Expected sample counts: source train input/gt 2000/2000, internal validation 100, effective train 1900, final test input/gt 150/150.
- [x] Image resolution: accepted as `960x540`; training uses full `540x960` images, validation uses center crop up to `512x512`.
- [x] Official `testing_set` is not used for training-time validation; it is reserved for final post-training evaluation.
- [x] No synthetic degradation policy is used in A01.

## Contamination Check

- [x] Config source: `experiments/V1/templates/T01/configs/a01_short_uhdll_down4.toml`.
- [x] Dataset paths match the A01 plan.
- [x] Model architecture parameters match current V1 defaults.
- [x] Loss terms: Charbonnier weight `1.0`, gradient loss weight `0.05`.
- [x] Optimizer: AdamW, lr `0.0002`, weight decay `0.0001`.
- [x] Evaluation protocol: RGB tensor range `[0,1]`, PSNR on clamped output, validation center crop.
- [x] Pretrained weight path: empty; A01 starts from random initialization.
- [x] Resume checkpoint path: empty; A01 is a fresh short run.
- [x] Output directory is tied to the new run `experiments/V1/runs/A01`.
- [x] Random seed: `20260424`.

## Runtime Check

- [x] Batch size: `28`.
- [x] Gradient accumulation: `1`.
- [x] Patch/full-image policy: full `540x960` images for training (`train_crop_size=0`).
- [x] Max steps: `4300`.
- [x] Planned duration: `62` minutes.
- [x] Validation interval: every `200` steps.
- [x] Checkpoint interval: every `200` steps.
- [x] Device plan: `auto`, CUDA if available.
- [x] AMP: disabled for A01 conservatism.
- [x] Expected memory: target `85G-93G`; smoke-calibrated peak reserved memory is about `89.453G`.

## Launch Decision

- [x] Gate file created and checked.
- [x] Any future exception must be written into `manifest.md`.
- [x] Corrected preflight command succeeded after the split-policy correction.
- [x] Deployment smoke calibration succeeded with batch size `28`, full-image training, and peak reserved memory about `89.453G`.
- [x] User requested the first deployment to run around 1 hour, with duration no shorter than 1 hour if possible; A01 is therefore capped at `62` minutes with `4300` max steps.
- [x] Formal A01 preflight succeeded with configured batch size `28` and full-image batches `(28, 3, 540, 960)`.
- [x] A01 is eligible for short-training launch after final user approval.
