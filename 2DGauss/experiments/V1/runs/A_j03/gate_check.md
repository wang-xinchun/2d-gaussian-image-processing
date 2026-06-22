# Gate Check: V1/T01/A_j03

Status: gate completed; queued deployment succeeded and training finished.

This branch exists to screen the next post-`M02` long-run direction with a
single-variable short probe before committing more budget.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether a larger consistency-training context than
      `A_i03` is worth promoting to the next long-budget run.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_j03`.
- [x] Full run identifier: `V1/T01/A_j03`.
- [x] Run folder: `experiments/V1/runs/A_j03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config exists.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`.
- [x] Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`.
- [x] Validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`.
- [x] Validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`.
- [x] Validation selection: first `20` train pairs.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Effective training size remains aligned with `A_i03`: `1924`.
- [x] Official test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/input300`.
- [x] Official test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300`.
- [x] Official test remains reserved for post-training evaluation only.

## Mechanism Check

- [x] Baseline protocol anchor: accepted `A_i03/M02`.
- [x] New mechanism class: none.
- [x] Main experimental variable: training context enlargement under the same
      consistency framework.
- [x] Consistency switch remains enabled.
- [x] Training context target: `864`.
- [x] Training supervision center target: `512`.
- [x] Model story preserved: `lowres -> gaussian -> highres` remains unchanged.
- [x] Data split preserved relative to `A_i03`.

## Runtime Check

- [x] Initial batch-size hypothesis recorded: `6`.
- [x] Validation crop remains full image, `crop_size=0`.
- [x] Planned validation interval: `500`.
- [x] Planned checkpoint interval: `250`.
- [x] Planned validation batches: `20`.
- [x] Planned train context size: `864`.
- [x] Planned supervision size: `512`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Branch-specific smoke calibration completed.
- [x] Training reserved VRAM reconfirmed inside `85.0-93.0G`.
- [x] Formal preflight completed.
- [x] Launch script created.
- [x] Explicit deployment authorization for this branch received.

## Static Checks To Complete In This Turn

- [x] `scripts/train_v1_t01.py`: py_compile ok.
- [x] `scripts/smoke_calibrate_v1_t01.py`: py_compile ok.
- [x] `scripts/evaluate_v1_t01.py`: py_compile ok.
- [x] Template config identity check.
- [x] Run-local config identity check.
- [x] Small-sample preflight on the new setting.

Runtime notes:

```text
deployment mode: queued launch after external Spark process released GPU
smoke batch=6 context=864: passed
formal preflight: passed
training finish step: 2500
training finish elapsed_minutes: 35.0091
posthoc official test step_001000 psnr: 25.0181
posthoc official test final psnr: 25.1965
```

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M02_A_i03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This branch does not modify the frozen milestone directory or its weights.
