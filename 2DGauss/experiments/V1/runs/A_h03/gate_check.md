# Gate Check: V1/T01/A_h03

Status: design prepared; **not** launch-approved yet.

This branch exists to prepare the next `UHD_deblur` experiment after the frozen
`A_g03` milestone, while explicitly respecting the user's instruction not to
deploy training in this turn.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether a `TLC`-style train-inference consistency repair
      can improve official full-image `UHD_deblur` performance beyond the
      frozen `A_g03` baseline.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_h03`.
- [x] Full run identifier: `V1/T01/A_h03`.
- [x] Run folder: `experiments/V1/runs/A_h03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config exists.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Train input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`.
- [x] Train target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`.
- [x] Validation input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/input_new`.
- [x] Validation target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/train/gt_new`.
- [x] Validation selection: first `20` train pairs.
- [x] Effective training selection: train pairs after the first `100`.
- [x] Official test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/input300`.
- [x] Official test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300`.
- [x] Official test remains reserved for post-training evaluation only.

## Mechanism Check

- [x] Baseline protocol anchor: `A_g03` accepted edge-guided +
      directional-frequency branch.
- [x] New mechanism class: train-inference consistency repair.
- [x] New config switch: `use_train_inference_consistency = true`.
- [x] Training context target: `768`.
- [x] Training supervision center target: `512`.
- [x] Model story preserved: `lowres -> gaussian -> highres` remains unchanged.
- [x] Backward compatibility: mechanism is default-off in code and should leave
      old branches unchanged when disabled.

## Runtime Check

- [x] Initial batch-size hypothesis recorded: `8`.
- [x] Validation crop remains full image, `crop_size=0`.
- [x] Planned validation interval: `500`.
- [x] Planned checkpoint interval: `250`.
- [x] Planned validation batches: `20`.
- [x] Planned train context size: `768`.
- [x] Planned supervision size: `512`.
- [ ] Branch-specific smoke calibration completed.
- [ ] Training reserved VRAM reconfirmed to be at least `80G`.
- [ ] Formal preflight completed.
- [ ] Launch script created.
- [ ] Explicit deployment authorization for this branch received.

## Static Checks Completed

- [x] `scripts/train_v1_t01.py`: py_compile ok.
- [x] `scripts/smoke_calibrate_v1_t01.py`: py_compile ok.
- [x] `scripts/evaluate_v1_t01.py`: py_compile ok.
- [x] Template config identity check.
- [x] Run-local config identity check.
- [x] CPU compatibility check on old accepted path.
- [x] CPU forward check on new consistency-enabled path.
- [x] Real-dataset sample check for `768x768` training context crop.

Static notes:

```text
template config full_id: V1/T01/A_h03
run-local config full_id: V1/T01/A_h03
use_edge_guidance: true
use_directional_frequency_guidance: true
use_train_inference_consistency: true
train_context_size: 768
train_supervision_crop_size: 512
old-path CPU forward: ok
new-path CPU forward: ok
dataset sample crop: (3, 768, 768)
```

## Deferred Work Before Launch

1. Perform serial smoke calibration on clean GPU using the `A_h03` config.
2. Search a stable high-VRAM batch size, likely starting near `8`.
3. Run formal preflight on the final selected launch setting.
4. Only then create a launch script and consider deployment.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M01_A_g03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This turn does not modify the frozen milestone directory or its weights.
