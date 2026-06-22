# Gate Check: V1/T01/A_i03

Status: gate completed; launch-approved in this turn after recalibration.

This branch exists to deploy a medium-budget post-milestone `UHD_deblur`
experiment after the frozen `A_g03` baseline, while explicitly respecting the
user's later requirement that real launch memory must stay inside
`85.0-93.0G`.

## Training Duration Tier

- [x] Tier target: medium training.
- [x] Planned wall-clock cap: `300` minutes.
- [x] Purpose: test whether the prepared `TLC`-style train-inference
      consistency repair becomes more effective under longer budget and a
      slightly larger effective train split.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_i03`.
- [x] Full run identifier: `V1/T01/A_i03`.
- [x] Run folder: `experiments/V1/runs/A_i03`.
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
- [x] Effective training size increases from `1864` to `1924`.
- [x] Official test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/input300`.
- [x] Official test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300`.
- [x] Official test remains reserved for post-training evaluation only.

## Mechanism Check

- [x] Baseline protocol anchor: `A_g03` accepted edge-guided +
      directional-frequency branch.
- [x] New mechanism class: train-inference consistency repair.
- [x] New config switch: `use_train_inference_consistency = true`.
- [x] Training context target: `784`.
- [x] Training supervision center target: `512`.
- [x] Model story preserved: `lowres -> gaussian -> highres` remains unchanged.
- [x] Backward compatibility: mechanism is default-off in code and should leave
      old branches unchanged when disabled.

## Runtime Check

- [x] Initial batch-size hypothesis recorded and later recalibrated.
- [x] Validation crop remains full image, `crop_size=0`.
- [x] Planned validation interval: `1000`.
- [x] Planned checkpoint interval: `500`.
- [x] Planned validation batches: `20`.
- [x] Planned train context size: `784`.
- [x] Planned supervision size: `512`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Planned wall-clock target recorded: `300` minutes.
- [x] Branch-specific smoke calibration completed.
- [x] Training reserved VRAM reconfirmed inside `85.0-93.0G`.
- [x] Estimated step count under `300` minutes recorded from real smoke output.
- [x] Formal preflight completed on the final `7x784` setting.
- [x] Launch script created.
- [x] Explicit deployment authorization for this branch received.

## Static Checks To Complete In This Turn

- [x] `scripts/train_v1_t01.py`: py_compile ok.
- [x] `scripts/smoke_calibrate_v1_t01.py`: py_compile ok.
- [x] `scripts/evaluate_v1_t01.py`: py_compile ok.
- [x] Template config identity check.
- [x] Run-local config identity check.
- [x] Small-sample preflight on new consistency-enabled path.
- [ ] CPU compatibility check on old accepted path.
- [ ] Real-dataset sample check for `768x768` training context crop.

Static notes:

```text
template config full_id: V1/T01/A_i03
run-local config full_id: V1/T01/A_i03
use_edge_guidance: true
use_directional_frequency_guidance: true
use_train_inference_consistency: true
train_context_size: 784
train_supervision_crop_size: 512
small-sample preflight: ok
preflight batch_input_shape: (8, 3, 768, 768)
preflight total_params: 246888
preflight val_psnr: 21.4903
smoke batch=7 context=768 reserved_gb: 84.4570
smoke batch=8 context=768 reserved_gb: 93.1113
smoke batch=8 context=752 reserved_gb: 91.9004
real launch trial batch=8 context=752 startup_memory_mib: 94813
real launch trial disposition: stopped for violating 93G upper bound
smoke batch=7 context=784 reserved_gb: 88.0137
selected launch setting: batch=7, context=784
estimated_steps_300min: 12244
formal preflight final setting: ok
formal preflight batch_input_shape: (7, 3, 784, 784)
formal preflight val_psnr: 21.7638
active tmux session: 2dgauss_a_i03_uhddeblur_tlc_medium
active log: experiments/V1/runs/A_i03/logs/train_20260514_022908.log
startup real memory_mib: 90833
startup metrics observed through step: 70
```

## Post-Launch Follow-Up

1. Training completion verified: `step=12000`, `elapsed=156.1359min`,
   `final.pt` and `step_012000.pt` present.
2. Post-hoc full-image validation and official full-image test completed on
   `step_011000.pt` and `final.pt`.
3. `final.pt` is the official winner for this branch.

Post-hoc notes:

```text
posthoc val step_011000 psnr: 29.2631
posthoc test step_011000 psnr: 25.7934
posthoc val final psnr: 29.2847
posthoc test final psnr: 25.8506
```

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M01_A_g03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This turn does not modify the frozen milestone directory or its weights.
