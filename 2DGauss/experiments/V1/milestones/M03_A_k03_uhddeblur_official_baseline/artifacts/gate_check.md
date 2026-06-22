# Gate Check: V1/T01/A_k03

Status: gate completed; training finished; branch formally accepted.

This branch exists to extend the accepted `A_i03` line without promoting the
negative `A_j03` context-enlargement direction.

## Training Duration Tier

- [x] Tier target: medium training.
- [x] Planned wall-clock cap: `300` minutes.
- [x] Purpose: test whether the accepted `A_i03` setting was stopped too early
      by `max_steps=12000`, rather than by a true runtime or generalization
      ceiling.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_k03`.
- [x] Full run identifier: `V1/T01/A_k03`.
- [x] Run folder: `experiments/V1/runs/A_k03`.
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
- [x] Effective training size: `1924`.
- [x] Official test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/input300`.
- [x] Official test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300`.
- [x] Official test remains reserved for post-training evaluation only.

## Mechanism Check

- [x] Baseline protocol anchor: frozen `A_i03/M02`.
- [x] New mechanism class: none.
- [x] Main experimental variable: remove the earlier `12000`-step cap while
      preserving the accepted setting.
- [x] Consistency switch remains enabled.
- [x] Training context target: `784`.
- [x] Training supervision center target: `512`.
- [x] Model story preserved: `lowres -> gaussian -> highres` remains unchanged.
- [x] Data split preserved relative to `A_i03`.

## Runtime Check

- [x] Planned batch size: `7`.
- [x] Validation crop remains full image, `crop_size=0`.
- [x] Planned validation interval: `1000`.
- [x] Planned checkpoint interval: `500`.
- [x] Planned validation batches: `20`.
- [x] Planned train context size: `784`.
- [x] Planned supervision size: `512`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Planned wall-clock target recorded: `300` minutes.
- [x] Planned max steps raised to `24000`.
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
- [x] Small-sample preflight on the selected setting.

Runtime notes:

```text
baseline anchor: A_i03/M02
negative screening branch excluded: A_j03 context864
selected long-budget variable: extend max_steps only
template config full_id: V1/T01/A_k03
run-local config full_id: V1/T01/A_k03
smoke batch=7 context=784 reserved_gb: 88.4258
smoke estimated_steps_300min: 13108
formal preflight: ok
formal preflight batch_input_shape: (7, 3, 784, 784)
formal preflight total_params: 246888
formal preflight val_psnr: 21.7638
active tmux session: 2dgauss_a_k03_uhddeblur_tlc_stepcap_extension
active log: experiments/V1/runs/A_k03/logs/train_20260514_104102.log
startup real memory_mib: 90833
startup metrics observed through step: 20
```

## Post-Run Acceptance Follow-Up

- [x] Clean training completion verified.
- [x] Final step recorded: `23016`.
- [x] Completion reason confirmed: `wall_clock_minutes=300` cap.
- [x] Periodic checkpoints complete through `step_023000.pt`.
- [x] Distinct `final.pt` verified.
- [x] Post-hoc full-image internal validation completed.
- [x] Post-hoc official full-image test completed.
- [x] Official-test winner identified.
- [x] Same-protocol comparison against frozen `A_i03/M02` completed.
- [x] Branch accepted as new official `UHD_deblur` baseline.

Post-hoc notes:

```text
training clean finish: step=23016
final elapsed minutes: 299.9313
posthoc val step_022000 psnr: 29.4266
posthoc test step_022000 psnr: 26.0810
posthoc val final psnr: 29.2665
posthoc test final psnr: 26.2000
delta vs A_i03 final official baseline: +0.3493 dB
remaining gap to UHD-Blur 90% threshold 26.5743: 0.3743 dB
```

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M02_A_i03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This branch does not modify the frozen milestone directory or its weights.

Follow-up after acceptance:

- [ ] Freeze this branch as the next milestone package.
- [ ] Repoint the stable baseline link to the new milestone.
