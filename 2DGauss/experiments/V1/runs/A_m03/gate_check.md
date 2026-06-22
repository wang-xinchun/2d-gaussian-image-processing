# Gate Check: V1/T01/A_m03

Status: gate completed; training finished; post-hoc official test completed.

This branch is the first conservative architecture scout in the
`A_m03/A_n03/A_o03` short-probe family.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether low-strength Gaussian-guided decoder modulation is
      directionally healthy before the next `5`-hour run.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_m03`.
- [x] Full run identifier: `V1/T01/A_m03`.
- [x] Run folder: `experiments/V1/runs/A_m03`.
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
- [x] Effective training size remains aligned with the post-`M03` mainline.
- [x] Official test input: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/input300`.
- [x] Official test target: `/root/autodl-tmp/2D_Gaussian_Dataset/UHD_deblur/test/gt300`.
- [x] Official test remains corroborative post-training evidence rather than
      the short-probe promotion gate itself.

## Mechanism Check

- [x] Baseline protocol anchor: frozen `A_k03/M03`.
- [x] Main experimental variable: enable Gaussian-guided decoder modulation.
- [x] Probe strength setting: `gaussian_decoder_guidance_strength = 0.10`.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.
- [x] Training context target: `784`.
- [x] Training supervision center target: `512`.
- [x] Model story preserved: `lowres -> gaussian -> highres` remains unchanged.

## Runtime Check

- [x] Planned batch size: `6`.
- [x] Validation crop remains full image, `crop_size=0`.
- [x] Planned validation interval: `500`.
- [x] Planned checkpoint interval: `250`.
- [x] Planned validation batches: `20`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] No concurrent GPU contamination during accepted serial calibration.
- [x] Branch-specific smoke calibration completed.
- [x] Training reserved VRAM reconfirmed inside `85.0-93.0G`.
- [x] Formal preflight completed.
- [x] Direct serial launch completed.

Runtime notes:

```text
smoke batch=6 context=784 reserved_gb: 87.2676
formal preflight batch_input_shape: (6, 3, 784, 784)
formal preflight total_params: 267864
formal preflight val_psnr: 21.0643
training clean finish: step=2500
training elapsed minutes: 33.7139
training-time best val psnr: 28.8486 at step2500
posthoc official test final psnr: 25.2690
```

## Promotion Gate Follow-Up

- [x] Completion gate passed.
- [x] Checkpoints and metrics written through the full short budget.
- [x] Best training-time full-image validation PSNR reached `28.8486 dB`.
- [x] This exceeds the family strong-keep threshold `28.80 dB`.
- [x] Post-hoc official full-image test completed for corroboration.

Interpretation:

- `A_m03` is a valid strong keep candidate for the next long-run nomination
  stage.
- It is not an official baseline replacement branch.

## Milestone Isolation Check

- [x] Frozen milestone remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] Stable baseline link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This branch did not modify the frozen milestone directory or its
      weights.
