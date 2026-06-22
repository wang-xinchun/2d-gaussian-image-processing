# Gate Check: V1/T01/A_n03

Status: gate completed; training finished; post-hoc official test completed.

This branch is the default-strength counterpart to `A_m03` inside the
post-`M03` short-scout family.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: compare default-strength decoder guidance against the completed
      low-strength `A_m03` probe under the same runtime geometry.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_n03`.
- [x] Full run identifier: `V1/T01/A_n03`.
- [x] Run folder: `experiments/V1/runs/A_n03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config will be created automatically by the training entrypoint.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Baseline protocol anchor: frozen `A_k03/M03`.
- [x] Main experimental variable: `gaussian_decoder_guidance_strength = 0.15`.
- [x] Relative to `A_m03`, geometry remains fixed at `batch=6`, `context=784`.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.

## Runtime Check

- [x] Planned batch size: `6`.
- [x] Planned train context size: `784`.
- [x] Planned supervision size: `512`.
- [x] Planned validation interval: `500`.
- [x] Planned checkpoint interval: `250`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Clean GPU confirmed immediately before accepted smoke calibration.
- [x] Branch-specific smoke calibration completed.
- [x] Smoke result inside `85.0-93.0G`.
- [x] Formal preflight completed.
- [x] Short-budget training completed cleanly.
- [x] Post-hoc official full-image test completed.

Runtime notes:

```text
smoke batch=6 context=784 reserved_gb: 87.7734
formal preflight batch_input_shape: (6, 3, 784, 784)
formal preflight total_params: 267864
formal preflight val_psnr: 21.7206
training clean finish: step=2500
training elapsed minutes: 33.7058
training-time best val psnr: 28.9612 at step2500
posthoc official test final psnr: 25.2919
```

## Promotion Gate Follow-Up

- [x] Completion gate passed.
- [x] Checkpoints and metrics written through the full short budget.
- [x] Best training-time full-image validation PSNR reached `28.9612 dB`.
- [x] This exceeds the family strong-keep threshold `28.80 dB`.
- [x] Post-hoc official full-image test corroboration collected.

Interpretation:

- `A_n03` is a valid strong keep candidate for the next long-run nomination
  stage.
- At the training-time validation level it currently outranks `A_m03`.

## Milestone Isolation Check

- [x] Frozen milestone remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] This prepared branch must not modify frozen milestone directories or
      weights.
