# Gate Check: V1/T01/A_q03

Status: gate completed; short-budget training and post-hoc evaluation finished.

This branch is the first terminal-residual short probe after the completed
`A_p03` bottleneck diagnosis.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether a lightweight Gaussian-guided terminal residual
      head gives a cleaner final-reconstruction signal than decoder guidance
      alone.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_q03`.
- [x] Full run identifier: `V1/T01/A_q03`.
- [x] Run folder: `experiments/V1/runs/A_q03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config has been staged under `runs/A_q03/configs/`.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Baseline protocol anchor: frozen `A_k03/M03`.
- [x] Short-structure anchor: completed `A_n03`.
- [x] Bottleneck reference: completed `A_p03`.
- [x] Main experimental variable: direct lightweight Gaussian-guided terminal
      residual control.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Terminal residual guidance is enabled at `strength = 0.10`.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.

## Runtime Check

- [x] Planned batch size: `6`.
- [x] Planned train context size: `744`.
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

## Milestone Isolation Check

- [x] Frozen milestone remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] This branch must not modify frozen milestone directories or weights.

Calibration notes:

```text
batch6 context736: reserved_gb=85.6250
batch6 context744: reserved_gb=87.5020
selected launch target: batch6 context744
selection reason: stays inside the 85-93G window while remaining closer to the original 6x784 short-structure anchor than 6x736
```

Preflight summary:

```text
formal preflight batch_input_shape: (6, 3, 744, 744)
formal preflight total_params: 290091
formal preflight val_psnr: 22.6355
```

Completion notes:

```text
training finish step: 2500
training finish elapsed_minutes: 34.4349
official test step_001500 psnr: 25.0953
official test final psnr: 25.1826
checkpoint identity: step_002500 == latest == final
```
