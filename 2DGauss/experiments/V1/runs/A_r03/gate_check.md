# Gate Check: V1/T01/A_r03

Status: design prepared; smoke-calibrated to a provisional geometry and formal
preflight completed; **not** launch-approved yet.

This branch exists to prepare the first feature-space refinement probe after
the completed negative `A_q03` terminal-residual result.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether Gaussian-guided feature refinement gives a cleaner
      final-reconstruction signal than direct terminal RGB residual control.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_r03`.
- [x] Full run identifier: `V1/T01/A_r03`.
- [x] Run folder: `experiments/V1/runs/A_r03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config exists.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Baseline protocol anchor: completed `A_n03`.
- [x] Negative-quality reference: completed `A_q03`.
- [x] New mechanism class: Gaussian-guided feature-space refinement.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement is enabled at `strength = 0.10`.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.

## Runtime Check

- [x] Planned batch size: `6`.
- [x] Planned train context size: `728`.
- [x] Planned supervision size: `512`.
- [x] Planned validation interval: `500`.
- [x] Planned checkpoint interval: `250`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Branch-specific smoke calibration completed.
- [x] Formal preflight completed on the prepared config.
- [ ] Launch script created.
- [ ] Explicit deployment authorization for this branch received.

## Static Checks Completed

- [x] `src/uhd_gaussian/models/feature_refinement.py`: py_compile ok.
- [x] `src/uhd_gaussian/models/restorer.py`: py_compile ok.
- [x] `src/uhd_gaussian/config.py`: py_compile ok.
- [x] `src/uhd_gaussian/__init__.py`: py_compile ok.
- [x] `scripts/train_v1_t01.py`: py_compile ok.
- [x] Template config identity check.
- [x] Run-local config identity check.
- [x] True-config formal preflight completed.

Preflight notes:

```text
smoke 6x736: reserved_gb=93.7676 (above window)
smoke 6x728: reserved_gb=91.7852 (accepted provisional geometry)
formal preflight batch_input_shape: (6, 3, 728, 728)
formal preflight total_params: 303912
formal preflight val_psnr: 22.3030
```

## Deferred Work Before Launch

1. Reconfirm the selected `6x728` geometry under a clean startup observation.
2. Re-run formal preflight on the final selected launch geometry if it changes.
3. Only then create a launch script and consider deployment.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This turn does not modify the frozen milestone directory or its weights.
