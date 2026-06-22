# Gate Check: V1/T01/A_s03

Status: gate completed; smoke-calibrated, preflighted, trained, and post-hoc
official evaluation completed.

This branch exists to prepare the first supervision-only probe after the
completed negative `A_q03` terminal-residual result.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether a small edge-weighted reconstruction term can
      improve detail fidelity without changing the accepted model structure.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_s03`.
- [x] Full run identifier: `V1/T01/A_s03`.
- [x] Run folder: `experiments/V1/runs/A_s03`.
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
- [x] Main experimental variable: edge-weighted reconstruction supervision.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Edge-weighted reconstruction is enabled at `strength = 0.03`.
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
- [x] Branch-specific smoke calibration completed.
- [x] Formal preflight completed on the prepared config.
- [x] Launch script created.
- [x] Explicit deployment authorization for this branch received.

## Static Checks Completed

- [x] `scripts/train_v1_t01.py`: py_compile ok.
- [x] `src/uhd_gaussian/config.py`: py_compile ok.
- [x] `src/uhd_gaussian/__init__.py`: py_compile ok.
- [x] Template config identity check.
- [x] Run-local config identity check.
- [x] True-config formal preflight completed.

Preflight notes:

```text
formal preflight batch_input_shape: (6, 3, 784, 784)
formal preflight total_params: 267864
formal preflight val_psnr: 21.7247
```

## Post-Hoc Outcome Notes

- [x] Training completed cleanly at `step=2500`.
- [x] `step_002500.pt`, `latest.pt`, and `final.pt` share identical `model`
      weights.
- [x] Post-hoc official test completed on `final.pt`.
- [x] Official test result: `loss_mean=0.03783569399267435`,
      `psnr_mean=25.25673342402521`.
- [x] Result is below `A_n03`, so this branch is not the preferred next
      promotion anchor.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This turn does not modify the frozen milestone directory or its weights.
