# Gate Check: V1/T01/A_t03

Status: gate completed; training finished; post-hoc official test completed.

This branch exists to prepare the combined feature-refinement plus
edge-weighted probe after the completed negative `A_q03` result.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether mechanism-side and supervision-side detail fixes are
      complementary under the same lightweight budget.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_t03`.
- [x] Full run identifier: `V1/T01/A_t03`.
- [x] Run folder: `experiments/V1/runs/A_t03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config exists.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Baseline protocol anchor: completed `A_n03`.
- [x] Single-variable companions: `A_r03` and `A_s03`.
- [x] Negative-quality reference: completed `A_q03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement is enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction is enabled at `strength = 0.03`.
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
- [x] Launch script created.
- [x] Explicit deployment authorization for this branch received.
- [x] Short-budget training completed cleanly.
- [x] Post-hoc official full-image test completed.

## Static Checks Completed

- [x] `src/uhd_gaussian/models/feature_refinement.py`: py_compile ok.
- [x] `src/uhd_gaussian/models/restorer.py`: py_compile ok.
- [x] `src/uhd_gaussian/config.py`: py_compile ok.
- [x] `src/uhd_gaussian/__init__.py`: py_compile ok.
- [x] `scripts/train_v1_t01.py`: py_compile ok.
- [x] Template config identity check.
- [x] Run-local config identity check.
- [x] True-config formal preflight completed.

Runtime notes:

```text
branch-local smoke 6x728: reserved_gb=91.7852 (accepted provisional geometry)
formal preflight batch_input_shape: (6, 3, 728, 728)
formal preflight total_params: 303912
formal preflight val_psnr: 22.3029
real startup memory.used≈94067 MiB (slightly above original practical 93G ceiling)
training clean finish: step=2500
training elapsed minutes: 35.6375
training-time best val psnr: 28.7877 at step2500
posthoc official test final psnr: 25.2944
```

## Promotion Gate Follow-Up

- [x] Completion gate passed.
- [x] Checkpoints and metrics written through the full short budget.
- [x] Best training-time full-image validation PSNR reached `28.7877 dB`.
- [ ] This exceeds the family strong-keep threshold `28.80 dB`.
- [x] Post-hoc official full-image test corroboration collected.

Interpretation:

- `A_t03` is the strongest completed short result inside the post-`A_q03`
  family.
- But the official margin over `A_n03` is only about `+0.0025 dB`, so this is
  not a decisive short-run promotion win.
- If this family is extended, `A_t03` is the best short candidate for a
  confirming medium run; otherwise the line should return to structural
  redesign.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This turn does not modify the frozen milestone directory or its weights.
