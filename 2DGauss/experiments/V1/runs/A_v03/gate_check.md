# Gate Check: V1/T01/A_v03

Status: prepared only; smoke, preflight, launch, and post-hoc evaluation all
pending.

This branch exists to test whether `A_t03` needs slightly stronger
mechanism-side feature refinement rather than another supervision-side change.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test a stronger feature-refinement rebalance around `A_t03`.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_v03`.
- [x] Full run identifier: `V1/T01/A_v03`.
- [x] Run folder: `experiments/V1/runs/A_v03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config exists.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Structural anchor: completed `A_t03`.
- [x] Earlier short anchor: completed `A_n03`.
- [x] Companion rebalance probes: `A_u03` and `A_w03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled.
- [x] Only branch-specific change is `gaussian_feature_refinement_strength = 0.12`.
- [x] Edge-weighted reconstruction remains enabled at `strength = 0.03`.
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
- [ ] Explicit deployment completed.
- [ ] Short-budget training completed cleanly.
- [ ] Post-hoc official full-image test completed.

## Static Checks Completed

- [x] Template config identity check.
- [x] Run-local config identity check.
- [x] True-config formal preflight completed.

Runtime notes:

```text
branch-local smoke 6x728: reserved_gb=91.7871, avg_total_step_seconds=1.8449, estimated_steps_58min=1886
formal preflight batch_input_shape: (6, 3, 728, 728)
formal preflight total_params: 303912
formal preflight val_psnr: 22.2929
```

## Promotion Gate Follow-Up

- [ ] Completion gate passed.
- [ ] Checkpoints and metrics written through the full short budget.
- [ ] Best training-time full-image validation PSNR collected.
- [ ] Post-hoc official full-image test corroboration collected.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This prepared branch does not modify the frozen milestone directory or
      its weights.
