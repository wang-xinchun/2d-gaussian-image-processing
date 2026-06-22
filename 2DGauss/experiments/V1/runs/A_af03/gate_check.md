# Gate Check: V1/T01/A_af03

Status: smoke and preflight passed; not launched. Kept as a valid backup short probe while `A_ah03` is launched first.

This branch exists to test the next refinement of the shared-control bottleneck hypothesis: decoder-state closed-loop Gaussian decoder guidance.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether decoder-state feedback improves the current shared-control surface.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_af03`.
- [x] Full run identifier: `V1/T01/A_af03`.
- [x] Run folder: `experiments/V1/runs/A_af03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local manifest and gate sheet exist.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Structural anchor: frozen `M04/A_y03`.
- [x] Matching short anchors: `A_z03`, `A_ae03`, `A_ad03`, and `A_ac03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction remains enabled at `strength = 0.02`.
- [x] Branch-specific change is `gaussian_decoder_guidance_use_decoder_state_gate = true`.
- [x] No decoder self-detail hint is enabled.
- [x] No direct edge-aware prior injection is enabled.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.

## Runtime Check

- [x] Accepted batch size: `6`.
- [x] Accepted train context size: `720`.
- [x] Planned supervision size: `512`.
- [x] Planned validation interval: `500`.
- [x] Planned checkpoint interval: `250`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Branch-specific smoke calibration completed on the true config.
- [x] Formal preflight completed on the true config.
- [ ] Short-budget training completed cleanly.
- [ ] Post-hoc official full-image test completed.

Runtime notes:

```text
branch-local smoke 6x720: reserved_gb=91.28125, avg_total_step_seconds=2.0052, estimated_steps_58min=1735
formal preflight batch_input_shape: (6, 3, 720, 720)
formal preflight total_params: 305304
formal preflight val_psnr: 20.2675
```

## Milestone Isolation Check

- [x] Current frozen baseline remains `experiments/V1/milestones/M04_A_y03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This prepared branch does not modify the frozen milestone directory or its weights.
