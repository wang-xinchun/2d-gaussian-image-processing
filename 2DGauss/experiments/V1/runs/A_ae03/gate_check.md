# Gate Check: V1/T01/A_ae03

Status: gate completed; short-budget training finished successfully; official post-hoc evaluation completed.

This branch exists to test the next refinement of the shared-control bottleneck hypothesis: shared-prior confidence-gated Gaussian control.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether a shared-prior confidence gate improves the current lightweight Gaussian line.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_ae03`.
- [x] Full run identifier: `V1/T01/A_ae03`.
- [x] Run folder: `experiments/V1/runs/A_ae03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local manifest and gate sheet exist.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Structural anchor: frozen `M04/A_y03`.
- [x] Matching short anchors: `A_z03`, `A_ab03`, `A_ac03`, and `A_ad03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction remains enabled at `strength = 0.02`.
- [x] Branch-specific change is shared-prior confidence gating inside Gaussian feature refinement.
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
- [x] Short-budget training completed cleanly.
- [x] Post-hoc official full-image test completed.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M04_A_y03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This prepared branch does not modify the frozen milestone directory or
      its weights.
