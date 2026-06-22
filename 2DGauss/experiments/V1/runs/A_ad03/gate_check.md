# Gate Check: V1/T01/A_ad03

Status: gate completed; short-budget training finished successfully; official
post-hoc evaluation completed.

This branch exists to test the second literature-guided bottleneck hypothesis:
shared-prior coupling between Gaussian decoder guidance and Gaussian feature
refinement.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether tighter coordination between the two Gaussian
      control modules improves the current lightweight line.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_ad03`.
- [x] Full run identifier: `V1/T01/A_ad03`.
- [x] Run folder: `experiments/V1/runs/A_ad03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local manifest and gate sheet exist.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Structural anchor: frozen `M04/A_y03`.
- [x] Matching short anchors: `A_z03`, `A_ab03`, and `A_ac03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction remains enabled at `strength = 0.02`.
- [x] Branch-specific change is `use_gaussian_shared_prior = true`.
- [x] No decoder self-detail hint is enabled.
- [x] No direct edge-aware prior injection is enabled.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.

## Runtime Check

- [x] Accepted batch size: `6`.
- [x] Accepted train context size: `728`.
- [x] Planned supervision size: `512`.
- [x] Planned validation interval: `500`.
- [x] Planned checkpoint interval: `250`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Branch-specific smoke calibration completed on the true config.
- [x] Formal preflight completed on the true config.
- [x] Short-budget training completed cleanly.
- [x] Post-hoc official full-image test completed.

Runtime notes:

```text
branch-local smoke 6x728: reserved_gb=86.0547, avg_total_step_seconds=1.8772, estimated_steps_58min=1853
formal preflight batch_input_shape: (6, 3, 728, 728)
formal preflight total_params: 289992
parameter delta vs A_ab03: -16080
formal preflight val_psnr: 17.4351
live checkpoints: step_000250.pt, step_000500.pt, step_000750.pt, step_001000.pt, step_001250.pt, step_001500.pt, step_001750.pt, step_002000.pt, step_002250.pt, step_002500.pt, latest.pt, final.pt
live val step500: val_loss=0.0240792, val_psnr=28.6524
live val step1000: val_loss=0.0241585, val_psnr=28.6875
live val step1500: val_loss=0.0240461, val_psnr=28.6774
live val step2000: val_loss=0.0245466, val_psnr=28.5878
live val step2500: val_loss=0.0244863, val_psnr=28.6896
training clean finish: step=2500
training elapsed minutes: 33.6803
checkpoint relation: step_002500.pt == latest.pt == final.pt (model weights)
posthoc official test final psnr: 25.2008
```

## Promotion Gate Follow-Up

- [x] Completion gate passed.
- [x] Checkpoints/logs/metrics are written under the clean `A_ad03`
      branch identity.
- [x] Best training-time full-image validation PSNR collected.
- [x] Post-hoc official full-image test corroboration collected.

Interpretation:

- `A_ad03` is the better branch inside the current serial pair.
- But because the official endpoint is still below `A_z03` and the older
  short anchors, this branch is not yet promotion-ready.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M04_A_y03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This prepared branch does not modify the frozen milestone directory or
      its weights.
