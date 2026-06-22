# Gate Check: V1/T01/A_ac03

Status: gate completed; short-budget training finished successfully; official
post-hoc evaluation completed.

This branch exists to test the first literature-guided bottleneck hypothesis:
confidence-normalized Gaussian feature refinement.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether selective confidence-normalized refinement improves
      the current lightweight Gaussian line.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_ac03`.
- [x] Full run identifier: `V1/T01/A_ac03`.
- [x] Run folder: `experiments/V1/runs/A_ac03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local manifest and gate sheet exist.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Structural anchor: frozen `M04/A_y03`.
- [x] Matching short anchors: `A_z03`, `A_ab03`, and `A_ad03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction remains enabled at `strength = 0.02`.
- [x] Branch-specific change is
      `gaussian_feature_refinement_use_confidence_normalization = true`.
- [x] No decoder self-detail hint is enabled.
- [x] No direct edge-aware prior injection is enabled.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.

## Runtime Check

- [x] Accepted batch size: `6`.
- [x] Accepted train context size: `716`.
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
branch-local smoke 6x716: reserved_gb=92.3633, avg_total_step_seconds=1.9850, estimated_steps_58min=1753
formal preflight batch_input_shape: (6, 3, 716, 716)
formal preflight total_params: 319320
parameter delta vs A_ab03: +13248
parameter delta vs M04/A_y03: +15408
formal preflight val_psnr: 21.9700
live checkpoints: step_000250.pt, step_000500.pt, step_000750.pt, step_001000.pt, step_001250.pt, step_001500.pt, step_001750.pt, step_002000.pt, step_002250.pt, step_002500.pt, latest.pt, final.pt
live val step500: val_loss=0.0243968, val_psnr=28.5750
live val step1000: val_loss=0.0231508, val_psnr=28.8683
live val step1500: val_loss=0.0235234, val_psnr=28.6933
live val step2000: val_loss=0.0239413, val_psnr=28.6603
live val step2500: val_loss=0.0250348, val_psnr=28.3405
training clean finish: step=2500
training elapsed minutes: 37.2581
checkpoint relation: step_002500.pt == latest.pt == final.pt (model weights)
posthoc official test final psnr: 25.1474
```

## Promotion Gate Follow-Up

- [x] Completion gate passed.
- [x] Checkpoints/logs/metrics are written under the clean `A_ac03`
      branch identity.
- [x] Best training-time full-image validation PSNR collected.
- [x] Post-hoc official full-image test corroboration collected.

Interpretation:

- `A_ac03` is a clean engineering completion, but not a positive short
  continuation.
- Because the official endpoint is below `A_ad03`, below `A_ab03`, and well
  below `A_z03`, this branch should not be promoted.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M04_A_y03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This prepared branch does not modify the frozen milestone directory or
      its weights.
