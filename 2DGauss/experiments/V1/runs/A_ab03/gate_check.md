# Gate Check: V1/T01/A_ab03

Status: gate completed; short-budget training finished successfully; official
post-hoc evaluation completed.

This branch exists to test the first post-`A_aa03` structural redesign
hypothesis: edge-aware Gaussian feature refinement.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether explicit high-resolution edge priors improve the
      current feature-refinement line.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_ab03`.
- [x] Full run identifier: `V1/T01/A_ab03`.
- [x] Run folder: `experiments/V1/runs/A_ab03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config exists.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Structural anchor: frozen `M04/A_y03`.
- [x] Matching short anchor: `A_z03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction remains enabled at `strength = 0.02`.
- [x] Branch-specific change is
      `gaussian_feature_refinement_use_edge_aware_prior = true`.
- [x] Decoder self-detail hint is disabled.
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
- [x] Explicit deployment completed.
- [x] Short-budget training completed cleanly.
- [x] Post-hoc official full-image test completed.

## Static Checks Planned

- [x] Template config identity check.
- [x] Run-local config identity check.
- [x] True-config formal preflight completed.

Runtime notes:

```text
branch-local smoke 6x728: reserved_gb=91.8457, avg_total_step_seconds=1.8733, estimated_steps_58min=1857
formal preflight batch_input_shape: (6, 3, 728, 728)
formal preflight total_params: 306072
parameter delta vs A_y03/M04: +2160
formal preflight val_psnr: 23.3538
tmux session: 2dgauss_a_ab03_uhddeblur_edgeaware_feature_refinement_probe
startup nvidia-smi: memory.used≈94507 MiB, utilization.gpu=100%
train log: experiments/V1/runs/A_ab03/logs/train_20260517_234834.log
live train_metrics: step1 loss_total=0.15306, step10=0.06798, step20=0.05440
live checkpoints: step_000250.pt, step_000500.pt, step_000750.pt, step_001000.pt, step_001250.pt, step_001500.pt, step_001750.pt, step_002000.pt, step_002250.pt, step_002500.pt, latest.pt, final.pt
live val step500: val_loss=0.0248542, val_psnr=28.5939
live val step1000: val_loss=0.0246931, val_psnr=28.5592
live val step1500: val_loss=0.0239989, val_psnr=28.5976
live val step2000: val_loss=0.0257546, val_psnr=28.2654
live val step2500: val_loss=0.0240276, val_psnr=28.7948
training clean finish: step=2500
training elapsed minutes: 36.1174
checkpoint relation: step_002500.pt == latest.pt == final.pt (model weights)
posthoc official test final psnr: 25.1723
```

## Promotion Gate Follow-Up

- [x] Completion gate passed.
- [x] Checkpoints/logs/metrics are written under the clean `A_ab03`
      branch identity.
- [x] Best training-time full-image validation PSNR collected.
- [x] Post-hoc official full-image test corroboration collected.

Interpretation:

- `A_ab03` is a clean engineering completion, but not a positive short
  continuation.
- Because the official endpoint is below `A_z03` and below the older short
  anchors, this branch should not be promoted.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M04_A_y03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This prepared branch does not modify the frozen milestone directory or
      its weights.
