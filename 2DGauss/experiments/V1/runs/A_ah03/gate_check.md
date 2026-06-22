# Gate Check: V1/T01/A_ah03

Status: smoke and preflight passed; short training completed cleanly; official full-image test on `final.pt` passed.

This branch exists to test the next refinement of the shared-control bottleneck hypothesis: control-gated edge refinement.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether edge-module role separation improves the current shared-control surface.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_ah03`.
- [x] Full run identifier: `V1/T01/A_ah03`.
- [x] Run folder: `experiments/V1/runs/A_ah03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local manifest and gate sheet exist.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Structural anchor: frozen `M04/A_y03`.
- [x] Matching short anchors: `A_z03`, `A_ae03`, `A_ad03`, and `A_ab03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction remains enabled at `strength = 0.02`.
- [x] Branch-specific change is `edge_guidance_use_control_gate = true`.
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
- [x] First validation completed at `step=500`.
- [x] Short-budget training completed cleanly.
- [x] Post-hoc official full-image test completed.

Runtime notes:

```text
branch-local smoke 6x720: reserved_gb=91.826171875, avg_total_step_seconds=1.9132, estimated_steps_58min=1818
formal preflight batch_input_shape: (6, 3, 720, 720)
formal preflight total_params: 305256
formal preflight val_psnr: 23.6768
launch tmux: 2dgauss_a_ah03_uhddeblur_edge_control_gate_probe
launch log: experiments/V1/runs/A_ah03/logs/train_20260518_234142.log
startup nvidia-smi: memory.used≈94347 MiB, utilization.gpu≈97%
first train metric: step=1, loss_total=0.1084759533405304
first validation: step=500, val_psnr=28.527268073041363, val_loss=0.024733865913003684, val_batches=20
second validation: step=1000, val_psnr=28.61423288220579, val_loss=0.024983534589409828, val_batches=20
final validation: step=2500, val_psnr=29.039882709180723, val_loss=0.022765929251909255, val_batches=20
latest train metric: step=2500, elapsed_minutes=36.139669915040336, loss_total=0.044364772737026215
final checkpoint relation: step_002500.pt == latest.pt == final.pt (model weights)
official test: checkpoint_step=2500, samples=300, loss_mean=0.03634987864177674, psnr_mean=25.345361164304613
checkpoints present: step_000250.pt, step_000500.pt, step_000750.pt, step_001000.pt, step_001250.pt, step_001500.pt, step_001750.pt, step_002000.pt, step_002250.pt, step_002500.pt, latest.pt, final.pt
```

## Milestone Isolation Check

- [x] Current frozen baseline remains `experiments/V1/milestones/M04_A_y03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This prepared branch does not modify the frozen milestone directory or its weights.
