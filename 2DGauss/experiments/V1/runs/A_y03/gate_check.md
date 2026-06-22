# Gate Check: V1/T01/A_y03

Status: gate completed; medium-budget launch completed; training finished successfully; official post-hoc test completed.

This branch exists to confirm whether the tiny `A_x03` edge over frozen
`A_k03` is reproducible enough to justify a later milestone refresh.

## Training Duration Tier

- [x] Tier target: medium training.
- [x] Planned wall-clock cap: `300` minutes.
- [x] Purpose: confirm the `A_x03` result under a fresh seed.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_y03`.
- [x] Full run identifier: `V1/T01/A_y03`.
- [x] Run folder: `experiments/V1/runs/A_y03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config exists.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Replay anchor: completed `A_x03`.
- [x] Frozen milestone for confirmation: `M03/A_k03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction remains enabled at `strength = 0.02`.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.

## Runtime Check

- [x] Planned batch size target: `6`.
- [x] Planned train context target: `728`.
- [x] Planned supervision size: `512`.
- [x] Planned validation interval: `1000`.
- [x] Planned checkpoint interval: `500`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Branch-specific smoke calibration completed.
- [x] Final formal preflight completed on the accepted launch config.
- [x] Launch script created.
- [x] Explicit deployment completed.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This branch must not modify the frozen milestone directory or weights.

Runtime notes:

```text
branch-local smoke 6x728: reserved_gb=91.7871, avg_total_step_seconds=1.8740, estimated_steps_300min=9605
formal preflight batch_input_shape: (6, 3, 728, 728)
formal preflight total_params: 303912
formal preflight val_psnr: 14.3881
tmux session: 2dgauss_a_y03_uhddeblur_feature_refinement_plus_edge_weighted_lightedge_5h_confirmation
startup nvidia-smi: memory.used≈94067 MiB, utilization.gpu≈96%
live train_metrics snapshot: step=480 at 2026-05-16 09:04:40 CST
val_metrics.jsonl: not yet created; first validation pending step1000
```

## Confirmation Gate Follow-Up

- [x] Launch gate passed.
- [x] Checkpoints/logs/metrics are being written under the clean `A_y03`
      branch identity.
- [x] Medium-budget training clean finish completed.
- [x] Best training-time full-image validation PSNR collected.
- [x] Post-hoc official full-image test completed.

Interpretation:

- `A_y03` is not a new structure-search branch; it is a strict confirmation
  replay of `A_x03`.
- `A_y03` has now completed the key confirmation task successfully: it
  reproduces and materially improves the official margin over frozen
  `A_k03`.

Completion summary:

```text
training clean finish: step=21444, elapsed_minutes=299.9066
checkpoint relation: step_021000.pt == latest.pt; final.pt is distinct
best training-time val_psnr: 29.3667 at step15000
lowest training-time val_loss: 0.0210560 at step21000
posthoc official test final psnr: 26.2727
delta vs frozen A_k03 final: +0.0727 dB
delta vs A_x03 final: +0.0650 dB
```
