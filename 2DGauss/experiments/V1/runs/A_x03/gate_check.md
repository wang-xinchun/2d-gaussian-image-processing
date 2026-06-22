# Gate Check: V1/T01/A_x03

Status: gate completed; medium-budget launch completed; run completed successfully.

This branch exists to promote the current rebalance-family leader `A_u03`
into a `5`-hour medium-budget run.

## Training Duration Tier

- [x] Tier target: medium training.
- [x] Planned wall-clock cap: `300` minutes.
- [x] Purpose: test whether the light-edge feature-refinement line benefits
      from long-budget optimization.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_x03`.
- [x] Full run identifier: `V1/T01/A_x03`.
- [x] Run folder: `experiments/V1/runs/A_x03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config exists.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Promotion anchor: completed `A_u03`.
- [x] Family verification counterexample: completed `A_v03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction remains enabled at `strength = 0.02`.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.

## Runtime Check

- [x] Accepted batch size: `6`.
- [x] Accepted train context size: `728`.
- [x] Planned supervision size: `512`.
- [x] Planned validation interval: `1000`.
- [x] Planned checkpoint interval: `500`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Branch-specific smoke calibration completed.
- [x] Final formal preflight completed on the accepted launch config.
- [x] Launch script created.
- [x] Explicit deployment completed.

Runtime notes:

```text
branch-local smoke 6x728: reserved_gb=91.7871, avg_total_step_seconds=1.9002, estimated_steps_300min=9472
branch-local smoke 6x736: reserved_gb=93.7695, over the configured 85-93G window
accepted launch target: batch_size=6, train_context_size=728
```

Launch summary:

```text
formal preflight batch_input_shape: (6, 3, 728, 728)
formal preflight total_params: 303912
formal preflight val_psnr: 16.7676
tmux session: 2dgauss_a_x03_uhddeblur_feature_refinement_plus_edge_weighted_lightedge_5h
startup nvidia-smi: memory.used≈94067 MiB, utilization.gpu=100%
training clean finish: step=21441, elapsed_minutes=299.9925
posthoc official test final psnr: 26.2077
```

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This branch must not modify the frozen milestone directory or weights.
