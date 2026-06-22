# Gate Check: V1/T01/A_u03

Status: gate completed; training finished; post-hoc official test completed.

This branch exists to test whether the slight positive signal in `A_t03` gets
clearer when the supervision-side edge-weighted pressure is reduced.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test a lighter supervision rebalance around `A_t03`.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_u03`.
- [x] Full run identifier: `V1/T01/A_u03`.
- [x] Run folder: `experiments/V1/runs/A_u03`.
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
- [x] Companion rebalance probes: `A_v03` and `A_w03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction remains enabled.
- [x] Only branch-specific change is `edge_weighted_reconstruction_strength = 0.02`.
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

## Static Checks Completed

- [x] Template config identity check.
- [x] Run-local config identity check.
- [x] True-config formal preflight completed.

Runtime notes:

```text
branch-local smoke 6x728: reserved_gb=91.7871, avg_total_step_seconds=1.8851, estimated_steps_58min=1846
formal preflight batch_input_shape: (6, 3, 728, 728)
formal preflight total_params: 303912
formal preflight val_psnr: 22.3029
real startup memory.used≈94067 MiB (slightly above original practical 93G ceiling)
training clean finish: step=2500
training elapsed minutes: 35.6395
training-time best val psnr: 28.7799 at step2500
posthoc official test final psnr: 25.3101
```

## Promotion Gate Follow-Up

- [x] Completion gate passed.
- [x] Checkpoints and metrics written through the full short budget.
- [x] Best training-time full-image validation PSNR collected.
- [x] Post-hoc official full-image test corroboration collected.

Interpretation:

- `A_u03` does not beat `A_t03` on internal final validation.
- But it does improve same-protocol official testing by about `+0.0156 dB`,
  which is the clearest positive result so far inside this rebalance family.
- The current family leader is therefore no longer `A_t03`; it is now
  `A_u03`, pending the remaining `A_v03` and `A_w03` completions.

## Milestone Isolation Check

- [x] Current frozen baseline remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This prepared branch does not modify the frozen milestone directory or
      its weights.
