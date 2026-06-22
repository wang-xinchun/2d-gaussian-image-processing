# Gate Check: V1/T01/A_p03

Status: gate completed; medium-budget launch completed; run completed successfully.

This branch is the first `5`-hour promotion run for the completed
Gaussian-guided decoder-modulation short-scout family.

## Training Duration Tier

- [x] Tier target: medium training.
- [x] Planned wall-clock cap: `300` minutes.
- [x] Purpose: promote the winning `A_n03` structure into a `5`-hour run under
      a safe high-VRAM geometry.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_p03`.
- [x] Full run identifier: `V1/T01/A_p03`.
- [x] Run folder: `experiments/V1/runs/A_p03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config will be created automatically by the training entrypoint.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Baseline protocol anchor: frozen `A_k03/M03`.
- [x] Promotion anchor: completed short-run winner `A_n03`.
- [x] Main experimental variable: medium-budget promotion of default-strength
      Gaussian-guided decoder modulation.
- [x] `gaussian_decoder_guidance_strength = 0.15`.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.

## Runtime Check

- [x] Planned batch size: `6`.
- [x] Planned train context size: `800`.
- [x] Planned supervision size: `512`.
- [x] Planned validation interval: `1000`.
- [x] Planned checkpoint interval: `500`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Clean GPU confirmed immediately before accepted smoke calibration.
- [x] Branch-specific smoke calibration completed.
- [x] Smoke result inside `85.0-93.0G`.
- [x] Formal preflight completed.
- [x] Medium-budget launch completed.

Calibration notes:

```text
candidate geometry from branch-local smoke:
batch6 context792: reserved_gb=89.5762
batch6 context800: reserved_gb=91.3105
selected launch target: batch6 context800
selection reason: larger safe context than short-probe 6x784, while avoiding
the real-startup overflow observed on A_o03 batch7 high-fill geometry
```

Launch summary:

```text
formal preflight batch_input_shape: (6, 3, 800, 800)
formal preflight total_params: 267864
formal preflight val_psnr: 21.6963
tmux session: 2dgauss_a_p03_uhddeblur_gaussian_decoder_guidance_default_5h
startup nvidia-smi: memory.used=93751 MiB, utilization.gpu=100%
train_metrics first confirmed step: 60
```

## Promotion Gate Context

- [x] `A_n03` won the short-run family by both best validation signal and best
      corroborative official test.
- [x] `A_o03` is recorded as deployment-envelope evidence, not the promotion
      winner.

## Milestone Isolation Check

- [x] Frozen milestone remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] This branch must not modify frozen milestone directories or weights.
