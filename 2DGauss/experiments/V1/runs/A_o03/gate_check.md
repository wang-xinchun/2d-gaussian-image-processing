# Gate Check: V1/T01/A_o03

Status: gate completed; training finished; post-hoc official test completed.

This branch is the high-fill launch-shape scout in the post-`M03` short-probe
family.

## Training Duration Tier

- [x] Tier target: short training.
- [x] Planned wall-clock cap: `58` minutes.
- [x] Purpose: test whether the new decoder-guidance branch can stay close to
      the accepted `A_k03` deployment geometry while remaining inside the real
      `85-93G` runtime window.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_o03`.
- [x] Full run identifier: `V1/T01/A_o03`.
- [x] Run folder: `experiments/V1/runs/A_o03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config will be created automatically by the training entrypoint.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Baseline protocol anchor: frozen `A_k03/M03`.
- [x] Main experimental variable: high-fill launch shape with
      `batch=7`, `context=776`.
- [x] Decoder-guidance strength remains at the default `0.15`.
- [x] Edge-guided refinement remains enabled.
- [x] Directional/frequency guidance remains enabled.
- [x] Train-inference consistency remains enabled.

## Runtime Check

- [x] Planned batch size: `7`.
- [x] Initial planned train context size: `776`.
- [x] Planned supervision size: `512`.
- [x] Planned validation interval: `500`.
- [x] Planned checkpoint interval: `250`.
- [x] Target reserved VRAM window recorded: `85.0-93.0G`.
- [x] Clean GPU confirmed immediately before accepted smoke calibration.
- [x] Branch-specific smoke calibration completed.
- [x] Accepted smoke result inside `85.0-93.0G`.
- [x] Formal preflight completed.
- [x] Short-budget training completed cleanly.
- [x] Post-hoc official full-image test completed.

Calibration notes:

```text
batch7 context776: immediate OOM, reserved_gb=93.9688
batch7 context772: immediate OOM, reserved_gb=93.6875
batch7 context768: OOM after 5 steps, reserved_gb=94.2813
batch7 context760: ok but over upper bound, reserved_gb=93.9082
batch7 context752: ok but over upper bound, reserved_gb=93.5742
batch7 context748: accepted, reserved_gb=92.5859
batch7 context744: accepted, reserved_gb=91.6055
preferred near-A_k03 launch candidate: batch7 context748
real startup memory observed around 95.5G, above the practical 93G ceiling
```

Training summary:

```text
formal preflight batch_input_shape: (7, 3, 748, 748)
formal preflight total_params: 267864
formal preflight val_psnr: 21.7611
training clean finish: step=2500
training elapsed minutes: 34.9640
training-time best val psnr: 28.8394 at step2500
posthoc official test final psnr: 25.2809
```

## Promotion Gate Follow-Up

- [x] Completion gate passed.
- [x] Checkpoints and metrics written through the full short budget.
- [x] Best training-time full-image validation PSNR reached `28.8394 dB`.
- [x] This exceeds the family strong-keep threshold `28.80 dB`.
- [x] Post-hoc official full-image test corroboration collected.
- [x] Branch recorded as runtime-envelope evidence rather than the preferred
      `5`-hour anchor.

Interpretation:

- `A_o03` is a successful high-fill runtime scout.
- It does not outrank `A_n03` for promotion, and its real startup memory makes
  it a riskier long-run deployment anchor.

## Milestone Isolation Check

- [x] Frozen milestone remains
      `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`.
- [x] This prepared branch must not modify frozen milestone directories or
      weights.
