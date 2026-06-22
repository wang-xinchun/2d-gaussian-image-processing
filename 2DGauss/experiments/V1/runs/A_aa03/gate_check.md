# Gate Check: V1/T01/A_aa03

Status: gate completed; medium-budget training finished; official post-hoc
evaluation completed.

This branch exists to promote the completed short leader `A_z03` into a
single medium-budget run above frozen `M04/A_y03`.

## Training Duration Tier

- [x] Tier target: medium training.
- [x] Planned wall-clock cap: `300` minutes.
- [x] Purpose: test whether the `A_z03` decoder-detail-aware hint compounds
      under the medium budget.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_aa03`.
- [x] Full run identifier: `V1/T01/A_aa03`.
- [x] Run folder: `experiments/V1/runs/A_aa03`.
- [x] Run folder is independent and does not overwrite prior runs.
- [x] Run-local config exists.

## Dataset Check

- [x] Dataset name: `UHD_deblur`.
- [x] Effective training selection: train pairs after the first `40`.
- [x] Validation selection: first `20` train pairs.
- [x] Official test split remains the `300` paired images in `test`.

## Mechanism Check

- [x] Promotion anchor: completed `A_z03`.
- [x] Frozen baseline for comparison: `M04/A_y03`.
- [x] Decoder guidance remains enabled at `strength = 0.15`.
- [x] Feature refinement remains enabled at `strength = 0.10`.
- [x] Edge-weighted reconstruction remains enabled at `strength = 0.02`.
- [x] Branch-specific change is
      `gaussian_feature_refinement_use_decoder_detail_hint = true`.
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
      `experiments/V1/milestones/M04_A_y03_uhddeblur_official_baseline`.
- [x] Stable milestone link remains
      `experiments/V1/milestones/current_uhddeblur_official_baseline`.
- [x] This branch must not modify the frozen milestone directory or weights.

Runtime notes:

```text
planned launch target: batch_size=6, train_context_size=728
planned wall_clock_minutes: 300
planned max_steps: 24000
branch-local smoke 6x728: reserved_gb=91.8105, avg_total_step_seconds=1.9587, estimated_steps_300min=9189
formal preflight batch_input_shape: (6, 3, 728, 728)
formal preflight batch_target_shape: (6, 3, 728, 728)
formal preflight total_params: 304776
formal preflight val_psnr: 20.9476
parameter delta vs M04/A_y03: +864
tmux session: 2dgauss_a_aa03_uhddeblur_feature_refinement_detailhint_5h_promotion
startup nvidia-smi: memory.used≈94507 MiB, utilization.gpu=100%, temperature≈45C, power.draw≈374.30W
live train_metrics snapshot: step=1 at 2026-05-16 16:14:41 CST
live train_metrics snapshot update: step=340 at 2026-05-16 16:19:21 CST
live elapsed_minutes snapshot: 4.7123
live loss_total snapshot: 0.04885
live checkpoint status: none yet, consistent with checkpoint_interval=500
live val_metrics status: not yet created, consistent with first val at step=1000
live train_metrics snapshot update: step=2550 at 2026-05-16 16:50:48 CST
live elapsed_minutes snapshot: 35.7780
live recent loss_total snapshot: 0.04498
live nvidia-smi update: memory.used≈94615 MiB, utilization.gpu=100%, temperature≈56C, power.draw≈375.30W
live checkpoints: step_000500.pt, step_001000.pt, step_001500.pt, step_002000.pt, step_002500.pt, latest.pt
live val step1000: val_loss=0.0247865, val_psnr=28.5406
live val step2000: val_loss=0.0245704, val_psnr=28.6234
training clean finish: step=21256
training elapsed minutes: 299.9259
completion checkpoint relation: step_021000.pt == latest.pt (model weights)
completion checkpoint relation: final.pt != step_021000.pt
best training-time val psnr: 29.4790 at step19000
lowest training-time val loss: 0.0209086 at step19000
posthoc official test final psnr: 26.0850
```

## Promotion Gate Follow-Up

- [x] Launch gate passed.
- [x] Checkpoints/logs/metrics are being written under the clean `A_aa03`
      branch identity.
- [x] Medium-budget training clean finish completed.
- [x] Best training-time full-image validation PSNR collected.
- [x] Post-hoc official full-image test completed.

Interpretation target:

- `A_aa03` should be judged first against `M04/A_y03`, not only against the
  nearby short-family anchors.
- The branch is intended to answer one narrow question: whether the validated
  `A_z03` hint still matters after promotion into the five-hour regime.

Outcome:

- `A_aa03` remains above the older medium anchor `A_p03`, so the promoted line
  is not a collapse.
- But it stays clearly below frozen `M04/A_y03` and also below `A_x03`.
- Therefore the correct conclusion is that
  `gaussian_feature_refinement_use_decoder_detail_hint = true` is a real
  short-run local positive, but it is insufficient by itself as a
  medium-budget upgrade mechanism over the accepted `M04/A_y03` line.
