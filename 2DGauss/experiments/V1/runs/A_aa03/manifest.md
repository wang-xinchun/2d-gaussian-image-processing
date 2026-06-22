# Run Manifest: V1/T01/A_aa03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_aa03`
- Full identifier: `V1/T01/A_aa03`

## Current Status

This branch has now completed branch-local smoke calibration, formal
preflight, formal tmux launch, full medium-budget training, and same-protocol
official post-hoc evaluation. It is now a closed medium-promotion probe for
completed short leader `A_z03`, and it did not surpass frozen `M04/A_y03`.

## Purpose

`A_aa03` exists to answer the next mainline promotion question:

> if we keep the full `M04/A_y03` medium recipe fixed and add only the
> `A_z03` decoder-detail-aware feature-refinement hint, can that short-run
> local gain compound into a meaningful medium-budget absolute-performance
> improvement?

## Main Design Choice

Relative to frozen `M04/A_y03`, this branch keeps:

- `gaussian_decoder_guidance_strength = 0.15`
- `gaussian_feature_refinement_strength = 0.10`
- `edge_weighted_reconstruction_strength = 0.02`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- effective `UHD_deblur` training split after the first `40` pairs
- accepted medium-budget `6x728` geometry
- `300`-minute runtime tier

It changes only one mechanism:

- `gaussian_feature_refinement_use_decoder_detail_hint = true`

## Comparison Anchor

- Frozen milestone:
  `experiments/V1/milestones/M04_A_y03_uhddeblur_official_baseline`
- Stable milestone link:
  `experiments/V1/milestones/current_uhddeblur_official_baseline`
- Promotion source:
  `experiments/V1/runs/A_z03`
- Earlier medium references:
  `experiments/V1/runs/A_x03`
  `experiments/V1/runs/A_y03`

## Runtime Envelope

- Tier target: `medium`
- Planned wall-clock cap: `300` minutes
- Planned max steps: `24000`
- Planned batch size target: `6`
- Planned train context target: `728`
- Planned supervision crop: `512`
- Target reserved VRAM window: `85.0-93.0G`

## Deployment Rule

This branch has now passed:

1. clean-GPU branch-local smoke on the true `A_aa03` config;
2. formal preflight on the final run-local config;
3. tmux launch and startup-window verification;
4. milestone isolation recheck against frozen `M04/A_y03`.

## Working Hypothesis

`A_z03` already proved that the decoder-detail-aware hint is a real positive
short-run local improvement. The remaining question is whether that gain only
reorders the short family, or whether it scales into the five-hour regime and
reduces the still-large gap to `M04/A_y03`.

Acceptance boundary:

- If `A_aa03` beats `A_z03`-free medium anchors and narrows the gap to
  `M04/A_y03`, the hint graduates from short-run clue to serious medium-budget
  promotion mechanism.
- If `A_aa03` stays clearly below `M04/A_y03`, the correct interpretation is
  not “the hint is useless,” but “this hint alone cannot close the medium-line
  absolute-performance gap.”

## Calibration Notes

Branch-local smoke on the true `A_aa03` config shows:

- `6x728`: `reserved_gb=91.8105`, accepted
- `avg_total_step_seconds=1.9587`
- estimated `300`-minute steps: `9189`

Formal preflight on the final run-local config shows:

- `batch_input_shape=(6, 3, 728, 728)`
- `batch_target_shape=(6, 3, 728, 728)`
- `total_params=304776`
- `val_psnr=20.9476`

These gate results keep the branch inside the same lightweight and practical
VRAM envelope as `A_z03`, while promoting the runtime budget to the five-hour
medium tier.

Completion status now shows:

- tmux session:
  `2dgauss_a_aa03_uhddeblur_feature_refinement_detailhint_5h_promotion`
- training log:
  `experiments/V1/runs/A_aa03/logs/train_20260516_161430.log`
- clean finish at `step=21256`
- final logged elapsed time: `299.9259` minutes
- latest live `nvidia-smi` before finish: `memory.used≈94615 MiB`,
  `utilization.gpu=100%`
- checkpoints written through `step_021000.pt`, `latest.pt`, and `final.pt`
- `step_021000.pt` and `latest.pt` share identical `model` weights
- `final.pt` is a distinct true endpoint
- best training-time validation PSNR:
  `29.4790 dB @ step=19000`
- lowest training-time validation loss:
  `0.0209086 @ step=19000`
- same-protocol official full-image test on `final.pt`:
  `26.0850 dB`

Relative to key anchors:

- `A_aa03 final.pt` is about `-0.1877 dB` below frozen `M04/A_y03 final.pt`
- `A_aa03 final.pt` is about `-0.1227 dB` below `A_x03 final.pt`
- `A_aa03 final.pt` is about `+0.0763 dB` above `A_p03 final.pt`
- `A_aa03 final.pt` is about `+0.6993 dB` above promoted short source
  `A_z03 final.pt`
