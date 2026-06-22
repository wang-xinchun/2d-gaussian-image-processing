# Run Manifest: V1/T01/A_y03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_y03`
- Full identifier: `V1/T01/A_y03`

## Current Status

This branch has now completed smoke calibration, formal preflight, formal
tmux launch, full training, and official post-hoc evaluation. It is the
completed confirmation replay for `A_x03`, after `A_x03` produced the current
best observed `UHD_deblur` result but only by a very small margin over frozen
milestone `M03/A_k03`.

## Purpose

`A_y03` exists to answer the milestone-refresh confirmation question:

> if we rerun the exact `A_x03` structure under the same five-hour budget with
> a fresh seed and fresh run identity, does the light-edge feature-refinement
> line confirm a real margin over frozen `M03/A_k03`, or does the previous
> `+0.0077 dB` edge collapse back into noise?

## Main Design Choice

Relative to `A_x03`, this branch keeps:

- `gaussian_decoder_guidance_strength = 0.15`
- `gaussian_feature_refinement_strength = 0.10`
- `edge_weighted_reconstruction_strength = 0.02`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- effective `UHD_deblur` training split after the first `40` pairs
- accepted `6x728` medium-budget launch geometry
- `300`-minute runtime tier

It changes:

- run identity `A_x03 -> A_y03`
- random seed `20260516 -> 20260517`

## Comparison Anchor

- Frozen milestone:
  `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`
- Stable milestone link:
  `experiments/V1/milestones/current_uhddeblur_official_baseline`
- Best observed current result:
  `experiments/V1/runs/A_x03`

## Runtime Envelope

- Tier target: `medium`
- Planned wall-clock cap: `300` minutes
- Planned max steps: `24000`
- Accepted batch size target: `6`
- Accepted training context target: `728`
- Planned supervision crop: `512`
- Target reserved VRAM window: `85.0-93.0G`

## Deployment Rule

This branch has now passed:

1. clean-GPU branch-local smoke recheck;
2. formal preflight on the run-local config;
3. tmux launch and startup-window verification;
4. milestone isolation checks.

## Calibration Notes

Fresh branch-local smoke on the true `A_y03` config shows:

- `6x728`: `reserved_gb=91.7871`, accepted

Formal preflight on the final run-local config shows:

- `batch_input_shape=(6, 3, 728, 728)`
- `batch_target_shape=(6, 3, 728, 728)`
- `total_params=303912`
- `val_psnr=14.3881`

## Outcome Summary

`A_y03` completed successfully and confirms the `A_x03` line under a fresh
seed:

- clean finish at `step=21444`
- final logged elapsed time `299.9066` minutes
- best training-time validation PSNR `29.3667 dB` at `step=15000`
- lowest training-time validation loss `0.0210560` at `step=21000`
- post-hoc official full-image test `26.2727 dB` at `final.pt`

Relative to key same-task anchors:

- `A_y03 final.pt` is about `+0.0650 dB` above `A_x03 final.pt`
- `A_y03 final.pt` is about `+0.26399 dB` above `A_p03 final.pt`
- `A_y03 final.pt` is about `+0.0727 dB` above frozen `A_k03 final.pt`

The branch therefore upgrades the previous replay question from “is the tiny
`A_x03` edge real?” to “the line has now been confirmed as reproducibly
better than frozen `A_k03` under the same protocol.”
