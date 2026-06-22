# Run Manifest: V1/T01/A_x03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_x03`
- Full identifier: `V1/T01/A_x03`

## Current Status

This branch has now completed smoke calibration, formal preflight, formal
tmux launch, full training, and official post-hoc evaluation. It is the
completed `5`-hour medium-budget promotion run for the currently best
completed post-`A_t03` rebalance result.

## Purpose

`A_x03` exists to answer the next clean promotion question:

> if we keep the strongest currently completed rebalance structure (`A_u03`)
> fixed and give it a full `5`-hour window under a safe high-VRAM geometry,
> can the lighter-edge feature-refinement line recover a meaningful part of
> the large gap back to `A_p03` and frozen `A_k03/M03`?

## Main Design Choice

Relative to `A_u03`, this branch keeps:

- `gaussian_decoder_guidance_strength = 0.15`
- `gaussian_feature_refinement_strength = 0.10`
- `edge_weighted_reconstruction_strength = 0.02`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- effective `UHD_deblur` training split after the first `40` pairs
- lightweight width/depth and Gaussian-first architecture story

It changes:

- training tier `short -> medium`
- wall-clock cap `58 -> 300` minutes
- max steps `2500 -> 24000`
- accepted launch geometry `6x728` after fresh branch-local smoke calibration

## Comparison Anchor

- Frozen milestone:
  `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`
- Stable milestone link:
  `experiments/V1/milestones/current_uhddeblur_official_baseline`
- Promotion winner:
  `experiments/V1/runs/A_u03`
- Officially closed stronger-feature comparison:
  `experiments/V1/runs/A_v03`

## Runtime Envelope

- Tier target: `medium`
- Planned wall-clock cap: `300` minutes
- Planned max steps: `24000`
- Accepted batch size: `6`
- Accepted training context: `728`
- Planned supervision crop: `512`
- Target reserved VRAM window: `85.0-93.0G`

## Deployment Rule

This branch has now passed:

1. clean-GPU branch-local smoke calibration;
2. formal preflight on the final run-local config;
3. tmux launch and startup-window verification;
4. milestone isolation checks.

## Calibration Notes

Fresh branch-local smoke on the true `A_x03` config shows:

- `6x728`: `reserved_gb=91.7871`, accepted
- `6x736`: `reserved_gb=93.7695`, over the configured window

Therefore the accepted medium-run launch point is `batch_size=6`,
`train_context_size=728`.

## Outcome Summary

`A_x03` completed successfully and produces the current best observed deblur
result under the local protocol:

- clean finish at `step=21441`
- final logged elapsed time `299.9925` minutes
- best training-time full-image validation `29.5140 dB` at `step=19000`
- post-hoc official full-image test `26.2077 dB` at `final.pt`

Relative to key same-task anchors:

- `A_x03 final.pt` is about `+0.1990 dB` above `A_p03 final.pt`
- `A_x03 final.pt` is about `+0.0077 dB` above frozen `A_k03 final.pt`
- `A_x03 final.pt` is about `+0.8976 dB` above `A_u03 final.pt`

The branch is therefore a real positive continuation result. But because the
margin over frozen `A_k03` is only about `+0.0077 dB`, it should be treated as
the current best observed result rather than an automatic milestone refresh.
