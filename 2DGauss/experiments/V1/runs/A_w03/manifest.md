# Run Manifest: V1/T01/A_w03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_w03`
- Full identifier: `V1/T01/A_w03`

## Current Status

This branch is prepared-only. No smoke, formal preflight, tmux launch,
training, checkpoint acceptance, or official evaluation has been accepted yet.

## Purpose

`A_w03` is the rebalanced continuation probe prepared after `A_t03`.

The immediate question is:

> if the best continuation needs more mechanism-side pressure and less
> supervision-side pressure at the same time, can that rebalance produce the
> strongest short-run result in this family?

## Experimental Question

Can a coordinated rebalance outperform the two single-axis probes while still
staying inside the same lightweight short-run budget?

More concretely:

1. keep the `A_t03` branch structure fixed;
2. raise the feature-refinement contribution;
3. lower the edge-weighted supervision contribution;
4. test whether the family winner needs both adjustments together.

## Chosen Mechanism

`A_w03` keeps:

```text
use_gaussian_feature_refinement = true
use_edge_weighted_reconstruction = true
```

and changes:

```text
gaussian_feature_refinement_strength = 0.12
edge_weighted_reconstruction_strength = 0.02
```

## Baseline For Comparison

- Primary short anchor inside this family: `V1/T01/A_t03`
- Earlier short anchor: `V1/T01/A_n03`
- Companion probes in this rebalance family: `V1/T01/A_u03`, `V1/T01/A_v03`
- Frozen milestone: `experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline`

## Planned Protocol

Kept from `A_t03`:

- dataset: `UHD_deblur`
- effective training split: train pairs after the first `40`
- internal validation: first `20` train pairs
- validation crop: full image, `crop_size=0`
- official test: `300` full-image pairs, post-hoc only
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- decoder guidance `strength = 0.15`
- feature refinement enabled
- edge-weighted reconstruction enabled
- prepared geometry target: `6x728`
- short-tier wall-clock target

Changed from `A_t03`:

- `gaussian_feature_refinement_strength: 0.10 -> 0.12`
- `edge_weighted_reconstruction_strength: 0.03 -> 0.02`

## Planned Runtime Envelope

- Tier target: short
- Wall-clock target: `58` minutes
- Max steps target: `2500`
- Initial batch-size hypothesis: `6`
- Gradient accumulation target: `1`
- Train context size: `728`
- Train supervision size: `512`
- Validation interval target: `500`
- Checkpoint interval target: `250`
- Validation batches target: `20`
- Target reserved VRAM window: `85.0-93.0G`

## Verification Planned

Planned, not yet accepted:

1. branch-local smoke calibration on the true `A_w03` config;
2. formal `train_v1_t01.py --preflight` on the run-local config;
3. clean-GPU serial launch validation;
4. short-budget training only if both gates pass;
5. post-hoc official full-image test only after clean finish.

## Working Hypothesis

If the post-`A_q03` family needs more mechanism-side pressure but slightly less
loss-side pressure, `A_w03` should emerge as the strongest short continuation.
If it brings no gain over `A_t03`, then the family is likely truly saturating.
