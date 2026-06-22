# Run Manifest: V1/T01/A_v03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_v03`
- Full identifier: `V1/T01/A_v03`

## Current Status

This branch is prepared-only. No smoke, formal preflight, tmux launch,
training, checkpoint acceptance, or official evaluation has been accepted yet.

## Purpose

`A_v03` is the stronger-feature continuation probe prepared after `A_t03`.

The immediate question is:

> if `A_t03` is structurally right but slightly underpowered on the mechanism
> side, can a modestly stronger feature-refinement stage improve the short-run
> endpoint without destabilizing the lightweight budget?

## Experimental Question

Can a slightly stronger feature-refinement stage convert the weakly positive
`A_t03` signal into a more trustworthy short-run gain while keeping the rest
of the family unchanged?

More concretely:

1. keep the `A_t03` branch story fixed;
2. keep the edge-weighted supervision unchanged;
3. strengthen only the feature-refinement contribution;
4. test whether the remaining gain is mechanism-limited rather than
   supervision-limited.

## Chosen Mechanism

`A_v03` keeps:

```text
use_gaussian_feature_refinement = true
use_edge_weighted_reconstruction = true
edge_weighted_reconstruction_strength = 0.03
```

and changes only:

```text
gaussian_feature_refinement_strength = 0.12
```

## Baseline For Comparison

- Primary short anchor inside this family: `V1/T01/A_t03`
- Earlier short anchor: `V1/T01/A_n03`
- Companion probes in this rebalance family: `V1/T01/A_u03`, `V1/T01/A_w03`
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
- edge-weighted reconstruction enabled at `0.03`
- prepared geometry target: `6x728`
- short-tier wall-clock target

Changed from `A_t03`:

- `gaussian_feature_refinement_strength: 0.10 -> 0.12`

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

1. branch-local smoke calibration on the true `A_v03` config;
2. formal `train_v1_t01.py --preflight` on the run-local config;
3. clean-GPU serial launch validation;
4. short-budget training only if both gates pass;
5. post-hoc official full-image test only after clean finish.

## Working Hypothesis

If the post-`A_q03` family is still mechanism-limited, `A_v03` should be the
highest-upside short continuation. If it destabilizes memory or regresses
official quality, then the `A_t03` feature-refinement strength was already near
the practical limit for this lightweight branch.
