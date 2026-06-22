# Next Experiment Design: V1/T01/A_z03

## Record Time

- Time: `2026-05-16 15:05:00 CST`

## Position In The Line

`A_z03` is prepared after:

1. frozen official deblur baseline `M04/A_y03`;
2. confirmed successful light-edge feature-refinement line;
3. explicit success analysis showing the right current direction is
   feature-space Gaussian guidance plus light edge supervision.

## Why This Branch Exists

`M04/A_y03` already answered the replay question. The next useful question is
no longer “is this line real?”, but:

> can we improve high-resolution terminal detail expression further without
> abandoning the current lightweight successful recipe?

The narrowest plausible upgrade is not a heavier backbone and not a new loss
term. It is a slightly better prior interface for the existing
`GaussianGuidedFeatureRefinement`.

## Chosen Adjustment

The new branch is:

```text
V1/T01/A_z03
```

It keeps the accepted `A_y03/M04` structure fixed:

- decoder guidance on, `strength = 0.15`
- feature refinement on, `strength = 0.10`
- edge-weighted reconstruction on, `strength = 0.02`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- `batch_size=6`, `train_context_size=728`
- short-tier `58`-minute probe budget

It changes only one mechanism:

- enable `gaussian_feature_refinement_use_decoder_detail_hint = true`

## What This Branch Tests

`A_z03` asks:

> if Gaussian feature refinement can see a very cheap summary of the current
> decoder feature energy and local detail deficit, can it place its refinement
> pressure more accurately and outperform the plain `A_u03`/`A_y03` style
> feature-refinement interface?

## Mechanism Intuition

Current feature refinement already uses:

- Gaussian attention map
- Gaussian local contrast
- directional prior

But it does **not** know whether the current decoder feature at a location
already contains enough detail energy or is still under-expressed.

The new hint injects only two extra single-channel priors:

1. decoder energy mean
2. decoder local detail residual

This should let the Gaussian prior act more like a “where detail is still
missing” controller, without creating a new heavyweight branch.

## Promotion Logic

If `A_z03` beats `A_u03` clearly on short official test while preserving the
same runtime envelope, it becomes the next highest-confidence promotion anchor
above the old short rebalance family. If it is flat or negative, the project
learns that this level of self-conditioned feature refinement is not the
highest-value next knob, and future work should move to a different lightweight
detail-expression redesign rather than more local tweaks.
