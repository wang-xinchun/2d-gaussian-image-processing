# Next Experiment Design: V1/T01/A_u03 + A_v03 + A_w03

## Record Time

- Time: `2026-05-15 23:13:33 CST`

## Position In The Line

This short-probe family is prepared after:

1. frozen official deblur baseline `M03/A_k03`;
2. completed decoder-guidance short family `A_m03/A_n03/A_o03`;
3. completed medium follow-up `A_p03`;
4. completed negative terminal-control probe `A_q03`;
5. completed post-`A_q03` short family `A_r03/A_s03/A_t03`.

## Why These Probes Exist

`A_t03` already established the best result inside the post-`A_q03` short
family, but only by a tiny margin over `A_n03`.

That means the next useful question is no longer:

> does the feature-refinement-plus-edge-weighted direction work at all?

That is already answered weakly in the positive.

The new question is:

> can a small rebalancing around `A_t03` convert its tiny positive signal into
> a more trustworthy short-run gain without changing the structure family?

## Family Design

All three probes keep the same structural story as `A_t03`:

- `A_n03`-derived short mainline
- edge-guided refinement enabled
- directional/frequency guidance enabled
- train-inference consistency enabled
- decoder guidance `strength = 0.15`
- feature refinement enabled
- edge-weighted reconstruction enabled
- prepared `6 x 728` geometry

Only the balance between mechanism-side and supervision-side pressure changes.

Three probes:

1. `A_u03`: lighter edge-weighted supervision
2. `A_v03`: stronger feature refinement
3. `A_w03`: stronger feature refinement + lighter edge-weighted supervision

## Probe Meanings

### `A_u03`

Question:

> if `A_t03` is slightly over-regularized by the added edge-weighted term, can
> reducing that loss-side pressure preserve the structural gain and improve the
> final metric?

Mechanism:

- keep `gaussian_feature_refinement_strength = 0.10`
- reduce `edge_weighted_reconstruction_strength = 0.03 -> 0.02`

### `A_v03`

Question:

> if `A_t03` is slightly underpowered on the mechanism side, can a modestly
> stronger feature refinement stage improve transfer to official testing?

Mechanism:

- increase `gaussian_feature_refinement_strength = 0.10 -> 0.12`
- keep `edge_weighted_reconstruction_strength = 0.03`

### `A_w03`

Question:

> if the best short continuation needs more mechanism-side pressure and less
> supervision-side pressure at the same time, does a rebalanced combination
> beat the single-axis adjustments?

Mechanism:

- increase `gaussian_feature_refinement_strength = 0.10 -> 0.12`
- reduce `edge_weighted_reconstruction_strength = 0.03 -> 0.02`

## Promotion Logic

These probes are still short-run screening only.

Promotion should remain conservative:

1. branch must train cleanly within the real startup-memory envelope;
2. internal full-image validation should at least remain competitive with
   `A_t03`;
3. official test remains post-hoc only;
4. only the strongest short result should be considered for any confirming
   medium-budget follow-up.
