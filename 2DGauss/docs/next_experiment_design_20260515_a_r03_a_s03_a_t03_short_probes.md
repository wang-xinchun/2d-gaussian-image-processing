# Next Experiment Design: V1/T01/A_r03 + A_s03 + A_t03

## Record Time

- Time: `2026-05-15 12:26:37 CST`

## Position In The Line

This short-probe family is prepared after:

1. frozen official deblur baseline `M03/A_k03`;
2. completed decoder-guidance short family `A_m03/A_n03/A_o03`;
3. completed medium follow-up `A_p03`;
4. completed negative terminal-control probe `A_q03`.

## Why These Probes Exist

The current question is no longer:

> does Gaussian help final reconstruction at all?

That is already answered.

The new question is:

> how should Gaussian help final reconstruction without becoming a direct
> terminal RGB writer?

`A_q03` already showed that the current direct terminal residual path is not
the right answer. So the next family should separate two remaining hypotheses:

1. the bottleneck is mainly **mechanism-side**:
   high-resolution feature expression is still too weak;
2. the bottleneck is partly **supervision-side**:
   edges and high-frequency regions are still under-weighted during training.

## Family Design

All three probes keep the same short-run anchor and change only one narrow
dimension at a time.

Shared anchor:

- `A_n03` short structure
- edge-guided refinement enabled
- directional/frequency guidance enabled
- train-inference consistency enabled
- decoder guidance `strength = 0.15`
- short-tier `6 x 784` geometry as the original design anchor

Three probes:

1. `A_r03`: feature refinement only
2. `A_s03`: edge-weighted supervision only
3. `A_t03`: feature refinement + edge-weighted supervision

Prepared geometry after initial verification:

- `A_s03` keeps `6 x 784`
- `A_r03` is tightened to `6 x 728`
- `A_t03` is tightened to `6 x 728`

Reason:

- branches that enable feature refinement exceed the memory window at `6 x 784`
  during true-config preflight, so they must first re-enter the `85-93G` band
  before deployment work continues.

## Probe Meanings

### `A_r03`

Question:

> if we move the final-detail intervention earlier, from RGB residual space into
> decoder feature space, do we get a cleaner positive signal?

Mechanism:

- `use_gaussian_feature_refinement = true`
- `gaussian_feature_refinement_strength = 0.10`

Interpretation:

- positive result: mechanism-side expression was the main bottleneck;
- negative result: simply moving the intervention earlier is not enough.

### `A_s03`

Question:

> if we keep the structure fixed and only change how strongly edges are
> emphasized in reconstruction loss, do we recover detail more effectively?

Mechanism:

- `use_edge_weighted_reconstruction = true`
- `edge_weighted_reconstruction_strength = 0.03`
- `edge_weighted_reconstruction_floor = 1.0`

Interpretation:

- positive result: optimization / supervision weighting is still part of the gap;
- negative result: the main issue is more likely architectural than loss-side.

### `A_t03`

Question:

> are the two ideas complementary, or do they over-correct together?

Mechanism:

- combines `A_r03` and `A_s03`

Interpretation:

- positive result: both structure and supervision contribute;
- unstable or weak result: the interventions are not automatically additive.

## Why This Family Is Better Than Replaying `A_q03`

These probes preserve the key lesson from `A_q03`:

- Gaussian should still answer `where / how much / how directional`;
- it should not directly answer the final RGB residual itself.

So the family keeps Gaussian as a structural prior and moves the intervention
either:

1. earlier into feature space; or
2. sideways into the supervision signal.

## Promotion Logic

This family is prepared for short-run screening only.

Promotion rule should remain conservative:

1. short-run branch must train cleanly inside the `85-93G` window;
2. full-image internal validation should stay at least competitive with the
   `A_n03` band;
3. official test remains post-hoc corroboration only;
4. only a clean positive short result should be promoted into a longer run.

## Additional Engineering Note

While preparing this family, the edge-weighted reconstruction implementation was
also normalized so that:

1. `torch.nn.functional as F` is imported explicitly;
2. `edge_weighted_reconstruction_strength` acts as a single global weight on
   the auxiliary term rather than being counted twice.

This keeps the new probes interpretable and avoids baking an avoidable loss bug
into the family.
