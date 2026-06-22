# Next Experiment Design: V1/T01/A_ab03

## Record Time

- Time: `2026-05-17 00:45:00 CST`

## Position In The Line

`A_ab03` is prepared after:

1. frozen official deblur baseline `M04/A_y03`;
2. completed short positive probe `A_z03`;
3. completed medium promotion attempt `A_aa03`, which proved that decoder self-detail hinting is not enough by itself.

## Why This Branch Exists

`A_aa03` left one specific bottleneck exposed:

> the current Gaussian feature refinement line still lacks an explicit
> high-resolution structure signal that tells it where detail geometry is
> actually missing.

The next useful probe should therefore not be another decoder self-hint
variant. It should test whether Gaussian feature refinement improves when it
can see a compact edge descriptor derived from the high-resolution input.

## Chosen Adjustment

The new branch is:

```text
V1/T01/A_ab03
```

It keeps the accepted `A_y03/M04` short envelope fixed:

- decoder guidance on, `strength = 0.15`
- feature refinement on, `strength = 0.10`
- edge-weighted reconstruction on, `strength = 0.02`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- `batch_size=6`, `train_context_size=728`
- short-tier `58`-minute probe budget

It changes only one mechanism:

- disable `gaussian_feature_refinement_use_decoder_detail_hint`
- enable `gaussian_feature_refinement_use_edge_aware_prior`

## What This Branch Tests

`A_ab03` asks:

> if Gaussian feature refinement receives a compact explicit edge descriptor
> instead of decoder self-detail hints, can it place refinement pressure more
> accurately and outperform the post-`M04` short continuation family?

## Mechanism Intuition

The current feature refinement path already sees:

- Gaussian attention map
- local contrast
- directional prior

`A_ab03` adds one more prior family:

- gray structure
- `x/y` edge responses
- gradient magnitude
- local Laplacian/high-pass edge energy

This keeps the model lightweight, but shifts the extra signal from
"decoder self-awareness" to "explicit high-resolution geometry."

## Promotion Logic

If `A_ab03` becomes the strongest completed short result under the same
runtime envelope, it should become the highest-priority promotion candidate
for the next medium-budget or long-budget run. If it is flat or negative,
then the next probe should move to confidence-normalized refinement rather than
revisiting more edge-aware variants immediately.
