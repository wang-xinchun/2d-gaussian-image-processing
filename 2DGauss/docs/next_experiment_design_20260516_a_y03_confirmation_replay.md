# Next Experiment Design: V1/T01/A_y03

## Record Time

- Time: `2026-05-16 08:53:47 CST`

## Position In The Line

`A_y03` is prepared after:

1. frozen official deblur baseline `M03/A_k03`;
2. completed rebalance-family leader `A_u03`;
3. completed medium promotion `A_x03`, which reached the current best observed
   result `26.2077 dB` but only by about `+0.0077 dB` over frozen `A_k03`.

## Why This Branch Exists

`A_x03` answered the structural promotion question positively, but did not yet
answer the milestone-refresh question strongly enough.

The remaining uncertainty is:

> is the tiny `A_x03 > A_k03` edge a stable consequence of the light-edge
> feature-refinement line, or just too small to trust without replay?

## Chosen Adjustment

The new branch is:

```text
V1/T01/A_y03
```

It keeps exactly the same structure and budget as `A_x03`:

- decoder guidance on, `strength = 0.15`
- feature refinement on, `strength = 0.10`
- edge-weighted reconstruction on, `strength = 0.02`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- `batch_size=6`, `train_context_size=728`
- `wall_clock_minutes=300`, `max_steps=24000`

It changes only:

- run identity
- seed: `20260516 -> 20260517`

## What This Branch Tests

`A_y03` asks the narrowest possible confirmation question:

> if we rerun the `A_x03` recipe under the same protocol and same envelope,
> does the official result still stay at or above frozen `A_k03`, and by a
> margin large enough to support milestone refresh?

## Promotion Logic

If `A_y03` confirms a stable or larger gain over `A_k03`, then the project can
consider freezing a new milestone package around this line. If it falls back
to `A_k03`-level or lower, then `A_x03` should remain only a best observed
result rather than a frozen replacement.
