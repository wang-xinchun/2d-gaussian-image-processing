# Next Experiment Design: V1/T01/A_aa03

## Record Time

- Time: `2026-05-16 16:15:00 CST`

## Position In The Line

`A_aa03` is prepared after:

1. frozen official deblur baseline `M04/A_y03`;
2. completed short positive detail-hint probe `A_z03`;
3. milestone isolation verification on `M04`.

## Why This Branch Exists

`A_z03` answered the local short-tier question positively: the
decoder-detail-aware feature-refinement hint is not just runtime-safe, it
improves same-protocol official test over the nearby short anchors.

But `A_z03` also left the real mainline question unresolved:

> does this hint only reshuffle short probes, or can it still matter after a
> five-hour medium-budget promotion against the true baseline `M04/A_y03`?

## Chosen Adjustment

The new branch is:

```text
V1/T01/A_aa03
```

It keeps the accepted `A_y03/M04` medium recipe fixed:

- decoder guidance on, `strength = 0.15`
- feature refinement on, `strength = 0.10`
- edge-weighted reconstruction on, `strength = 0.02`
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- medium-tier `300`-minute budget
- `batch_size=6`, `train_context_size=728`

It changes only one mechanism:

- enable `gaussian_feature_refinement_use_decoder_detail_hint = true`

## What This Branch Tests

`A_aa03` asks:

> if we preserve the full successful `M04/A_y03` medium training recipe and
> promote only the validated `A_z03` hint, can that lightweight local prior
> interface reduce the still-large absolute gap to `M04/A_y03`?

## Promotion Logic

If `A_aa03` narrows the medium-line gap and approaches or exceeds `M04/A_y03`,
the detail-hint mechanism graduates from short-run clue to serious mainline
upgrade candidate.

If `A_aa03` remains clearly below `M04/A_y03`, then the correct lesson is:

- the hint is real but insufficient by itself at medium scale;
- future absolute-performance work should move to a new lightweight structure
  redesign rather than more short-tier local tweaks.
