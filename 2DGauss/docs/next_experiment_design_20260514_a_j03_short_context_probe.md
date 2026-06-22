# Next Experiment Design: V1/T01/A_j03

## Record Time

- Time: `2026-05-14 09:10:00 CST`

## Position In The Line

`A_j03` is the first short-budget screening branch after the new frozen
baseline:

1. accepted and frozen baseline `A_i03/M02`;
2. user instruction to short-train first, then upgrade the next long run.

It does not replace `A_i03`; it exists to decide how the next longer run
should branch from `A_i03`.

## Why A Short Probe First

`A_i03` was a combined success:

1. larger consistency context than the earlier short probe;
2. longer optimization budget;
3. slightly larger effective training split.

That makes it a strong baseline, but not a clean attribution study.

Before spending another longer budget, the next best move is to isolate one
high-value variable with a short probe. The chosen variable is:

```text
more training context under the same VRAM window
```

## Chosen Adjustment

The new branch is:

```text
V1/T01/A_j03
```

It keeps from `A_i03`:

- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- effective train split after the first `40` pairs
- official full-image validation/test protocol
- reserved VRAM target window `85-93G`

It changes:

- tier: `medium -> short`
- wall-clock: `300 -> 58` minutes
- max steps target: `12000 -> 2500`
- training context target: `784 -> 864`
- batch-size hypothesis: `7 -> 6`

## What This Branch Tests

This branch asks a very specific question:

> if we reallocate memory budget away from batch and toward a larger training
> context, does the post-`A_i03` consistency line show a promising short-run
> signal?

This is meant to guide the next longer run, not to replace the frozen baseline
directly on its own.

## Why This Variable Comes First

Among the plausible next ablations after `A_i03`, this one is the most useful
for the next long-run decision because:

1. it stays on the now-validated consistency mainline;
2. it does not mix in another data-split change;
3. it can be deployed quickly with only config-level changes;
4. if it is positive, the next long run gets a cleaner upgrade target than
   simply “train `A_i03` longer again”.

## Deployment Expectation

Before launch, this branch should:

1. run serial smoke calibration around `batch_size=6`;
2. verify that the real selected setting remains inside `85-93G`;
3. run formal preflight;
4. then launch in tmux as the active short probe.
