# Next Experiment Design: V1/T01/A_i03

## Record Time

- Time: `2026-05-14 01:20:00 CST`

## Position In The Line

`A_i03` is the medium-budget continuation of the post-milestone consistency
line after:

1. accepted and frozen baseline `A_g03`;
2. prepared short-budget consistency branch `A_h03`.

It does **not** replace `A_h03`; instead it forks a new run identity so the
short and medium variants remain auditable as separate hypotheses.

## Why A New Branch Instead Of Editing A_h03

`A_h03` was prepared as a short design package for a narrow question:

> does a `TLC`-style context/supervision mismatch repair help at all?

The new user request changes several dimensions at once:

1. training duration expands from short to `5` hours;
2. the memory gate is tightened from loose high-VRAM acceptance to a target
   window of `85-93G`;
3. the effective training split should be enlarged.

Those changes are large enough that they should be isolated under a fresh run
ID instead of silently mutating the existing `A_h03` design package.

## Design Summary

`A_i03` keeps:

- `A_g03` edge-guided + directional/frequency mainline;
- `A_h03` train-inference consistency mechanism;
- official full-image validation/test protocol.

`A_i03` changes:

- tier: `short -> medium`
- wall-clock: `58 -> 300` minutes
- max steps target: `2500 -> 12000`
- effective training split: after first `100` pairs -> after first `40` pairs
- target reserved VRAM window: explicit `85.0-93.0G`

## Dataset Choice

Current `UHD_deblur` train size is `1964` pairs.

Accepted short-run convention:

- validation prefix / held-out prefix: first `100`
- effective train size: `1864`

New medium-budget choice:

- keep deterministic validation prefix anchored at the start of the train set;
- reduce the hold-out boundary from `100` to `40`;
- keep active runtime validation at first `20` pairs;
- enlarge the effective training set to `1924`.

This is intentionally conservative: we are not collapsing validation into the
full train set, but we are letting the longer run use more of the data than the
short screening branches.

## Runtime Strategy

The calibrated branch should satisfy **both**:

1. reserved VRAM not below `85G`;
2. reserved VRAM not above `93G`.

This is stricter than the older `>=80G` rule and should be checked directly in
smoke output rather than inferred loosely from batch size.

The calibration helper was therefore extended so future smoke runs can report:

- target wall-clock minutes
- estimated steps at that wall-clock
- reserved-memory window pass/fail

## What This Branch Tests

This branch is not only “longer A_h03”.

It jointly tests whether:

1. the consistency mechanism benefits from more optimization time;
2. the mechanism benefits from a slightly larger effective train split;
3. the same idea can still live inside a strict but practical `85-93G`
   deployment envelope.

## What Is Still Intentionally Not Done

This turn does **not**:

- run smoke calibration;
- run formal preflight;
- create a launch script;
- launch training;
- touch the frozen milestone package.

## Decision Expectation For Next Deployment Turn

Before deployment, the next turn should:

1. run serial smoke calibration around `batch_size=8`;
2. search the first batch size that lands inside `85-93G`;
3. record estimated steps at `300` minutes;
4. run formal preflight;
5. only then decide whether `A_i03` should be launched as the next active
   medium-budget `UHD_deblur` branch.
