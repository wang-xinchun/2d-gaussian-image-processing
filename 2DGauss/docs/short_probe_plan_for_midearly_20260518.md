# Short Probe Plan for Mid-Term Exploration

Time: 2026-05-18 22:20:00 CST

## Current Bottleneck

The current short-run ceiling is not a raw-cue shortage. The strongest evidence now says the model already has enough local signals:

- `A_aa03` showed decoder self-detail is useful but not enough.
- `A_ab03` showed direct edge-aware injection is runtime-safe but not a win.
- `A_ac03` showed confidence normalization alone is not enough.
- `A_ad03` showed shared-prior coupling is better than standalone confidence normalization.
- `A_ae03` showed shared-prior energy can improve confidence gating, but not enough to beat `A_z03`.

The core issue is therefore:

> Gaussian priors still do not form a sufficiently selective and coordinated control surface over decoder recovery.

## What Is Still Missing

The next short probes should test three distinct control failures:

1. decoder guidance does not yet look at the decoder state itself;
2. feature refinement does not yet separate confidence suppression from intervention demand;
3. edge refinement does not yet have a clear control role relative to the shared prior and decoder gate.

These are different enough to matter. They are not just three names for the same cue injection idea.

## Planned Short Probes

### `A_af03` - decoder-state closed loop

Hypothesis:

- decoder guidance should not be prior-only;
- it should also know whether the current decoder feature already carries enough detail;
- the gate should therefore depend on both shared prior and decoder state.

Expected signal:

- if this helps, the model is missing closed-loop control, not just more signal.

### `A_ag03` - intervention demand

Hypothesis:

- confidence normalization alone only tells the model where to suppress;
- the missing half is where to explicitly ask for intervention;
- feature refinement should therefore learn a positive demand signal, not only a confidence mask.

Expected signal:

- if this helps, the bottleneck is demand estimation rather than confidence estimation.

### `A_ah03` - edge-control gate

Hypothesis:

- edge refinement currently acts like a late residual helper;
- it should instead be allowed to respond to shared-prior control so it does not duplicate decoder guidance blindly.

Expected signal:

- if this helps, the bottleneck is module role separation rather than more edge detail.

## Why These Three

The three probes are orthogonal in the way that matters for diagnosis:

- `A_af03` asks whether the problem is missing decoder-state feedback.
- `A_ag03` asks whether the problem is missing positive intervention demand.
- `A_ah03` asks whether the problem is missing a coherent role for edge residual correction.

If all three fail, the result will be more informative than repeating another shared-prior tweak.

If one of them wins, it tells us which control function should be used in the next medium-stage design.

## Common Constraints

All three probes should keep:

- frozen baseline isolation from `M04/A_y03`;
- short-tier wall clock;
- the same dataset split policy;
- the same `85-93G` VRAM window;
- the same checkpoint and post-hoc evaluation protocol;
- the same `6x720` geometry unless a smoke test forces a minimal correction.

## Promotion Rule

Only the best same-protocol official test result from these probes should advance.

No branch should be promoted merely because it feels conceptually cleaner.

The goal is to identify a control mechanism that can plausibly beat `A_z03`, not just a nicer story.
