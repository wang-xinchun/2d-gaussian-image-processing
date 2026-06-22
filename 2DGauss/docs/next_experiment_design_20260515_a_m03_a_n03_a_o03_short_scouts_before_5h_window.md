# Next Experiment Design: V1/T01/A_m03 + A_n03 + A_o03

## Record Time

- Time: `2026-05-15 00:20:29 CST`

## Position In The Line

This probe family is prepared **after**:

1. frozen official deblur baseline `M03/A_k03`;
2. design-only architectural branch `A_l03`;
3. the user instruction to reserve a later `5`-hour long-training window and
   first use `1-3` short runs to scout direction quickly.

## Why Short Scouts Come Before The Next 5-Hour Run

The current mainline situation is unusually specific:

- `A_k03` already proved that longer training still helps;
- but `A_k03` also showed that "same template, more time only" is no longer
  the highest-information next variable;
- `A_l03` has already landed the new lightweight Gaussian-guided decoder
  modulation path, but it has not yet been tested by a clean training run;
- earlier partial deployment evidence also warned that `A_l03` needs a fresh,
  branch-local VRAM check before any medium/long run.

So the correct next move is not to spend the full `5` hours immediately on one
untested setting. The correct move is to first use a small probe family to
answer three questions:

1. does `A_l03` produce a positive short-run signal at all;
2. does the new decoder guidance prefer a gentler or default strength;
3. can we stay close to the accepted `A_k03` geometry while still respecting
   the real launch VRAM window.

## Probe Family Overview

| Run | Role | Main variable | Geometry | Intended question |
|---|---|---|---|---|
| `A_m03` | conservative architecture scout | `strength = 0.10` | `batch=6`, `context=784` | Is gentle decoder modulation already helpful without over-driving the high-res path? |
| `A_n03` | default architecture scout | `strength = 0.15` | `batch=6`, `context=784` | Does the original `A_l03` default strength give a better short-run signal than the conservative version? |
| `A_o03` | high-fill runtime scout | `context = 776`, `batch=7`, `strength = 0.15` | near-`A_k03` launch shape | Can the new mechanism stay close to the accepted `7`-sample medium-run geometry while aiming for the `85-93G` window? |

## Why These Three

### `A_m03`

This is the safest mechanism-first probe.

It keeps:

- the `A_k03` / `A_l03` accepted mainline recipe;
- `train_context_size=784`;
- the same effective train split after the first `40` pairs;
- all accepted edge-guided, directional-frequency, and consistency components.

It changes only:

- `use_gaussian_decoder_guidance = true`
- `gaussian_decoder_guidance_strength = 0.10`
- short-tier batch/runtime settings for screening

This run is the best first answer to:

> if the new decoder modulation is useful, does it first show up as a small,
> clean, low-risk gain rather than as a strong aggressive intervention?

### `A_n03`

This is the direct default-strength follow-up.

Relative to `A_m03`, it changes only:

- `gaussian_decoder_guidance_strength = 0.15`

This makes `A_m03 -> A_n03` a clean intensity comparison rather than a mixed
architecture comparison.

### `A_o03`

This is not mainly a "third architecture vote".
It is a **launch-shape scout**.

It keeps the default `A_l03` strength, but changes geometry to:

- `batch_size = 7`
- `train_context_size = 776`

The point is to check whether the new mechanism can remain close to the
accepted `A_k03` medium-run throughput shape after a small context trim, rather
than dropping all the way to a clearly lower-fill launch recipe.

## Ranking And Promotion Gates

Because the project has already recorded that the `20`-image internal
full-image validation slice does **not** perfectly rank the eventual official
test winner, these short probes should not be treated as formal baseline
replacement runs.

So the gate logic here is deliberately narrow:

### Gate 0: Launch gate

Before any probe is accepted as valid, it must pass clean serial calibration:

- no concurrent GPU contamination;
- no NaN or OOM during smoke/preflight;
- reserved VRAM should land inside the project window `85-93G`, or at minimum
  be close enough that only a branch-local micro-adjustment is still needed.

### Gate 1: Completion gate

The probe must:

- finish the full short budget cleanly;
- survive repeated validation/checkpoint cycles;
- produce complete checkpoints and metrics.

### Gate 2: Architecture keep/drop gate

Primary screening signal:

- `best training-time full-image val PSNR >= 28.70 dB`: keep candidate
- `best training-time full-image val PSNR >= 28.80 dB`: strong keep candidate
- `best training-time full-image val PSNR < 28.60 dB`: reject for long-run promotion

Why these numbers:

- `A_j03` short negative probe peaks at about `28.6460 dB`
- `A_g03` short positive branch peaks at about `28.7401 dB`

So `28.70-28.80 dB` is a reasonable screening band for "better than the known
negative short probe, and at least competitive with earlier short positive
signals".

### Gate 3: Long-run nomination rule

Among the probes that pass Gates `0-2`, nominate the next `5`-hour run anchor
by:

1. higher **final-stage** full-image validation quality first;
2. if two probes are within about `0.03 dB`, prefer the lower-strength /
   lower-risk version;
3. if `A_o03` is the only probe that cleanly satisfies the VRAM window while
   preserving a near-`A_k03` launch shape, it becomes the preferred deployment
   anchor even if its validation edge is only small.

Official test may still be run later for record, but it should remain
**corroborative**, not the formal short-probe promotion gate.

## Recommended Execution Order

Recommended order:

1. `A_m03`
2. `A_n03`
3. `A_o03` only if we need the high-fill launch-shape answer before the long run

This order is deliberate:

- first answer whether the new mechanism is directionally healthy at all;
- then compare it against the original default strength;
- only then spend extra probe time on the deployment-shape scout.

## What The 5-Hour Long Run Should Do After These Probes

The later `5`-hour run should **not** be launched from raw `A_l03` directly.

It should inherit from the best short probe that satisfies:

- clean runtime behavior;
- no post-validation instability;
- competitive full-image validation trajectory;
- continued alignment with the lightweight Gaussian-first story.

If none of the short probes passes the screening band cleanly, then the correct
decision is **not** to force a long `A_l03` run anyway. In that case, the
project should either:

1. fall back to the frozen `A_k03/M03` recipe for the long window; or
2. revise the decoder-guidance mechanism before spending the `5`-hour budget.
