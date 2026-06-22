# Next Experiment Design: V1/T01/A_l03

## Record Time

- Time: `2026-05-14 19:53:50 CST`

## Position In The Line

`A_l03` is the first design-only follow-up prepared **after** the accepted
`A_k03/M03` baseline.

It follows:

1. frozen baseline `A_k03/M03`;
2. the project-level decision to prioritize lightweight template upgrades
   over another near-identical "longer only" replay.

## Why This Branch Exists

`A_k03` already answered one important question:

- extending the accepted `A_i03` setting further is still useful.

But it also left the next gap explicit:

- even after that extension, the current official baseline still sits about
  `0.3743 dB` below the project’s lightweight `90%` PSNR gate on
  `UHD-Blur`.

So the next branch should not simply become "more of `A_k03` again".
It should add one lightweight, Gaussian-first mechanism that can plausibly
improve final reconstruction quality without replacing the backbone with a
heavier generic restorer.

## Chosen Adjustment

The new branch is:

```text
V1/T01/A_l03
```

It keeps from `A_k03/M03`:

- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- `train_context_size=784`
- `train_supervision_crop_size=512`
- effective training split after the first `40` pairs
- batch size `7`
- target real-launch VRAM window `85-93G`
- official full-image validation/test protocol

It adds one new lightweight mechanism:

- `Gaussian-guided decoder modulation`

## What The New Mechanism Does

Until `A_k03`, the Gaussian prior mainly affected reconstruction through:

1. the rendered attention map;
2. affine modulation;
3. the edge-guided refinement branch.

`A_l03` pushes that same prior one step deeper into the high-resolution path:

1. build a compact Gaussian prior stack from:
   - rendered attention map
   - local attention contrast
   - optional directional/frequency prior
2. encode that stack with a tiny depthwise residual adapter
3. predict bounded feature-wise scale/bias
4. modulate the decoder feature *before* the final RGB prediction

This keeps the architecture Gaussian-first:

- the new signal still comes from Gaussian-derived priors;
- the backbone width and macro depth stay unchanged;
- the added compute is small compared with replacing the whole decoder.

## Why This Variable Comes Next

Among the plausible post-`A_k03` upgrades, this one has the best alignment
with the current project constraints:

1. it preserves the accepted `A_k03` training recipe;
2. it does not increase the global backbone width or block counts;
3. it uses Gaussian outputs more directly instead of treating them as a side
   branch only;
4. it is naturally compatible with the already accepted
   edge-guided + directional-frequency + consistency line.

## Compatibility Requirement

The implementation is required to remain safe for the frozen baseline line:

- old configs keep the mechanism disabled by default;
- old checkpoints keep loading under their original configs without key
  mismatches;
- milestone packages `M01`, `M02`, and `M03` remain untouched.

## Current Status

This branch is currently **design-only prepared**.

At this stage, the expected work is:

1. land the default-off code path;
2. add the branch config;
3. run static validation and a real-config forward smoke;
4. do **not** deploy training yet.
