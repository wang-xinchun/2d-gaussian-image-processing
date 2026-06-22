# Next Experiment Design: V1/T01/A_q03

## Record Time

- Time: `2026-05-15 09:10:45 CST`

## Position In The Line

This probe is prepared after:

1. frozen official deblur baseline `M03/A_k03`;
2. completed decoder-guidance short family `A_m03/A_n03/A_o03`;
3. completed medium follow-up `A_p03`;
4. the new bottleneck diagnosis that the remaining gap sits mainly in the
   high-resolution terminal reconstruction path.

## Why This Probe Exists

`A_p03` gave a stable five-hour run and still did not beat `A_k03`.

That means the current question is no longer:

> does Gaussian guidance help at all?

That question is already answered.

The real question now is:

> is the current Gaussian prior still too indirect because it mainly modulates
> decoder features but does not directly shape the final RGB residual enough?

## Main Idea

Add a very small default-off terminal residual head.

This head:

- reads the same rendered attention / local contrast / optional directional
  prior family already validated elsewhere;
- builds a tiny prior feature with depthwise residual blocks;
- gates the final decoder feature;
- predicts a bounded additive residual contribution to the RGB output.

This is lighter and more targeted than widening the backbone, and it matches
the current bottleneck diagnosis more directly than another pure budget replay.

## Anchor And Geometry

The correct short-run comparison anchor is `A_n03`, not `A_p03`.

Reason:

- `A_n03` is the best completed short-structure winner;
- `A_p03` changed both structure and budget;
- we want the next answer to stay interpretable at short-run scale first.

So `A_q03` keeps:

- `batch_size = 6`
- `train_context_size = 784`
- `train_supervision_crop_size = 512`
- `gaussian_decoder_guidance_strength = 0.15`

and adds only:

- `use_gaussian_terminal_residual = true`
- `gaussian_terminal_residual_blocks = 1`
- `gaussian_terminal_residual_strength = 0.10`

## Gate Expectations

Because this is a short scout:

- runtime gate must still satisfy the existing `85-93G` window;
- validation should be judged relative to the same family screening band;
- official test remains post-hoc corroboration only.

## Promotion Logic

If `A_q03` clears the same short-run keep band and shows healthier final-stage
behavior than `A_n03`, it becomes the next candidate for a longer follow-up.

If it fails the keep band, then the project should revise the terminal-control
design itself before spending more long-budget compute.
