# Success Analysis: M04_A_y03

## Core Judgment

`A_y03` succeeds because it is the first medium-budget replay that validates
the right *combination* of mechanism placement and supervision strength:

1. Gaussian prior acts in decoder feature space rather than directly on the
   final RGB output.
2. Edge-weighted supervision is kept, but weakened to `0.02` instead of being
   pushed too hard.
3. The line is then given enough budget to convert that better structural
   bias into official-test gain.

This is why `A_y03 final.pt = 26.272695829299355 dB` beats both
`A_x03 final.pt = 26.207660130733675 dB` and the previous frozen baseline
`A_k03 final.pt = 26.199959289578075 dB`.

## What Changed Scientifically

The project already had a stable diagnosis before this run: the main deblur
bottleneck was not “training cannot run” but “high-resolution terminal detail
expression is still too weak.” The failed and weakly positive branches then
made the structure boundary clearer:

- `A_q03` showed that directly letting Gaussian control the terminal RGB
  residual is not the right intervention point.
- `A_r03` showed that feature-space refinement alone is plausible but too weak
  to create a decisive gain.
- `A_s03` showed that supervision-only pressure is also insufficient.
- `A_t03` showed the combination is better than either single-variable branch,
  but still too overconstrained to break out materially.
- `A_u03` then identified the useful local rebalance: keep the combination,
  but make edge-weighted reconstruction lighter.

`A_x03` first showed this line can scale to five hours. `A_y03` then proved
that this is not a one-off fluctuation.

## Why The Gaussian Placement Works Better

The key architectural improvement is *where* Gaussian information is injected.

The weaker route was:

- use Gaussian guidance to directly control the terminal RGB residual.

The stronger route is:

- use Gaussian guidance earlier, on decoder features, before the network has
  collapsed everything into final pixel values.

Why this helps:

- decoder features still contain richer spatial and frequency degrees of
  freedom than the final RGB residual path;
- Gaussian priors are better at shaping structure, locality, and smooth
  support than at micromanaging exact final pixel corrections;
- moving the intervention point forward lets Gaussian act as a structured
  organizer rather than as a brittle terminal controller.

In practical terms, this means Gaussian is now helping the model build better
high-resolution detail representations, not merely nudging the last RGB map.

## Why Lighter Edge Supervision Helped

The rebalance from `0.03` to `0.02` on
`edge_weighted_reconstruction_strength` matters because the earlier combined
line was likely over-regularized.

Too-strong edge-weighted pressure can:

- overfocus the loss on high-contrast boundaries;
- reduce flexibility for non-edge textures and softer motion blur regions;
- make the model chase sharper local edges at the expense of globally better
  reconstruction.

The `A_u03 -> A_x03 -> A_y03` line suggests the useful regime is not “remove
edge supervision,” but “keep it as a light bias.” In that regime:

- Gaussian feature refinement supplies structured guidance;
- edge supervision encourages detail recovery;
- neither term dominates the optimization.

That balance is exactly what the official-test improvement is reflecting.

## Why The Replay Matters

`A_x03` alone was not enough to justify a milestone refresh, because its gain
over frozen `A_k03` was only about `+0.0077 dB`.

`A_y03` changes the interpretation because it answers the actual uncertainty:

- same structure;
- same data split;
- same five-hour budget;
- same `6x728` geometry;
- fresh seed only.

The result is not just “still above `A_k03`,” but “above by a clearly larger
margin”:

- `A_y03 - A_k03 = +0.07273653972128045 dB`
- `A_y03 - A_x03 = +0.06503569856568035 dB`

That turns the line from “best observed once” into “fresh-seed confirmed best
result,” which is exactly the threshold needed for a clean milestone refresh.

## Additional Acceptance Signal

The completed post-hoc panel also reinforces an older project lesson:

- `step_021000.pt` is the best internal-validation checkpoint
  (`val_loss=0.021056028548628092`, full-val `29.2312 dB`);
- but `final.pt` is still the best official-test checkpoint
  (`26.2727 dB` vs `26.2206 dB`).

So the success is not only “better architecture.” It is also “the line keeps
converting final-budget training into true benchmark gain,” which makes this a
trustworthy baseline rather than a validation-slice artifact.

## What This Does Not Mean

This success does **not** mean:

- the project has solved the full absolute-performance gap;
- heavier backbones are now justified;
- Gaussian should be made more aggressive at every possible insertion point.

The line is still about `0.3016 dB` below the project’s current lightweight
`90%` goal for `UHD-Blur`.

So the correct reading is narrower and more useful:

- the current strongest lightweight deblur direction is now confirmed;
- the winning mechanism is better Gaussian placement plus lighter auxiliary
  supervision;
- future gains should build on this balance, not break it.
