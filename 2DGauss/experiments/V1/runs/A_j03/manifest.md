# Run Manifest: V1/T01/A_j03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_j03`
- Full identifier: `V1/T01/A_j03`

## Current Status

This branch has now completed queued deployment, training, and post-hoc
evaluation. It later acquired a clean GPU, finished the short run at
`step=2500`, and is now recorded as a negative-but-informative short screening
branch rather than a replacement baseline candidate.

## Purpose

`A_j03` is the first short-budget branch created directly from the frozen
`A_i03/M02` baseline rather than from the older `A_g03/A_h03` anchor.

Current frozen baseline:

```text
V1/T01/A_i03 final.pt official full-image test PSNR = 25.85061787687282 dB
```

The project now knows that:

1. `A_i03` achieved the first near-`+0.5 dB` jump on the deblur line;
2. that jump came from a combination of consistency repair, larger context,
   longer optimization time, and a slightly larger effective train split;
3. the next long run should therefore not be chosen blindly.

`A_j03` exists to isolate one high-value question before the next long-budget
upgrade:

> if we keep the `A_i03` story and data split fixed, does reallocating the same
> VRAM envelope toward an even larger training context show a positive short-run
> signal?

## Experimental Question

Relative to `A_i03`, can a short probe with:

1. the same accepted edge-guided + directional/frequency + consistency line;
2. the same effective training split after the first `40` pairs;
3. the same `85-93G` deployment window;
4. but a larger training context target (`864` instead of `784`);

show enough positive signal to justify the next longer training run on this
direction?

## Main Design Choice

This branch intentionally changes only the context-allocation side of the
accepted `A_i03` setup:

- keep `train_offset = 40`
- keep `train_supervision_crop_size = 512`
- keep the full-image validation/test protocol
- keep the same model width and restoration mechanisms
- shorten the runtime budget back to a short probe
- increase the target training context from `784` to `864`
- compensate by reducing batch-size expectation from `7` to `6`

The branch is therefore best understood as a memory-reallocation probe rather
than a new mechanism branch.

## Planned Runtime Envelope

- Tier target: `short`
- Planned wall-clock cap: `58` minutes
- Max steps target: `2500`
- Initial batch-size hypothesis: `6`
- Initial context-size target: `864`
- Supervision size: `512`
- Target reserved VRAM window: `85.0-93.0G`
- Validation interval target: `500`
- Checkpoint interval target: `250`
- Validation batches target: `20`

## Comparison Anchor

- Frozen milestone: `experiments/V1/milestones/M02_A_i03_uhddeblur_official_baseline`
- Stable milestone link: `experiments/V1/milestones/current_uhddeblur_official_baseline`
- Short predecessor on the consistency idea: `V1/T01/A_h03`
- Medium accepted anchor: `V1/T01/A_i03`

## Intended Decision Rule

If `A_j03` shows a cleaner or stronger short-run signal than the existing
pre-`A_i03` short probes, the next long-budget branch should upgrade from the
`A_j03` winner rather than simply replaying `A_i03` for longer.

If it does not, the next long-budget branch should stay closer to the accepted
`A_i03/M02` setting.
