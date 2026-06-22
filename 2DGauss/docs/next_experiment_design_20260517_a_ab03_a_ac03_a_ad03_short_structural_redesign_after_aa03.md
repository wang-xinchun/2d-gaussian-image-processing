# Next Experiment Design: `V1/T01/A_ab03` / `A_ac03` / `A_ad03`

## Record Time

- Time: `2026-05-17 00:28:04 CST`

## Position In The Line

This family is prepared after:

1. frozen official deblur baseline `M04/A_y03`;
2. completed short positive probe `A_z03`;
3. completed medium promotion attempt `A_aa03`, which proved that `decoder_detail_hint` is a real local positive but not a standalone medium-line replacement.

## Why This Family Exists

`A_aa03` answered the promotion question clearly enough:

- the detail-hint path is not dead;
- it is also not strong enough to beat frozen `M04/A_y03` by itself;
- continuing to tune only self-conditioned decoder hints would be low-information.

So the next useful question is no longer:

> can we squeeze another tiny gain out of the same decoder-detail hint?

The next question is:

> which lightweight structural interface lets Gaussian prior control high-resolution detail more effectively, without giving up the current lightweight envelope?

This family therefore tests three different, but still lightweight, hypotheses. Each one keeps the `M04/A_y03` envelope as the comparison anchor and changes only one structural idea.

## Common Short-Budget Envelope

All three probes should stay in the same short-tier regime:

- `batch_size=6`
- `train_context_size=728`
- short wall-clock budget, about `58` minutes
- clean reserved-memory target window `85-93G`
- same paired `UHD_deblur` protocol and same official full-image post-hoc evaluation

The purpose of the short tier here is not to win the final benchmark immediately. It is to select the best structural candidate for the next medium promotion.

## Candidate `A_ab03`: Edge-Aware Feature Refinement

### Hypothesis

The missing signal is not more decoder self-statistics. The missing signal is explicit high-resolution structure.

### Structural Change

Keep the current Gaussian-first lightweight recipe, but replace the decoder-detail hint with a compact edge-aware prior inside `GaussianGuidedFeatureRefinement`.

The intended prior source is the same kind of high-resolution edge descriptor already used by the terminal edge refiner family:

- gray structure
- gradient magnitude
- directional edge traces
- local high-pass / Laplacian response

### Why This Is Worth Testing

`A_aa03` showed that self-conditioned decoder energy is too weak as the only extra hint. If the medium-gap is really about missing geometry rather than missing self-awareness, then explicit edge structure should be a better early interface for Gaussian refinement.

### What Success Would Mean

If `A_ab03` beats `A_aa03` and remains stable under the same short budget, it becomes the strongest candidate for the next 5-hour promotion run.

## Candidate `A_ac03`: Confidence-Normalized Gaussian Refinement

### Hypothesis

The failure mode is not only under-guidance. It may also be over-refinement in regions that are already well solved.

### Structural Change

Keep the current Gaussian feature refinement path, but add a lightweight confidence normalization term so refinement strength is suppressed where decoder/attention confidence is already high.

The confidence signal can be derived from compact summaries of:

- attention map concentration
- local contrast
- decoder feature energy stability

### Why This Is Worth Testing

`A_aa03` may be doing the right kind of correction, but in the wrong spatial balance. A confidence-normalized gate tests whether the model needs less aggressive Gaussian intervention in easy regions, not more signal everywhere.

### What Success Would Mean

If `A_ac03` improves official test without increasing parameter footprint materially, then the next medium promotion should probably keep the Gaussian line but change its spatial gating policy.

## Candidate `A_ad03`: Coupled Decoder-Guidance / Feature-Refinement Prior

### Hypothesis

The current pipeline may be too sequential: decoder guidance and feature refinement each see partial priors, but they do not share a compact representation strongly enough.

### Structural Change

Keep decoder guidance and feature refinement both enabled, but make them consume a shared compact prior feature instead of two mostly separate prior encoders.

The idea is to let the same lightweight prior summary drive both:

- decoder modulation
- feature-space refinement

### Why This Is Worth Testing

This tests whether the bottleneck is the interface between the two Gaussian modules, not the modules themselves. If the same prior can be reused more coherently, the model may gain more absolute performance without becoming heavier.

### What Success Would Mean

If `A_ad03` is the best of the three, it becomes the cleanest medium-promotion anchor because it improves the coordination of the existing Gaussian stack instead of adding more ad hoc hints.

## Recommended Order

1. `A_ab03` first, because it is the most direct answer to the `A_aa03` failure mode.
2. `A_ac03` second, because it tests whether the remaining problem is over-refinement rather than missing structure.
3. `A_ad03` third, because it tests whether the two Gaussian modules need a shared prior interface.

## Promotion Logic

Only the short probe that shows the best official test and the cleanest validation behavior should be promoted into the next medium-budget run.

If none of the three outperforms `A_aa03` in a meaningful way, then the correct lesson is not to keep adding more near-neighbor hints. The next move should be another lightweight structural redesign, not a long-budget replay of the same idea.
