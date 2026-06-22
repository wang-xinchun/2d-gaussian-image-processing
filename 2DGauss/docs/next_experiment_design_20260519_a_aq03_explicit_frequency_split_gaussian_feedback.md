# Next Experiment Design: A_aq03 Explicit Frequency-Split Gaussian Feedback

日期: 2026-05-19

## Motivation

The current bottleneck is no longer just whether a router can learn role selectivity. Recent structured-router probes showed that role control after feature mixing is not enough. The advisor's key question is earlier: how does the model separate low-frequency structure from high-frequency details at the branch entrance, and how does the Gaussian heatmap return as a meaningful control canvas?

`A_aq03` tests this directly.

## Hypothesis

If the LR Gaussian branch receives a cleaner low-frequency/structure input, its control canvas should become lower entropy and more semantically aligned with restoration demand. If the HR branch separately receives explicit high-frequency detail, Gaussian feedback can modulate detail recovery rather than trying to represent RGB texture through the Gaussian bottleneck.

This should help both performance and the paper story:

- branch entrance becomes explicit and visualizable;
- Gaussian heatmap has a clearer physical meaning as a structure/control canvas;
- high-frequency detail remains in the HR path;
- feedback/reinjection becomes the core innovation rather than a vague attention map.

## Implementation

Default-off code path:

- `ExplicitFrequencySplitConfig` in `src/uhd_gaussian/config.py`
- `ExplicitFrequencySplitter` in `src/uhd_gaussian/models/frequency.py`
- `CrossResolutionGaussianRestorer` support in `src/uhd_gaussian/models/restorer.py`
- TOML keys read by `scripts/train_v1_t01.py`

Active `A_aq03` keys:

```toml
use_explicit_frequency_split = true
explicit_frequency_split_lowpass_kernel_size = 7
explicit_frequency_split_highres_high_band_strength = 0.35
explicit_frequency_split_gaussian_feedback_strength = 0.12
explicit_frequency_split_detail_blocks = 1
```

The split is:

```text
low_band = avg_pool(input)
high_band = input - low_band
```

`low_band` goes through `pixel_unshuffle` into the LR Gaussian branch. `high_band` goes through a lightweight HR detail projector and is added into the HR feature stream. The rendered Gaussian attention map softly modulates that high-detail feature before injection.

## Supervision

The run keeps the established fidelity losses and adds only small deblur proxies:

```toml
edge_magnitude_weight = 0.02
high_frequency_weight = 0.02
local_contrast_weight = 0.01
```

These weights stay under the supervision safety guardrails and are not allowed to dominate.

## Evaluation

Required:

- same-protocol official `UHD_deblur/test300` PSNR/loss;
- `scripts/evaluate_task_specific_metrics.py` diagnostics;
- low-band/high-band/control-canvas visualization;
- detail crop panels;
- failure-mode checklist for ringing, halo, over-sharpening, color shift, checkerboard, edge breaks, and noise amplification.

## Acceptance

Positive evidence:

- official short-tier result is competitive with or above `A_z03/A_ah03`;
- task diagnostics improve without visual artifacts;
- low/high band and Gaussian control visualizations support the intended story.

Negative evidence:

- PSNR or task diagnostics degrade sharply;
- visual detail is merely over-sharpened;
- Gaussian attention no longer aligns with structure/demand.

If negative, do not reject Gaussian feedback as a whole. Reject this specific split/feedback formulation and retune the split strength, lowpass kernel, or feedback injection point.
