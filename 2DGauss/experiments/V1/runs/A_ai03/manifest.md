# Run Manifest: V1/T01/A_ai03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ai03`
- Full identifier: `V1/T01/A_ai03`

## Current Status

Completed short training run. This branch passed static/preflight/smoke gates, launched as tmux session `2dgauss_a_ai03_uhddeblur_shared_detail_gate_probe` at `2026-05-19 10:27:25 CST`, finished cleanly at `step=2500`, and completed same-protocol official full-image testing. It kept the short-tier `6x720` envelope and frozen `M04/A_y03` isolation, but changed the shared-control feature refinement gate so it could read decoder-detail feedback.

## Purpose

`A_ai03` exists to test the root bottleneck sharpened after `A_ah03`:

> the shared control surface improved coordination, but lost the decoder-detail feedback that made `A_z03` the current short leader.

## Experimental Question

Can shared-prior feature refinement recover the missing `A_z03`-style decoder-detail feedback without reverting to direct cue stacking?

## Chosen Mechanism

`A_ai03` keeps:

```text
use_gaussian_decoder_guidance = true
use_gaussian_feature_refinement = true
use_edge_weighted_reconstruction = true
use_edge_guidance = true
use_directional_frequency_guidance = true
use_train_inference_consistency = true
use_gaussian_shared_prior = true
```

and changes:

```text
gaussian_feature_refinement_use_decoder_detail_gate = true
```

The branch does not concatenate decoder-detail channels into the shared-prior encoder. Instead, it uses decoder energy and local detail contrast as a multiplicative gate on the feature-refinement control path after the shared prior has been formed.

## Baseline For Comparison

- Frozen medium baseline: `M04/A_y03`, official test `26.272695829299355`
- Current short leader: `V1/T01/A_z03`, official test `25.385710820365173`
- Best shared-control local result: `V1/T01/A_ah03`, official test `25.345361164304613`
- Earlier shared-prior result: `V1/T01/A_ae03`, official test `25.253274175974838`

## Planned Protocol

Kept from the current short envelope:

- dataset: `UHD_deblur`
- effective training split: train pairs after the first `40`
- internal validation: first `20` train pairs
- validation crop: full image, `crop_size=0`
- official test: `300` full-image pairs, post-hoc only
- `6x720` context geometry
- train supervision crop `512`
- `max_steps=2500`
- wall-clock cap `58` minutes
- validation interval `500`
- checkpoint interval `250`
- target reserved VRAM window `85.0-93.0G`

## Gate Requirements

Before launch:

1. static import/compile check;
2. formal `train_v1_t01.py --preflight` on the true config;
3. true `6x720` smoke calibration within the target reserved-memory window.

After launch:

1. tmux startup confirmation;
2. GPU utilization and memory sanity check;
3. train metrics writing without NaN/OOM;
4. final checkpoint and same-protocol official test after completion.

## Gate Status

- Static py_compile: passed
- Config parser: passed, `gaussian_feature_refinement_use_decoder_detail_gate = true`
- Formal preflight: passed, `total_params=305208`, `batch_input_shape=(6, 3, 720, 720)`
- True `6x720` smoke: passed, `max_memory_reserved_gb=90.95703125`, `avg_total_step_seconds=2.0054664835333824`
- Launch: passed, tmux session `2dgauss_a_ai03_uhddeblur_shared_detail_gate_probe`
- Startup metric: `step=1`, `loss_total=0.22921311855316162`
- Training finish: passed, `step=2500`, `elapsed_minutes=39.654192618529`
- Checkpoint endpoint: `step_002500.pt == latest.pt == final.pt` by model weights
- Official full-image test: `psnr_mean=25.267052413067724`, `loss_mean=0.03668969906090448`

## Final Result

`A_ai03` is a cleanly completed diagnostic run, but it is not a breakthrough. It improves slightly over `A_ae03` by `+0.013778237092886059 dB`, but remains below `A_ah03` by `0.0783087512368894 dB`, below `A_z03` by `0.11865840729744903 dB`, and below frozen `M04/A_y03` by `1.0056434162316314 dB`.

Final decision:

```text
medium-promotion candidate = no
short leader = no
recommended next action = use as negative evidence for decoder-detail gate alone
```

## Working Hypothesis

If `A_ai03` improves over `A_ah03` and preferably crosses `A_z03`, the root bottleneck is specifically the absence of decoder-state/detail feedback inside the shared-control surface.

If it fails while `A_ah03` remains close to `A_z03`, the next likely bottleneck is not decoder-detail feedback alone but a stronger multi-module closed loop such as the still-idle `A_af03` or a combined control design.
