# G_4kd01 Manifest

- Created: `2026-06-21 16:06:13 CST`
- Architecture line: `G-RSCN`
- Full ID: `G-RSCN/G_4kd01`
- Historical alias: `G/B_4kd01`
- Status: `G_4kd01_smoke03` evaluated; formal training remains blocked.

## Purpose

Test the first Gaussian-preserved route-specific control architecture:

```text
A37 shared Gaussian foundation
+ Gaussian-aware boundary router
+ route-specific high-preserve / low-dehaze Gaussian control
+ one Gaussian-conditioned pre-RGB route adapter
```

The target is not to beat `A_4kd40` aggregate PSNR first. The first goal is to move the Pareto front:

```text
improve high-route and A08-oracle behavior over A37
while preserving low-route and A15-oracle surplus.
```

## Separation From Old Architecture

This run is not part of `V1/T01/A_4kdXX`.

- Old line: `A_4kd01` to `A_4kd47`, route patches inside `experiments/V1/runs`.
- New line: `G_4kd01`, Gaussian-preserved route-specific control inside `experiments/G_RSCN/runs`.
- Training logs, configs, metrics, and checkpoints must stay under this G-line directory.

## Baselines And Anchors

- A37 checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`.
- A37 config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/configs/a_4kd37_fast_4kdehaze_a33_high_route_a36_teacher_compress.toml`.
- A37 key metrics:
  - test PSNR: `17.889658453438816`;
  - high-route gap: `-0.11159004428502844`;
  - low-route gap: `+0.1591364853806252`;
  - A08-oracle gap: `-0.19984098726138502`;
  - A15-oracle gap: `+0.2493535773419097`.
- A40 aggregate-only warning:
  - test PSNR: `17.900562951956445`;
  - high-route gap worsens to `-0.1180115677202543`.
- A46/A47 gate-only warning:
  - A46 improves high/A08 but collapses low/A15;
  - A47 protects low/A15 but gives back high/A08.

## Required Mechanism

The first implementation adds default-off modules:

1. `GaussianRouteControl`.
   - Inputs: Gaussian attention map, local contrast, directional prior or shared prior feature, decoder feature, control-canvas statistics, and `input_std` prior.
   - Outputs: `route_gate`, `route_confidence`, `high_preserve_control`, `low_dehaze_control`.
   - The route gate initializes to the A18 `input_std` prior.

2. Route-specific Gaussian control.
   - High-preserve branch protects A08-like detail/fidelity.
   - Low-dehaze branch preserves A15-like restoration strength.
   - Control is blended before RGB residual generation.

3. One Gaussian-conditioned route adapter after shared decoder feature formation and before the RGB head.
   - The adapter must consume Gaussian control features.
   - It must be zero-initialized so the initial output equals A37 within tolerance.

4. Frozen A37 output anchor plus shared RGB head.
   - The final RGB projection remains shared for compactness.
   - Route-specific capacity must already exist in feature/control tensors before this head.

## Current Implementation Evidence

- Code: `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`.
- Config parser: `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py` and `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`.
- Equivalence script: `/home/ldj/wxc/2DGauss/scripts/check_g_4kd01_a37_equivalence.py`.
- Equivalence metric: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/metrics/a37_zero_init_equivalence_cpu.json`.
- CPU equivalence result:
  - A37 reference checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`;
  - G_4kd01 expected missing keys: only `gaussian_route_control.*`;
  - unexpected keys: `[]`;
  - output diff vs A37: `max_abs=0.0`, `mean_abs=0.0`;
  - route-control feature delta: `max_abs=0.0`;
  - route gate vs prior: `max_abs=1.0013580322265625e-05`, within clamp tolerance;
  - total params: `206104`;
  - trainable params: `37698`, all under `gaussian_route_control`;
  - frozen params: `168406`, preserving the A37 anchor.

## Smoke01 Deployment

- Run ID: `G_4kd01_smoke01`.
- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke01_grscn_gpu_short.toml`.
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/launch_smoke01_gpu3.sh`.
- Hard-case list: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/hard_case_validation_list.md`.
- Scope: physical GPU3, `max_steps=20`, `batch_size=1`, `grad_accum=4`, `max_val_batches=24`.
- Trainable prefix: `gaussian_route_control` only.
- Loss: short smoke uses `charbonnier_weight=1.0` so the new pre-RGB control module receives real reconstruction gradients.
- Boundary: this is not formal training and cannot promote without post-run hard-case analysis.

## Smoke02 Deployment

- Run ID: `G_4kd01_smoke02`.
- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke02_a08_hard_positive_short.toml`.
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/launch_smoke02_gpu3.sh`.
- Evaluation launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/evaluate_smoke02_gpu3.sh`.
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke01/checkpoints/best_val.pt`.
- Scope: physical GPU3, `max_steps=60`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- Trainable prefix: `gaussian_route_control` only.
- Added pressure: `input_std_route_a08_oracle_anchor_weight=0.10`, strict positive A08-vs-A15 advantage, and A46 hard-positive sampler cache with `input_std_route_balanced_hard_case_fraction=0.65`.
- Protection: `charbonnier_weight=0.35` and `low_demand_identity_weight=0.03`; low/A15 preservation remains a promotion gate.
- Boundary: this is still a short diagnostic run. It cannot unlock formal training unless A37-aligned evaluation improves A08/high behavior without low/A15 collapse.

Smoke02 outcome:

- best checkpoint step `10` improves high-route and A08-oracle vs A37, but aggregate/low-route/A15-oracle regress slightly;
- final checkpoint step `60` improves high-route and A08-oracle more, but low-route/A15-oracle regress further;
- conclusion: the A08/hard-positive pressure is real, but this configuration is not promotable because it starts to repeat the old high/A08 vs low/A15 trade-off.

## Smoke03 Deployment

- Run ID: `G_4kd01_smoke03`.
- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke03_a08_hard_positive_trust_anchor_short.toml`.
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/launch_smoke03_gpu3.sh`.
- Evaluation launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/evaluate_smoke03_gpu3.sh`.
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/best_val.pt`.
- Scope: physical GPU3, `max_steps=40`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- Trainable prefix: `gaussian_route_control` only.
- Added pressure: `input_std_route_a08_oracle_anchor_weight=0.08`, strict positive A08-vs-A15 advantage, and mixed hard-case cache with `20` A08-positive plus `20` A15-protective samples.
- Protection: smoke01 best-val trust anchor with `input_std_route_trust_anchor_weight=0.06`.
- Boundary: this was a short diagnostic run, not a formal train.

Smoke03 outcome:

- best checkpoint step `10` improves high-route by `+0.000134629466901` and A08-oracle by `+0.000967361035998` vs A37, but loses low-route by `-0.001269316128138` and A15-oracle by `-0.002056677034871`;
- final checkpoint step `40` improves high-route by `+0.000272539448355` and A08-oracle by `+0.002163544399084` vs A37, but loses low-route by `-0.002993743963581` and A15-oracle by `-0.004786060222272`;
- conclusion: smoke01 trust anchoring slowed but did not stop the same high/A08 vs low/A15 trade-off. This configuration is not promotable and must not be lengthened into formal training.

## Smoke04 Deployment

- Run ID: `G_4kd01_smoke04`.
- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke04_a08_a15_paired_anchor_short.toml`.
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/launch_smoke04_gpu3.sh`.
- Evaluation launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/evaluate_smoke04_gpu3.sh`.
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/best_val.pt`.
- Scope: physical GPU3, `max_steps=40`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- Trainable prefix: `gaussian_route_control` only.
- Added pressure: paired strict-positive A08 and A15 oracle anchors, both at weight `0.08`.
- Protection: trust anchor disabled to isolate paired-anchor signal; mixed hard-case cache retained from smoke03.
- GPU policy: training/eval used physical GPU3, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`, single visible GPU, no CPU fallback.
- Boundary: this was a short diagnostic run, not a formal train.

Smoke04 outcome:

- training completed with exit code `0`, final step `40`, best step `10`, best val PSNR `16.84318164737956`, peak CUDA reserved `3.0546875GiB`;
- A08 strict-positive anchor active in `13/40` steps; A15 strict-positive anchor active in `27/40` steps;
- best checkpoint improves high-route by `+0.000057449910584` and A08-oracle by `+0.001250165306230` vs A37, but loses low-route by `-0.001627673817840` and A15-oracle by `-0.002737308621266`;
- final checkpoint improves A08-oracle by `+0.001696735654359` vs A37, but loses high-route by `-0.000031935593701`, low-route by `-0.003224132512157`, and A15-oracle by `-0.004874048589308`;
- conclusion: paired A08/A15 scalar anchors are live but still not enough. This configuration is not promotable and must not be lengthened into formal training.

## Initial Trainable Prefix Plan

The exact prefixes must be confirmed by implementation and preflight. The active first-stage trainable prefix is:

```text
gaussian_route_control
```

Allowed auxiliary prefixes only after written approval:

```text
gaussian_feature_refinement
gaussian_decoder_guidance
```

Forbidden as first-stage-only trainable prefixes:

```text
input_std_route_decoder_heads.gate_head
input_std_route_decoder_heads.high_head
input_std_route_decoder_heads.low_head
rgb_head-only or output/post adapters
GaussianParameterHead replacement
TextureAwareGaussianRenderer replacement
```

## Parameter Budget

Known reference:

- A37 total parameters: `168406`.
- A37 trainable parameters: `76643`.
- A36 full dual-backbone router is the capacity upper-bound reference and should not be copied as the compact contribution.

G_4kd01 parameter budget before formal training:

- total parameter increase over A37 should be documented in `gate_check.md`;
- first implementation target: keep total params below `1.35x` A37 unless a written exception is added;
- trainable params must be enough to test route-specific pre-RGB control, but not a hidden full dual backbone.

## Mandatory Gates Before Training

Training is blocked until `gate_check.md` records PASS for all of these:

1. G-RSCN module implementation exists and is default-off compatible.
2. Loading A37 has only expected new missing keys and no unexpected replacement of Gaussian core.
3. With zero-initialized G-RSCN residuals, output difference vs A37 on a fixed CPU/GPU smoke batch is near zero.
4. Initial route gate equals A18 input-std prior within clamp tolerance.
5. Trainable prefixes include only approved G-RSCN modules.
6. Parameter count budget is recorded vs A37 and A36 reference.
7. Hard-positive and hard-negative evaluation records are named before training.
8. GPU gate passes under the project shared-GPU policy.

## Promotion Criteria

Primary:

- high-route gap must beat A37 `-0.11159004428502844`;
- A08-oracle gap must beat A37 `-0.19984098726138502`;
- low-route gap must remain strongly positive and must not repeat A46 collapse;
- A15-oracle gap must remain strongly positive and must not repeat A46 collapse;
- nominal-low/A08-oracle cases such as `94_9400006.jpg`, `94_94000065.jpg`, and `94_94000081.jpg` must be tracked.

Secondary:

- aggregate PSNR can be reported, but cannot promote the run unless the high/A08 criteria move.
- A40 aggregate PSNR is not the first target for G_4kd01.

## Stop Conditions

Stop or do not promote if:

- high-route or A08-oracle gaps are worse than A37 after the first valid evaluation;
- low-route or A15-oracle surplus collapses toward A46-final behavior;
- high/A08 improves only by spending low/A15 surplus without a written preservation mechanism;
- learned route gate becomes the only effective mechanism;
- route-specific modules act only after RGB residual generation;
- Gaussian core modules are replaced or bypassed.
