# A_4kd42 Manifest

时间: 2026-06-16 CST

## Purpose

A42 is the first high-tail-first run after the A37-A41 retrospective. It is designed to attack the unresolved high-route/A08-oracle hard-case bottleneck, not to chase aggregate PSNR first.

The run combines two mechanism changes:

- earlier route-conditioned capacity via a zero-initialized `InputStdRouteFeatureExpert`;
- selected-output A08-oracle anchor loss, enabled by `loss.input_std_route_a08_oracle_anchor_weight`.

Unlike a test-set hard-case cache, A42 does not train on test worst-case records. It compares online A08/A15 teacher errors against the training crop GT and anchors the selected student output to A08 only when A08 is the local oracle.

## Key Paths

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd42/configs/a_4kd42_fast_4kdehaze_a37_feature_oracle_anchor.toml`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Teacher A config/checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/configs/a_4kd08_fast_4kdehaze_medium_detail_protected_blended_demand_control_probe.toml`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher B config/checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/configs/a_4kd15_fast_4kdehaze_a08_best_no_haze_guard_ema995_finetune.toml`, `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- Code changes: `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`

## Training Intent

- Start from A37, the better compressed high-route anchor.
- Add feature-level route capacity before terminal decoder heads while keeping zero-init safety.
- Preserve low route by training only high/shared route feature paths and existing A37 high/shared late modules.
- Use probabilistic high/low sampling with `high_fraction=0.70` so nominal low-route crops still appear.
- Use online A08-vs-A15 oracle advantage on train crops to cover both high input-std crops and nominal-low A08-better hard crops.

## Acceptance Focus

- Primary: high-route A18 alignment should beat A37 `-0.11159004428502844 dB` and ideally cross `>-0.10 dB`.
- Primary: A08-oracle gap should improve over A37 `-0.19984098726138502 dB` and avoid A40/A41-level regression near `-0.238 dB`.
- Safety: low-route selected delta should remain controlled and low-route A18 gap should stay positive.
- Aggregate: PSNR can initially be below A40 if high-tail metrics move in the right direction.

## Boundary

A42 is a fast-probe mechanism diagnostic on `4KDehaze_local_90_5_5_fast_probe_512`, not an official/full benchmark.
