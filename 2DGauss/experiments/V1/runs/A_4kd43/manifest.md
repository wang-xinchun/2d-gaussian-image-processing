# A_4kd43 Manifest

时间: 2026-06-17 CST

## Purpose

A43 is the follow-up to the A42 negative high-tail probe. It keeps the A42 earlier route-conditioned feature expert but changes the objective so that A08 pressure cannot fire when A08 is not actually the better teacher.

## Mechanism

- Start from A37 best-val.
- Keep `InputStdRouteFeatureExpert + InputStdRouteDecoderHeads`.
- Keep A08/A15 native teachers for high branch supervision.
- Require strict positive A08-vs-A15 advantage before selected-output A08 anchoring.
- Add A37 trust-anchor teacher, activated only when A37 is closer to the crop GT than the current student output.
- Keep 70/30 probabilistic high/low sampling so nominal-low A08-oracle hard cases remain visible.

## Key Paths

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd43/configs/a_4kd43_fast_4kdehaze_a37_feature_strict_a08_a37_trust_anchor.toml`
- Init/trust checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- A08 teacher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- A15 teacher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`

## Promotion Focus

- Improve A37 high-route gap `-0.11159004428502844 dB`.
- Avoid A42 A08-oracle collapse `-0.31086879727087796 dB`; target at least better than A40/A41 near `-0.238 dB`.
- Preserve positive low-route gap.
- Aggregate PSNR is secondary unless high-tail metrics move in the right direction.

## Boundary

A43 is a fast-probe mechanism diagnostic on `4KDehaze_local_90_5_5_fast_probe_512`, not an official/full benchmark.
