# A_4kd45 Manifest

时间: 2026-06-17 CST

## Purpose

A45 follows the A44 residual-logit gate failure. A44 proved that the A08-vs-A15 oracle gate target is available, but the
residual-logit gate cannot move out of near-zero fixed-prior saturation. A45 tests a prior-initialized bounded residual
gate with direct movement room when `input_std` prior is near zero.

## Mechanism

- Start from A37 best-val.
- Keep A37 decoder heads and route experts frozen.
- Enable `input_std_route_learned_gate_mode = "bounded_residual"`.
- Train only `input_std_route_decoder_heads.gate_head`.
- Use A08/A15 native-config online teachers to compute detached oracle gate targets from teacher-to-GT error.
- Focus oracle gate loss on fixed-low samples inside the route-boundary band.
- Use negative target ceiling `0.0` so true A15/low-route crops are not pushed upward.

## Key Paths

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd45/configs/a_4kd45_fast_4kdehaze_a37_oracle_bounded_residual_gate.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd45/launch_tmux.sh`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- A08 teacher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- A15 teacher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`

## Promotion Focus

- Improve A08-oracle router gap over A37/A44 about `-0.19984 dB`.
- Improve high-route gap over A37 `-0.11159004428502844 dB`, or at least avoid regression.
- Recover nominal-low/A08-oracle records such as `94_9400006.jpg`, `94_94000065.jpg`, and `94_94000081.jpg`.
- Preserve low-route positive surplus.

## Boundary

A45 is a mechanism probe on `4KDehaze_local_90_5_5_fast_probe_512`, not a final benchmark.
