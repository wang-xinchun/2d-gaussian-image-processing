# A_4kd44 Manifest

时间: 2026-06-17 CST

## Purpose

A44 follows the A43 clean negative. It tests whether the remaining bottleneck is the fixed `input_std` route boundary
rather than high-branch teacher fitting. The experiment starts from A37 and learns only a small residual gate on top of
the fixed prior.

## Mechanism

- Start from A37 best-val.
- Keep A37 decoder heads and route experts frozen.
- Enable `input_std_route_learned_gate_mode = "residual"` with zero-initialized gate residual, so launch behavior matches
  the fixed `input_std` prior.
- Train only `input_std_route_decoder_heads.gate_head`.
- Use A08/A15 native-config online teachers to compute a detached oracle gate target from teacher-to-GT error.
- Focus the oracle gate loss on fixed-low samples near or below the route boundary, where A08 can be the oracle winner.

## Key Paths

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd44/configs/a_4kd44_fast_4kdehaze_a37_oracle_residual_gate.toml`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- A08 teacher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- A15 teacher: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`

## Promotion Focus

- Recover nominal-low/A08-oracle boundary records such as `94_9400006.jpg`, `94_94000065.jpg`, and `94_94000081.jpg`.
- Improve A08-oracle router gap over A37 `-0.19984098726138502 dB`.
- Improve high-route gap over A37 `-0.11159004428502844 dB`, or at least not regress.
- Low-route gap must stay positive; aggregate PSNR is secondary.

## Boundary

A44 is a fast-probe mechanism diagnostic on `4KDehaze_local_90_5_5_fast_probe_512`, not an official/full benchmark.
