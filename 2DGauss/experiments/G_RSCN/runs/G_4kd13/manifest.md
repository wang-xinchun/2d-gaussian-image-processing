# G_4kd13 Manifest

- Created: `2026-06-22 21:15:00 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd13`
- First smoke run: `G_4kd13_smoke01`
- Status: `smoke01 complete; not promoted`

## Purpose

`G_4kd12_smoke01` showed that the transfer-aware guard works, but the same-structure shared `gaussian_route_control` slot only produced protect-only movement. It could improve or protect low/A15, and it approached the A08 transfer target, but it did not create a checkpoint that simultaneously passed high/A08 target and low/A15 protection.

`G_4kd13_smoke01` is the first structural decoupling probe after G12. It keeps the immutable Gaussian core and G12 transfer-aware selection surface, but changes `GaussianRouteControl` from one shared Gaussian-conditioned prior feature into a dual high/low prior-control bank:

```text
route_input
-> shared route feature for gate/confidence
-> high_prior_encoder -> high_control -> high_delta
-> low_prior_encoder  -> low_control  -> low_delta
-> input_std/Gaussian-aware gate blend before RGB
```

This tests whether separating high-preserve and low-dehaze Gaussian-conditioned route-control capacity can change the high/A08 vs low/A15 Pareto shape.

## Difference From G_4kd12

Kept from G12:

- A37 parent checkpoint;
- immutable `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control/refinement paths;
- trainable prefix only `gaussian_route_control`;
- A18 `val_records` transfer-aware guard;
- two-phase protect/target guard and interpolation projection;
- A37-aligned promotion gates.

New in G13:

- `gaussian_route_control_adapter_count = 2`;
- `GaussianRouteControl` uses separate high/low Gaussian-conditioned prior encoders when adapter count is 2;
- high and low route deltas no longer share the same `prior_encoder(route_input)` feature;
- output deltas and learned gate heads remain zero-initialized, so the initial restored output is still A37-equivalent.

## Smoke01 Artifacts

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd13/configs/g_4kd13_smoke01_a37_dual_prior_transfer_guard_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd13/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd13/evaluate_smoke01_gpu.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Transfer records: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd18/metrics/a08_a15_router_val_to_test_cpu.json`

## Deployment Envelope

- Short smoke only: `max_steps=20`, `batch_size=1`, `grad_accum=2`.
- Ordinary validation: `24` batches.
- Transfer validation: full A18 `val_records=100`.
- Physical GPU: only GPU `0` or `3`.
- Default GPU: physical GPU `3`.
- Memory policy: `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- No DDP, no multi-GPU visibility, no CPU fallback.

## Promotion Boundary

`G_4kd13_smoke01` is promotable only if post-training A37-aligned GPU evaluation passes all route/oracle gates simultaneously:

```text
high-route >= A37
A08-oracle >= A37
low-route >= A37
A15-oracle >= A37
```

Aggregate PSNR alone cannot promote. Formal long training remains blocked until the short-run result passes this gate.

## Smoke01 Outcome

`G_4kd13_smoke01` completed on physical GPU3 with train exit code `0`.

- Final step: `20`.
- Step guard checks: `10`.
- Accepted updates: `0`.
- Rollbacks: `10`.
- Full/protect/target gate passes: `0/0/0`.
- Best-val step: `0`, the A37 parent.
- Final GRC delta metrics: high `0.0`, low `0.0`.

The dual high/low prior-control bank was structurally wired, but every zero-init adapter update was rolled back by the every-2-step transfer-aware guard. G13 is therefore a clean negative result for strict step-level guarded learning from a zero-init dual adapter.
