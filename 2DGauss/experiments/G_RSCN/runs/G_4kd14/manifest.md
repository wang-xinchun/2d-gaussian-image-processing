# G_4kd14 Manifest

- Created: `2026-06-22 21:45:00 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd14`
- First smoke run: `G_4kd14_smoke01`
- Status: `smoke01 complete; not promoted`

## Purpose

`G_4kd13_smoke01` confirmed that the dual high/low prior-control bank is wired into `gaussian_route_control`, but early evidence showed repeated step-level rollback and zero visible GRC delta. Because `GaussianRouteControl` output deltas are zero-initialized, a strict every-2-step guard can erase the adapter before it accumulates a measurable direction.

`G_4kd14_smoke01` keeps the G13 structural decoupling and changes only the acceptance schedule:

```text
G13: zero-init dual adapter + step guard rollback every 2 steps
G14: zero-init dual adapter + no step rollback + transfer-gated best/final evaluation
```

The question is whether the new architecture can form non-zero high/low route-control deltas when its early steps are not repeatedly reset.

## Difference From G_4kd13

Kept from G13:

- A37 parent checkpoint;
- immutable Gaussian core;
- `gaussian_route_control_adapter_count = 2`;
- trainable prefix only `gaussian_route_control`;
- A18 `val_records` transfer-aware full gate for best selection;
- A37-aligned test promotion gates.

Changed in G14:

- `step_no_harm_guard_enabled = false`;
- no `training.step_no_harm_guard_*` gate sections, so the trainer will not perform step-level rollback;
- `val_interval = 20`, making the short run a deferred-transfer-gate probe rather than another step-guard probe;
- `best_val` still requires the same transfer baseline gate; `final` records the unconstrained 20-step adapter direction.

## Smoke01 Artifacts

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd14/configs/g_4kd14_smoke01_a37_dual_prior_deferred_transfer_gate_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd14/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd14/evaluate_smoke01_gpu.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Transfer records: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd18/metrics/a08_a15_router_val_to_test_cpu.json`

## Deployment Envelope

- Short smoke only: `max_steps=20`, `batch_size=1`, `grad_accum=2`.
- Ordinary validation: `24` batches at step 20.
- Transfer validation: full A18 `val_records=100`.
- Physical GPU: only GPU `0` or `3`.
- Default GPU: physical GPU `3`.
- Memory policy: `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- No DDP, no multi-GPU visibility, no CPU fallback.

## Promotion Boundary

`G_4kd14_smoke01` is promotable only if post-training A37-aligned GPU evaluation passes all route/oracle gates simultaneously:

```text
high-route >= A37
A08-oracle >= A37
low-route >= A37
A15-oracle >= A37
```

Aggregate PSNR alone cannot promote. Formal long training remains blocked until the short-run result passes this gate.

## Smoke01 Outcome

`G_4kd14_smoke01` completed on physical GPU3 with train exit code `0`; final GPU evaluation and a step-10 diagnostic evaluation both completed with exit code `0`.

- Final step: `20`.
- Step-level rollback: disabled by design.
- Best-val step: `0`, the A37 parent, because step 20 failed the transfer full gate.
- Final GRC delta metrics: high `2.574286554590799e-05`, low `1.9299128325656056e-05`.

The deferred window allowed the dual adapter to grow non-zero deltas, confirming that G13 was blocked by rollback-from-zero. However, the learned direction improved high/A08 on the A37-aligned test surface while spending low/A15, so it is not promotable.
