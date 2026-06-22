# G_4kd15 Manifest

- Created: `2026-06-22 22:18:00 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd15`
- First smoke run: `G_4kd15_smoke01`
- Status: `smoke01 complete; not promoted`

## Purpose

G13 and G14 isolated a sharper bottleneck:

```text
G13: dual high/low prior-control bank + strict step guard from zero
=> adapter signal is repeatedly rolled back and stays zero

G14: same dual bank + no step rollback
=> adapter signal becomes non-zero, but high/A08 gains spend low/A15
```

`G_4kd15_smoke01` tests whether the new architecture can use a short non-zero warmup and then preserve both branches through transfer-aware guard/projection. It does not change the promotion rule; it only changes the timing of when the no-harm guard begins.

## Difference From G_4kd14

Kept from G14:

- A37 parent checkpoint;
- immutable Gaussian core;
- `gaussian_route_control_adapter_count = 2`;
- trainable prefix only `gaussian_route_control`;
- A18 `val_records` transfer-aware best-selection gate;
- A37-aligned test promotion gates.

Changed in G15:

- added trainer support for `training.step_no_harm_guard_start_step`;
- `step_no_harm_guard_enabled = true`;
- `step_no_harm_guard_start_step = 10`;
- `step_no_harm_guard_interval = 2`;
- `step_no_harm_guard_projection_enabled = true`;
- projection ratios are `[0.75, 0.50, 0.25, 0.125]`;
- protect/target projection gates require non-regression versus the A37 transfer baseline;
- protect-only acceptance is disabled, so low/A15-only motion cannot masquerade as a breakthrough.

## Smoke01 Artifacts

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/configs/g_4kd15_smoke01_a37_dual_prior_warmup_projection_guard_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/evaluate_smoke01_gpu.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Transfer records: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd18/metrics/a08_a15_router_val_to_test_cpu.json`

## Deployment Envelope

- Short smoke only: `max_steps=20`, `batch_size=1`, `grad_accum=2`.
- Step guard checks begin at step `10`, every `2` steps.
- Guard validation uses `24` ordinary batches plus full A18 `val_records=100`.
- Ordinary validation and best-selection gate run at step `20`.
- Physical GPU: only GPU `0` or `3`.
- Default GPU: physical GPU `3`.
- Memory policy: `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- No DDP, no multi-GPU visibility, no CPU fallback.

## Promotion Boundary

`G_4kd15_smoke01` is promotable only if post-training A37-aligned GPU evaluation passes all route/oracle gates simultaneously:

```text
high-route >= A37
A08-oracle >= A37
low-route >= A37
A15-oracle >= A37
```

Aggregate PSNR alone cannot promote. Formal long training remains blocked until the short-run result passes this gate.

## Smoke01 Outcome

`G_4kd15_smoke01` completed on physical GPU3 with train exit code `0`; final A37-aligned GPU evaluation completed with exit code `0`.

- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/logs/smoke01_20260622_222651.log`
- Final eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15/logs/eval_smoke01_final_20260622_223732.log`
- Run dir: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd15_smoke01`

Training summary:

```text
final step: 20
guard start step: 10
guard checks: 6
guard accepted: 0
guard rollbacks: 6
projection attempts: 0
best_val step: 0
```

All delayed guard checks at steps `10/12/14/16/18/20` rolled back. The final candidate did lift the transfer high-route metric, but A08-oracle, low-route, A15-oracle, and low-A15 all remained below the A37 transfer baseline, so no projected update was accepted.

Final A37-aligned GPU evaluation:

```text
PSNR       17.889589332557936 (-0.000069120880880 vs A37)
high-route -0.11164105487970442 (-0.000051010594676)
A08-oracle -0.19993386251938058 (-0.000092875257996)
low-route   0.15905370235401622 (-0.000082783026609)
A15-oracle  0.24930469167116928 (-0.000048885670740)
```

G15 is not promoted. It proves delayed warmup plus guard can keep damage small, but it still does not find a simultaneous high/A08 and low/A15-safe update direction.
