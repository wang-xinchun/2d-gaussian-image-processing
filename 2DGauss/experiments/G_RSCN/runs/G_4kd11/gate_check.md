# G_4kd11 Gate Check

- Created: `2026-06-22 16:05:00 CST`
- Status: `SMOKE01 COMPLETED - NOT PROMOTED`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd11`.
- PASS: First short run uses `G_4kd11_smoke01`, not an A-line ID and not a G10 rerun.
- PASS: Config, launch script, evaluator, and local logs remain under the G-RSCN line.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: G11 changes update application rather than Gaussian architecture or scalar-only thresholds.
- PASS: Full five-metric step guard remains enabled.
- PASS: Two-phase guard is enabled: low/A15 protect gate and high/A08 target gate are tracked separately.
- PASS: Candidate projection is enabled with ratios `[0.75, 0.50, 0.25]`.
- PASS: Projection accepts only protect+target passing states or full-gate passing states.
- PASS: `input_std` remains a prior and sampler feature, not the architecture identity.

## Deployment Gate

- PASS: Config uses the existing fast-probe dataset view for comparability.
- PASS: Parent is A37 best-val.
- PASS: Short run only: `max_steps=20`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- PASS: Python compile, TOML parse, bash syntax checks, and CUDA preflight passed.
- PASS: CUDA preflight used physical GPU3, loaded A37 plus expected `gaussian_route_control.*` missing keys only, and kept trainable prefix only `gaussian_route_control`.
- PASS: GPU gate immediately before launch selected physical GPU3 with enough free memory under the user-approved short-run util exception.
- PASS: Training completed on physical GPU3 with exit code `0`, single visible GPU, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`, and no CPU fallback.
- PASS: A37-aligned GPU evaluation completed for both `best_val` and `final` with exit code `0`.
- BLOCKED: Formal long training.

## GPU Gate

- Approved physical GPUs are only `0` and `3`; GPU1/2 are forbidden.
- Single visible GPU only; no DDP and no `CUDA_VISIBLE_DEVICES=0,3`.
- Default memory policy is `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- Default free-memory gate is `TWODGAUSS_MIN_FREE_MIB=8192`.
- User-approved short-run exception allows `TWODGAUSS_MAX_GPU_UTIL=100` on GPU0/3 when free memory is sufficient and the run will not evict other users.

## Smoke01 Promotion Check

```text
must keep high-route at or above A37
must keep A08-oracle at or above A37
must keep low-route at or above A37
must keep A15-oracle at or above A37
step guard must report accepted_mode as full, protect_only, projection, or rollback
aggregate PSNR alone cannot promote
```

Still forbidden:

```text
formal long training
multi-GPU/DDP
physical GPU1/GPU2
CUDA_VISIBLE_DEVICES=0,3
CPU fallback
promoting validation-only movement without test-side gates
```

## Smoke01 Gate Result

Training guard:

- PASS: `10/10` step no-harm guard checks passed.
- PASS: `0` rollbacks.
- PASS: All accepted checks used `accepted_mode=full`.
- INFO: Projection and protect-only acceptance were enabled but not triggered.
- PASS: Best validation step was `6`.

A37-aligned test gate:

```text
best_val high-route: -0.111665766148642 vs A37 -0.111590044285028 -> FAIL by -0.000075721863613
best_val A08-oracle: -0.200214915987967 vs A37 -0.199840987261385 -> FAIL by -0.000373928726582
best_val low-route: 0.159365047167789 vs A37 0.159136485380625 -> PASS by +0.000228561787164
best_val A15-oracle: 0.249853071844794 vs A37 0.249353577341910 -> PASS by +0.000499494502884

final high-route: -0.111663922610018 vs A37 -0.111590044285028 -> FAIL by -0.000073878324990
final A08-oracle: -0.200062964555102 vs A37 -0.199840987261385 -> FAIL by -0.000221977293717
final low-route: 0.159126179561716 vs A37 0.159136485380625 -> FAIL by -0.000010305818909
final A15-oracle: 0.249472961709662 vs A37 0.249353577341910 -> PASS by +0.000119384367752
```

Promotion decision: `NOT PROMOTED`. Aggregate PSNR improvement on `best_val` is too small and does not compensate for high-route/A08-oracle test regression.
