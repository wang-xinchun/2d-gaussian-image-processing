# G_4kd15 Gate Check

- Created: `2026-06-22 22:18:00 CST`
- Status: `GATES PASSED - SMOKE LAUNCH APPROVED`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd15`.
- PASS: First short run uses `G_4kd15_smoke01`, not an A-line ID and not a G14 rerun.
- PASS: Config, launch script, evaluator, and local logs remain under the G-RSCN line.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control/refinement paths are unchanged.
- PASS: G15 reuses the G13/G14 dual high/low Gaussian-conditioned prior-control bank.
- PASS: `adapter_count=1` compatibility remains unchanged.
- PASS: Gate/confidence heads and output delta convolutions remain zero-initialized, preserving A37-equivalent initial output.
- PASS: Trainable prefix remains only `gaussian_route_control`.
- PASS: A18 `val_records` transfer-aware best-selection gate remains enabled.
- PASS: Step-level no-harm guard now has an explicit `start_step`, allowing a short non-zero adapter warmup before rollback/projection.
- PASS: Protect-only acceptance is disabled; projected acceptance must satisfy both protect and target gates or the full gate.
- BLOCKED: Formal long training.

## Deployment Gate

- PASS: Python compile, TOML parse, bash syntax checks, launcher dry-run, and eval dry-run passed.
- PASS: CUDA preflight passed on physical GPU3 with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- PASS: A37 init checkpoint loaded with expected missing `gaussian_route_control.*` keys only and no unexpected keys.
- PASS: Trainable prefix is only `gaussian_route_control`; trainable params are `50114`.
- PASS: Transfer validation produced all planned `transfer_val_*` metrics with `100/100` A18 `val_records` matched.
- PASS: GPU3 had sufficient free memory under the user-approved short-run util exception immediately before launch checks.

## GPU Gate

- Approved physical GPUs are only `0` and `3`; GPU1/2 are forbidden.
- Single visible GPU only; no DDP and no `CUDA_VISIBLE_DEVICES=0,3`.
- Default memory policy is `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- Default free-memory gate is `TWODGAUSS_MIN_FREE_MIB=8192`.
- User-approved short-run exception may raise util waiting only on GPU0/3 when free memory is sufficient and the run will not evict other users.

## Smoke01 Promotion Check

```text
must keep high-route at or above A37
must keep A08-oracle at or above A37
must keep low-route at or above A37
must keep A15-oracle at or above A37
aggregate PSNR alone cannot promote
```

Promotion decision: `NOT PROMOTED`.
