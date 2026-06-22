# G_4kd06 Gate Check

- Created: `2026-06-21 21:55:31 CST`
- Status: `SMOKE01 PREPARED - tiny high/A08 lift under full-val no-harm; formal training blocked`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd06`.
- PASS: First short run uses `G_4kd06_smoke01`, not an A-line ID and not a G_4kd05 extension.
- PASS: Parent is a G-RSCN checkpoint; training artifacts remain G-RSCN.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: `step_no_harm_guard` is enabled and single-process only.
- PASS: Guard checks low-route, A15-oracle, and low-A15 validation gaps against the initialized G_4kd05-final baseline.
- PASS: Guard uses full `24` validation batches.
- PASS: Best selection still requires baseline no-regression before saving a trained checkpoint.
- PASS: The only intended pressure change is a small high/A08 lift: input-std A08 anchor `0.018 -> 0.024`, Gaussian high-branch anchor `0.045 -> 0.060`, LR `6e-6 -> 4e-6`, `max_steps 24 -> 16`.

## Deployment Gate

- PASS: Config uses existing fast-probe dataset view for comparability.
- PASS: Parent checkpoint is G_4kd05 final, the strongest near-miss final state so far.
- PASS: Short continuation only: `max_steps=16`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- PASS: Formal long training remains blocked until A37-aligned evaluation passes all promotion gates.

## GPU Gate

- PASS: Approved physical GPUs are only `0` and `3`; GPU1/2 are forbidden.
- PASS: Single visible GPU only; no DDP and no `CUDA_VISIBLE_DEVICES=0,3`.
- PASS: Default memory policy is `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- PASS: Default free-memory gate is `TWODGAUSS_MIN_FREE_MIB=8192`.
- PASS: Default util gate is `TWODGAUSS_MAX_GPU_UTIL=20`, with user-approved short-run exception only when GPU0/3 has enough free memory and the run will not evict other users.

## Smoke01 Promotion Check

```text
must improve A08-oracle vs A37
must keep high-route at or above A37
must keep low-route at or above A37
must keep A15-oracle at or above A37
step_no_harm_guard must write accepted/rollback evidence
aggregate PSNR alone cannot promote
```

Still forbidden:

```text
formal long training
multi-GPU/DDP
physical GPU1/GPU2
CUDA_VISIBLE_DEVICES=0,3
CPU fallback
promoting validation-only or high/A08-only movement
```
