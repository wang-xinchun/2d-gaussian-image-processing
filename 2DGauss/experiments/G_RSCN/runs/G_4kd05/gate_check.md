# G_4kd05 Gate Check

- Created: `2026-06-21 21:22:35 CST`
- Status: `SMOKE01 EVALUATED - near miss; formal training blocked`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd05`.
- PASS: First short run uses `G_4kd05_smoke01`, not an A-line ID and not a G_4kd04 extension.
- PASS: A37 is used only as initialization/legacy anchor; training artifacts remain G-RSCN.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: `step_no_harm_guard` is enabled and single-process only.
- PASS: Guard checks low-route, A15-oracle, and low-A15 validation gaps against the initialized A37-equivalent baseline.
- PASS: Guard uses full `24` validation batches rather than the G_4kd04 8-batch guard.
- PASS: Best selection still requires baseline no-regression before saving a trained checkpoint.

## Deployment Gate

- PASS: Config uses existing fast-probe dataset view for comparability.
- PASS: Parent checkpoint is A37 best, so low/A15 no-harm is measured against the formal A37 gate rather than a weaker G_4kd02 parent.
- PASS: Short run only: `max_steps=24`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- PASS: Train and best/final A37-aligned evaluations completed on physical GPU3 with exit code `0`.
- PASS: Formal long training remains blocked because high-route/A08-oracle are still slightly below A37.

## GPU Gate

- PASS: Approved physical GPUs are only `0` and `3`; GPU1/2 are forbidden.
- PASS: Single visible GPU only; no DDP and no `CUDA_VISIBLE_DEVICES=0,3`.
- PASS: Default memory policy is `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- PASS: Default free-memory gate is `TWODGAUSS_MIN_FREE_MIB=8192`.
- PASS: Default util gate is `TWODGAUSS_MAX_GPU_UTIL=20`, with user-approved short-run exception only when GPU0/3 has enough free memory and the run will not evict other users.

## Smoke01 Promotion Check

```text
PASS: step_no_harm_guard wrote accepted/rollback evidence
PASS: final test PSNR is above A37 by +0.00009609345127614688
PASS: final low-route is above A37 by +0.0001846056939061591
PASS: final A15-oracle is above A37 by +0.000247353157482616
FAIL: final high-route is below A37 by -0.000021236730810156312
FAIL: final A08-oracle is below A37 by -0.00008147229078794838
```

Still forbidden:

```text
formal long training
multi-GPU/DDP
physical GPU1/GPU2
CUDA_VISIBLE_DEVICES=0,3
CPU fallback
promoting validation-only or high/A08-only movement
extending G_4kd05_smoke01 into formal training
```
