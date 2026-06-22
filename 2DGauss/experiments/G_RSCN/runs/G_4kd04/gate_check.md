# G_4kd04 Gate Check

- Created: `2026-06-21 20:45:00 CST`
- Status: `SMOKE01 EVALUATED - update rollback works; promotion blocked`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd04`.
- PASS: First short run uses `G_4kd04_smoke01`, not an A-line ID and not a G_4kd03 extension.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: Added default-off `step_no_harm_guard` infrastructure in `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`.
- PASS: The guard is single-process only and explicitly refuses DDP.
- PASS: Smoke01 guard checks low-route, A15-oracle, and low-A15 validation gaps against the initialized baseline and rolls back harmful optimizer steps.
- PASS: G_4kd03 protected best-selection gate remains active.

## Deployment Gate

- PASS: Config uses existing fast-probe dataset view for comparability.
- PASS: Parent checkpoint is `G_4kd02_smoke01/best_val.pt`, preserving the strongest current branch-isolated high/A08 signal.
- PASS: Short run only: `max_steps=40`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- PASS: Train and best/final A37-aligned evaluations completed on physical GPU3 with exit code `0`.
- PASS: Formal long training remains blocked because A37 low-route/A15 promotion gates did not pass.

## GPU Gate

- PASS: Approved physical GPUs are only `0` and `3`; GPU1/2 are forbidden.
- PASS: Single visible GPU only; no DDP and no `CUDA_VISIBLE_DEVICES=0,3`.
- PASS: Default memory policy is `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- PASS: Default free-memory gate is `TWODGAUSS_MIN_FREE_MIB=8192`.
- PASS: Default util gate is `TWODGAUSS_MAX_GPU_UTIL=20`, with user-approved short-run exception only when GPU0/3 has enough free memory and the run will not evict other users.

## Smoke01 Promotion Check

```text
PASS: step_no_harm_guard wrote accepted/rollback evidence
PASS: final high-route remains above A37 by +0.00017073030127777444
PASS: final A08-oracle remains above A37 by +0.0016357684574719311
FAIL: final low-route remains below A37 by -0.0019570309529002294
FAIL: final A15-oracle remains below A37 by -0.003323235377853284
FAIL: best_val is still step-0 protected parent and also below A37 on low/A15
```

Still forbidden:

```text
formal long training
multi-GPU/DDP
physical GPU1/GPU2
CUDA_VISIBLE_DEVICES=0,3
CPU fallback
promoting validation-only or high/A08-only movement
extending G_4kd04_smoke01 into formal training
```
