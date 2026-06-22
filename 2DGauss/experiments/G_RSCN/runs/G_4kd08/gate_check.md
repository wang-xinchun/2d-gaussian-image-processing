# G_4kd08 Gate Check

- Created: `2026-06-21 22:32:14 CST`
- Status: `COMPLETE - validation pass, test gate failed; formal training blocked`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd08`.
- PASS: First short run uses `G_4kd08_smoke01`, not an A-line ID and not a G_4kd07 extension.
- PASS: Artifacts, config, launcher, evaluator, and logs remain under the G-RSCN line.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: `input_std` remains a prior and sampler feature.
- PASS: `step_no_harm_guard` is enabled and single-process only.
- PASS: Guard checks high-route, A08-oracle, low-route, A15-oracle, and low-A15 validation gaps every two steps.
- PASS: Guard uses full `24` validation batches and restores optimizer state on rollback.

## Deployment Gate

- PASS: Config uses the existing fast-probe dataset view for comparability.
- PASS: Parent is A37 best-val for a clean A37 restart.
- PASS: Short run only: `max_steps=12`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- PASS: Training completed with exit code `0`.
- PASS: A37-aligned GPU evaluation ran for best and final checkpoints.
- FAIL: Test-side high-route and A08-oracle gates did not pass.

## GPU Gate

- PASS: Approved physical GPUs are only `0` and `3`; GPU1/2 are forbidden.
- PASS: Single visible GPU only; no DDP and no `CUDA_VISIBLE_DEVICES=0,3`.
- PASS: `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- PASS: No CPU fallback was used.

## Smoke01 Promotion Check

```text
guard checks: 6
accepted updates: 2
rolled back updates: 4
best checkpoint: fails PSNR, high-route, and A08-oracle vs A37
final checkpoint: fails PSNR, high-route, A08-oracle, and low-route vs A37
promotion: failed
formal long training: blocked
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
