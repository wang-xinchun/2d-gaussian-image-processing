# G_4kd07 Gate Check

- Created: `2026-06-21 22:20:07 CST`
- Status: `COMPLETE - no accepted update; formal training blocked`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd07`.
- PASS: First short run uses `G_4kd07_smoke01`, not an A-line ID and not a G_4kd06 continuation artifact.
- PASS: Artifacts, config, launcher, evaluator, and logs remain under the G-RSCN line.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: `input_std` remains a prior and sampler feature, not a replacement for Gaussian route control.
- PASS: `step_no_harm_guard` is enabled and single-process only.
- PASS: Guard checks high-route, A08-oracle, low-route, A15-oracle, and low-A15 validation gaps every two steps.
- PASS: Guard uses full `24` validation batches and restores optimizer state on rollback.

## Deployment Gate

- PASS: Config uses the existing fast-probe dataset view for comparability.
- PASS: Parent is `G_4kd05_smoke01/final.pt`, the strongest A37-start near miss before this probe.
- PASS: Short run only: `max_steps=12`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- PASS: Training completed with exit code `0`.
- FAIL: No candidate update passed the guard; formal long training remains blocked.

## GPU Gate

- PASS: Approved physical GPUs are only `0` and `3`; GPU1/2 are forbidden.
- PASS: Single visible GPU only; no DDP and no `CUDA_VISIBLE_DEVICES=0,3`.
- PASS: `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- PASS: No CPU fallback was used.

## Smoke01 Promotion Check

```text
guard checks: 6
accepted updates: 0
rolled back updates: 6
test evaluation: skipped because no new accepted weight signal exists
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
promoting validation-only or high/A08-only movement
```
