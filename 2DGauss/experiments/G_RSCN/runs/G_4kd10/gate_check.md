# G_4kd10 Gate Check

- Created: `2026-06-22 00:15:44 CST`
- Status: `COMPLETE - all updates rolled back; formal training blocked`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd10`.
- PASS: First short run uses `G_4kd10_smoke01`, not an A-line ID and not a G_4kd09 extension.
- PASS: Artifacts, config, launcher, evaluator, and logs remain under the G-RSCN line.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: `input_std` remains a prior and sampler feature, not a replacement for Gaussian route control.
- PASS: `step_no_harm_guard` is enabled and single-process only.
- PASS: Guard checks high-route, A08-oracle, low-route, A15-oracle, and low-A15 validation gaps every two steps.
- PASS: Guard uses full `24` validation batches and restores optimizer state on rollback.
- PASS: G10 changes only the G09 bottleneck: relaxed A08 margin and slightly higher A08/high pressure, while high/low/A15 margins remain strict.

## Deployment Gate

- PASS: Config uses the existing fast-probe dataset view for comparability.
- PASS: Parent is A37 best-val, matching G08/G09 A37-start diagnostics.
- PASS: Short run only: `max_steps=18`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- PASS: Preflight completed on physical GPU3.
- PASS: Training completed with exit code `0`.
- FAIL: No candidate update passed all five validation-margin gates.
- PASS: Formal long training remains blocked.

## GPU Gate

- PASS: Approved physical GPUs are only `0` and `3`; GPU1/2 are forbidden.
- PASS: Single visible GPU only; no DDP and no `CUDA_VISIBLE_DEVICES=0,3`.
- PASS: Default memory policy is `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- PASS: Default free-memory gate is `TWODGAUSS_MIN_FREE_MIB=8192`.
- PASS: User-approved short-run exception allows `TWODGAUSS_MAX_GPU_UTIL=100` on GPU0/3 when free memory is sufficient and the run will not evict other users.

## Smoke01 Promotion Check

```text
must keep high-route at or above A37
must keep A08-oracle at or above A37
must keep low-route at or above A37
must keep A15-oracle at or above A37
step_no_harm_guard must write accepted/rollback evidence
aggregate PSNR alone cannot promote
```

Observed:

```text
guard checks: 9
accepted updates: 0
rolled back updates: 9
test evaluation: skipped because no new accepted weight signal exists
best gate-count candidate: step 4 passed high/A08 but failed low/A15/low-A15
closest all-gate candidate: step 18, worst margin -0.00016272430419922435
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
