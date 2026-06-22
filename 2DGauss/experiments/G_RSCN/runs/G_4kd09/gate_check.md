# G_4kd09 Gate Check

- Created: `2026-06-22 00:07:34 CST`
- Status: `COMPLETE - all updates rolled back by positive-margin guard; formal training blocked`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd09`.
- PASS: First short run uses `G_4kd09_smoke01`, not an A-line ID and not a G_4kd08 extension.
- PASS: Artifacts, config, launcher, evaluator, and logs remain under the G-RSCN line.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: `input_std` remains a prior and sampler feature.
- PASS: `step_no_harm_guard` is enabled and single-process only.
- PASS: Guard checks high-route, A08-oracle, low-route, A15-oracle, and low-A15 validation gaps every two steps.
- PASS: Guard uses full `24` validation batches and restores optimizer state on rollback.
- PASS: Positive-margin thresholds are stricter than G08 validation non-regression gates.

## Deployment Gate

- PASS: Config uses the existing fast-probe dataset view for comparability.
- PASS: Parent is A37 best-val for a clean A37 restart.
- PASS: Short run only: `max_steps=16`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- PASS: Training completed with exit code `0`.
- FAIL: No candidate update passed all positive-margin validation gates.

## GPU Gate

- PASS: Approved physical GPUs are only `0` and `3`; GPU1/2 are forbidden.
- PASS: Single visible GPU only; no DDP and no `CUDA_VISIBLE_DEVICES=0,3`.
- PASS: `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`.
- PASS: No CPU fallback was used.

## Smoke01 Promotion Check

```text
guard checks: 8
accepted updates: 0
rolled back updates: 8
test evaluation: skipped because no new accepted weight signal exists
closest bottleneck: step 4 missed A08 margin by 2.738494873047448e-05
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
