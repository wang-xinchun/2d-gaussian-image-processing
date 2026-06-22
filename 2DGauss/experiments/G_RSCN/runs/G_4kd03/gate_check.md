# G_4kd03 Gate Check

- Created: `2026-06-21 19:45:00 CST`
- Status: `SMOKE01 COMPLETE - protected selector worked as guard; promotion failed; formal training blocked`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd03`.
- PASS: First short run uses `G_4kd03_smoke01`, not an A-line ID and not a G_4kd02 smoke extension.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: Added default-off validation router metrics in `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`.
- PASS: Added default-off best-selection baseline gates; G_4kd03 initializes `best_val.pt` from step 0 and only overwrites it if selected validation A08 improves without low/A15 validation regression.
- PASS: Loss pressure is changed conservatively: lower A08 pressure, stronger selected-output A15 anchor, stronger low-branch anchor, tighter non-target delta budget.
- PASS: `allow_high_task_aware_loss_weights=true` is explicitly set for this diagnostic because `gaussian_route_control_low_branch_anchor_weight=0.13` exceeds the default `0.10` cap; formal training remains blocked regardless of this smoke setting.

## Deployment Gate

- PASS: Config uses existing fast-probe dataset view for comparability.
- PASS: Parent checkpoint is `G_4kd02_smoke01/best_val.pt`, preserving the best available branch-isolated high/A08 signal.
- PASS: Short run only: `max_steps=40`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
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

## Smoke01 Execution Result

- PASS: CUDA preflight passed on physical GPU3 with trainable prefix limited to `gaussian_route_control`.
- PASS: Successful short train used physical GPU3 only, single visible GPU, no DDP, no CPU fallback.
- PASS: Successful train exit code is `0`; final step is `40`; peak CUDA reserved is `3.130859375GiB`.
- PASS: Validation router metrics and baseline-gated best selection were live.
- PASS: `best_val.pt` stayed at step `0` because every trained candidate failed the low/A15 no-regression gate.
- FAIL: Best-val A37-aligned evaluation still regresses low-route by `-0.0017883107046233748` and A15-oracle by `-0.0032425407070728884`.
- FAIL: Final checkpoint improves high/A08 more than best-val, but regresses low-route by `-0.005180837680822031` and A15-oracle by `-0.008760974145471945`.

Final verdict:

```text
G_4kd03_smoke01 is deployment-complete and useful as a guardrail probe.
It is not a breakthrough and is not promotable.
Do not launch formal long training from this recipe.
Next G-RSCN probe must change the mechanism, not only the selector.
```
