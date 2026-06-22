# G_4kd12 Gate Check

- Created: `2026-06-22 16:55:00 CST`
- Status: `SMOKE01 COMPLETE - NOT PROMOTED`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd12`.
- PASS: First short run uses `G_4kd12_smoke01`, not an A-line ID and not a G11 rerun.
- PASS: Config, launch script, evaluator, and local logs remain under the G-RSCN line.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: G12 changes transfer-aware validation and guard selection, not model structure.
- PASS: A18 `val_records` are used for training-time transfer guard.
- PASS: A18 `test_records` remain reserved for post-training evaluation.
- PASS: Full transfer-aware step guard remains enabled.
- PASS: Two-phase protect/target guard and candidate projection remain enabled.
- PASS: `input_std` remains a prior and sampler feature, not the architecture identity.

## Deployment Gate

- PASS: Config uses the existing fast-probe dataset view for comparability.
- PASS: Parent is A37 best-val.
- PASS: Short run only: `max_steps=16`, `batch_size=1`, `grad_accum=2`.
- PASS: Python compile, TOML parse, bash syntax checks, and CUDA preflight passed.
- PASS: CUDA preflight used physical GPU3, loaded A37 plus expected `gaussian_route_control.*` missing keys only, and kept trainable prefix only `gaussian_route_control`.
- PASS: Transfer validation produced all planned `transfer_val_*` metrics with `100/100` A18 `val_records` matched.
- PASS: GPU gate immediately before launch used physical GPU3 only, single visible GPU, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`, and the user-approved short-run util exception.
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
aggregate PSNR alone cannot promote
```

Result:

```text
best_val:
PSNR 17.889658453438816 vs A37 17.889658453438816 -> PASS / identical parent
high-route -0.11159004428502844 vs A37 -0.11159004428502844 -> PASS / identical parent
A08-oracle -0.19984098726138502 vs A37 -0.19984098726138502 -> PASS / identical parent
low-route 0.1591364853806252 vs A37 0.1591364853806252 -> PASS / identical parent
A15-oracle 0.2493535773419097 vs A37 0.2493535773419097 -> PASS / identical parent

final:
PSNR 17.889701410936713 vs A37 17.889658453438816 -> PASS by +0.000042957497897
high-route -0.11164274339298112 vs A37 -0.11159004428502844 -> FAIL by -0.000052699107953
A08-oracle -0.2000703652075532 vs A37 -0.19984098726138502 -> FAIL by -0.000229377946168
low-route 0.15925160487943105 vs A37 0.1591364853806252 -> PASS by +0.000115119498806
A15-oracle 0.2496285242921635 vs A37 0.2493535773419097 -> PASS by +0.000274946950254
```

Promotion decision: `NOT PROMOTED`. `best_val` is the protected A37 parent because no full transfer gate candidate was accepted. `final` improves aggregate/low/A15 slightly but still regresses high-route and A08-oracle.
