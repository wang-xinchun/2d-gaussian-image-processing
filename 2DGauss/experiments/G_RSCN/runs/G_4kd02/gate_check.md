# G_4kd02 Gate Check

- Created: `2026-06-21 19:05:00 CST`
- Status: `SMOKE01 COMPLETE - not promoted; formal training blocked`

## Naming Gate

- PASS: New mechanism version uses `experiments/G_RSCN/runs/G_4kd02`.
- PASS: First short run uses `G_4kd02_smoke01`, not an A-line ID and not a G_4kd01 smoke extension.

## Mechanism Gate

- PASS: Gaussian core remains immutable: `GaussianParameterHead`, `TextureAwareGaussianRenderer`, and Gaussian attention/control paths are unchanged.
- PASS: `GaussianRouteControl` remains the only trainable prefix.
- PASS: Added auxiliary `gaussian_route_control_high_restored` and `gaussian_route_control_low_restored` outputs for branch-level supervision.
- PASS: Added default-off branch-level teacher loss in `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`.
- PASS: New loss is G-line specific and disabled unless `gaussian_route_control_*branch_anchor_weight` is positive.

## Deployment Gate

- PASS: Config uses existing fast-probe dataset view for comparability.
- PASS: Parent checkpoint is `G_4kd01_smoke02/best_val.pt`, preserving the cleanest high/A08 signal.
- PASS: Short run only: `max_steps=40`, `batch_size=1`, `grad_accum=2`, `max_val_batches=24`.
- PASS: Formal long training remains blocked until A37-aligned evaluation passes all promotion gates.

## CUDA Preflight

```text
preflight = ok
device = cuda
trainable_parameter_prefixes = ["gaussian_route_control"]
trainable params = 37698
total params = 206104
init checkpoint = G_4kd01_smoke02 best, strict load with no missing/unexpected keys
gaussian_route_control_low_branch_anchor_probability_mean = 0.7696662545204163
loss_gaussian_route_control_low_branch_anchor_weighted = 0.0007020853227004409
loss_gaussian_route_control_branch_anchor_weighted = 0.0007020853227004409
loss_gaussian_route_control_non_target_delta_weighted = 0.0
```

Interpretation: branch-level G-control anchor plumbing is live on CUDA. The sampled preflight crop is A15-better, so low-branch/A15 anchor activates and high-branch/A08 anchor correctly stays inactive.

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

Observed best checkpoint, step `10`:

```text
test_psnr = 17.888740401218694  (A37 delta -0.0009180522201219787)
test_ssim = 0.7027190274000168  (A37 delta +0.000034132003784215215)
overall_gap = 0.04180602540427511  (A37 delta -0.0009180522201190158)
high_route_gap = -0.111354498048944  (A37 delta +0.0002355462360844468)
low_route_gap = 0.15734817467600182  (A37 delta -0.0017883107046233748)
a08_oracle_gap = -0.19803029212725382  (A37 delta +0.0018106951341312072)
a15_oracle_gap = 0.2461110366348368  (A37 delta -0.0032425407070728884)
```

Observed final checkpoint, step `40`:

```text
test_psnr = 17.887757378098673  (A37 delta -0.0019010753401431657)
test_ssim = 0.7027128818631172  (A37 delta +0.00002798646688462192)
overall_gap = 0.04082300228425266  (A37 delta -0.0019010753401414657)
high_route_gap = -0.11133933763390716  (A37 delta +0.0002507066511212852)
low_route_gap = 0.1556121359067241  (A37 delta -0.003524349473901095)
a08_oracle_gap = -0.19711797674249362  (A37 delta +0.0027230105188914)
a15_oracle_gap = 0.24351346589962913  (A37 delta -0.0058401114422805545)
```

Gate verdict: `FAIL`. High-route and A08 move in the desired direction, but low-route and A15-oracle regress below A37. Formal long training remains blocked.

Still forbidden:

```text
formal long training
multi-GPU/DDP
physical GPU1/GPU2
CUDA_VISIBLE_DEVICES=0,3
CPU fallback
promoting aggregate-only or high/A08-only movement
```
