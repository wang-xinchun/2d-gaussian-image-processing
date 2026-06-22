# G_4kd12 Manifest

- Created: `2026-06-22 16:55:00 CST`
- Architecture line: `G-RSCN`
- Run package: `experiments/G_RSCN/runs/G_4kd12`
- First smoke run: `G_4kd12_smoke01`
- Status: `smoke01 complete; not promoted; formal training blocked`

## Difference From G_4kd11

`G_4kd11_smoke01` proved the route-control path can accept repeated five-metric validation-safe updates, but those updates did not transfer to A37-aligned test high-route and A08-oracle gates. `G_4kd12` therefore keeps the model structure unchanged and changes the validation guard surface.

Kept from G11:

- initialize from `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`;
- train only `gaussian_route_control`;
- keep the immutable Gaussian core;
- keep two-phase protect/target guard and interpolation projection;
- keep single approved GPU policy with no CPU fallback.

New in G12:

- compute `transfer_val_*` metrics from A18 `val_records`;
- use `transfer_val_a08_oracle_gap_mean` as best-selection metric;
- require transfer-aware high/A08 positive deltas before saving best-val;
- keep A18 `test_records` only for post-training A37-aligned evaluation.

## Smoke01 Artifacts

- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/configs/g_4kd12_smoke01_a37_transfer_guard_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/launch_smoke01_gpu.sh`
- Evaluator: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/evaluate_smoke01_gpu.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Transfer records: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd18/metrics/a08_a15_router_val_to_test_cpu.json`
- Scope: physical GPU0/3 only, single visible GPU, `max_steps=16`, `batch_size=1`, `grad_accum=2`, ordinary val `24` batches, transfer val `100` A18 records.

## Smoke01 Result

- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/logs/smoke01_20260622_170257.log`
- Train exit code: `0`
- Physical GPU: `3`
- Final step: `16`
- Trainable prefix: `gaussian_route_control`
- Trainable params: `37698`
- Peak CUDA reserved: `3.15234375GiB`
- A18 transfer-val records matched: `100/100`
- Guard checks: `8`
- Accepted updates: `6`
- Accepted full-gate updates: `0`
- Accepted protect-only updates: `6`
- Rollbacks: `2`
- Projection-accepted updates: `0`
- Best-val step: `0`, because no candidate passed the full transfer gate required for promotion to `best_val.pt`.

Transfer-val guard baseline:

```text
high-route: 0.03710367872908309
A08-oracle: 0.07303136937758502
low-route: 0.417457762218657
A15-oracle: 0.3613124905210553
low-A15-oracle: 0.4313206589012815
```

Best transfer-val A08 candidate during training was step `8`, but it was only `protect_only`:

```text
step 8 A08-oracle: 0.07311837813433479
required full-gate A08-oracle: 0.07313136937758502
high-route: 0.037214975099305846
low-route: 0.41819766211131265
A15-oracle: 0.36203632932720764
low-A15-oracle: 0.43207001267817985
```

A37-aligned GPU evaluation:

```text
A37 baseline:
PSNR 17.889658453438816
high-route -0.11159004428502844
A08-oracle -0.19984098726138502
low-route 0.1591364853806252
A15-oracle 0.2493535773419097

G_4kd12 best_val:
PSNR 17.889658453438816 (+0.0)
high-route -0.11159004428502844 (+0.0)
A08-oracle -0.19984098726138502 (+0.0)
low-route 0.1591364853806252 (+0.0)
A15-oracle 0.2493535773419097 (+0.0)

G_4kd12 final:
PSNR 17.889701410936713 (+0.000042957497897)
high-route -0.11164274339298112 (-0.000052699107953)
A08-oracle -0.2000703652075532 (-0.000229377946168)
low-route 0.15925160487943105 (+0.000115119498806)
A15-oracle 0.2496285242921635 (+0.000274946950254)
```

Evaluation logs:

- `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/logs/eval_smoke01_best_val_20260622_172216.log`
- `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd12/logs/eval_smoke01_final_20260622_172803.log`

## Promotion Boundary

`G_4kd12_smoke01` is promotable only if post-training A37-aligned GPU evaluation passes all route/oracle gates simultaneously:

```text
high-route >= A37
A08-oracle >= A37
low-route >= A37
A15-oracle >= A37
```

Aggregate PSNR alone cannot promote. Formal long training remains blocked until the short-run result passes the A37-aligned gate.

Smoke01 did not pass this boundary. The transfer-aware guard prevented any full-gate candidate from becoming `best_val`, and the protect-only final checkpoint still regressed high-route and A08-oracle on A37-aligned test.
