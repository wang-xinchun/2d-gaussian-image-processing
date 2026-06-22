# A_4kd22 Gate Check

## Status

Completed CPU-only saved-record cross-block router diagnostic.

## Checks

- Script py_compile passed:
  `/home/ldj/wxc/2DGauss/scripts/evaluate_router_records_crossblock_v1_t01.py`
- Inputs have matching common router feature keys across A18/A19/A21 records.
- No GPU was used.
- Output JSON was written:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd22/metrics/a18_a19_a21_crossblock_saved_router.json`

## Key Result

Aggregate leave-one-source-out test:

- Fixed A18 input_std rule: `PSNR=15.761350330784339`,
  oracle agreement `0.77`, choose A15 `204/400`.
- Logistic `input_mean_std`: `PSNR=15.758408523282599`,
  oracle agreement `0.7575`, choose A15 `219/400`.
- Logistic `input_std`: `PSNR=15.753757145011031`,
  oracle agreement `0.7525`, choose A15 `223/400`.
- Logistic `all_common`: `PSNR=15.748658098940552`,
  oracle agreement `0.7075`, choose A15 `237/400`.

## Boundary

This is an offline saved-record diagnostic. It does not produce a single-model
checkpoint. Its role is to determine whether learned routing should be
free-form or constrained. The evidence favors a constrained input-std-like
single-model router/distillation next.
