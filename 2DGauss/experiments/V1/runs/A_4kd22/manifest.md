# V1/T01/A_4kd22

## Purpose

`A_4kd22` evaluates whether the A18/A19/A21 router milestone can be converted
into a learned saved-record router before touching the training graph.

It uses only saved A08/A15 router records from completed evaluations. No model
forward pass and no GPU are required.

## Sources

- A18 fast-probe:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd18/metrics/a08_a15_router_val_to_test_cpu.json`
- A19 offset100 tail:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd19/metrics/a08_a15_router_tail_offset100_cpu.json`
- A21 offset200 larger block:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/metrics/a08_a15_gpu_router_offset200_val200_test200.json`

## Method

The diagnostic trains lightweight logistic routers on validation records from
all but one source, then evaluates on the held-out source's test records.
This leave-one-source-out protocol is intentionally stricter than A20 because it
tests cross-block stability rather than only fitting pooled records.

Feature sets:

- `input_std`
- `input_mean_std`
- `input_residual`
- `prediction_stats`
- `control_stats`
- `all_common`

The fixed comparison rule is the A18 rule:

```text
input_std <= 0.07026463374495506 -> A15, else A08
```

## Key Files

- Script:
  `/home/ldj/wxc/2DGauss/scripts/evaluate_router_records_crossblock_v1_t01.py`
- Metrics:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd22/metrics/a18_a19_a21_crossblock_saved_router.json`
- Stdout log:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd22/logs/a18_a19_a21_crossblock_saved_router.stdout.log`
- Result analysis:
  `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd22/result_analysis.md`

## Result

The fixed A18 `input_std` rule remains the best aggregate leave-one-source-out
method:

- Fixed rule aggregate PSNR: `15.761350330784339`
- Best logistic aggregate PSNR (`input_mean_std`): `15.758408523282599`
- Fixed rule oracle agreement: `0.77`
- Best logistic oracle agreement: `0.7575`

Conclusion: A22 supports moving toward a constrained/single-model
input-std-like router or distillation mechanism. It does not support replacing
the fixed rule with unconstrained richer-feature logistic routing.
