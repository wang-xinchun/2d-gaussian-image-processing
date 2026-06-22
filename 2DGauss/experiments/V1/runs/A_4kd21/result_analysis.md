# A_4kd21 Result Analysis

## Summary

`A_4kd21` is a GPU-accelerated A08/A15 router evaluation on a larger
non-overlapping `4KDehaze_local_90_5_5` subset:

- Validation offset/limit: `200/200`
- Test offset/limit: `200/200`
- Crop size: `512`
- Device: physical GPU3 through `CUDA_VISIBLE_DEVICES=3`
- GPU policy: memory-gated GPU0/GPU3 usage, `MIN_FREE_MIB=12288`,
  `MAX_GPU_UTIL=100`, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30`

The run confirms that the fixed `A_4kd18` input-std rule generalizes to another
larger held-out block, while the validation-selected richer feature rule is less
stable on test.

## Launch Notes

The original A21 queue waited for low GPU utilization. After the user clarified
that GPU0/GPU3 can be used as long as memory is available, A21 was relaunched
with `MAX_GPU_UTIL=100`.

The first memory-gated smoke exposed a CUDA memory-fraction device-index issue:
`torch.cuda.set_per_process_memory_fraction` cannot receive a bare
`torch.device("cuda")` in this environment. The evaluator now maps bare `cuda`
to `cuda:0` before setting the memory fraction. After that fix, smoke and full
evaluation completed on GPU3.

## Metrics

Full metric file:
`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd21/metrics/a08_a15_gpu_router_offset200_val200_test200.json`

Runtime:

- Full eval elapsed: `673.3705039024353` seconds
- Checkpoint A: `A_4kd08`, step `700`
- Checkpoint B: `A_4kd15`, step `300`

Validation:

| Method | PSNR | SSIM | Choose A15 |
| --- | ---: | ---: | ---: |
| A08 fixed | `18.561300435074248` | `0.725234932899475` | `0/200` |
| A15 fixed | `18.602815493890777` | `0.7281529796123505` | `200/200` |
| Fixed A18 rule | `18.66085112430875` | `0.7291304010152817` | `87/200` |
| Val-selected rule | `18.66436853377257` | `0.728944408595562` | `80/200` |
| Binary oracle | `18.66981253451648` | n/a | `93/200` |

Test:

| Method | PSNR | SSIM | Choose A15 |
| --- | ---: | ---: | ---: |
| A08 fixed | `14.61942131606054` | `0.6726915469765663` | `0/200` |
| A15 fixed | `14.570231632564433` | `0.6721390201151372` | `200/200` |
| Fixed A18 rule | `14.64236451612018` | `0.6756442764401436` | `107/200` |
| Val-selected rule | `14.634471636481214` | `0.6754481521248817` | `117/200` |
| Binary oracle | `14.662330739170043` | n/a | `73/200` |

## Router Rules

Fixed A18 rule:

```text
input_std <= 0.07026463374495506 -> A15, else A08
```

Validation-selected rule:

```text
pred_a_std <= 0.19304919987916946 -> A15, else A08
```

Oracle agreement:

| Split | Rule | Correct Choice | Choose A15 |
| --- | --- | ---: | ---: |
| val | fixed input_std | `162/200` | `87/200` |
| val | selected pred_a_std | `171/200` | `80/200` |
| test | fixed input_std | `164/200` | `107/200` |
| test | selected pred_a_std | `154/200` | `117/200` |

## Interpretation

`A_4kd21` strengthens the A18/A19 conclusion that A08 and A15 have stable
sample-adaptive complementarity. On the offset200 test block, the fixed A18
input-std router improves over A08 by `+0.02294320005963968 dB` PSNR and
`+0.002952729463577275` SSIM, and over A15 by `+0.07213288355574754 dB` PSNR
and `+0.003505256325006423` SSIM.

The validation-selected richer feature rule is slightly better on validation but
worse on test than the fixed input-std rule (`-0.007892879638966122 dB` PSNR
versus fixed on test) and has lower oracle agreement on test. Therefore the
immediate conclusion is not to replace the fixed rule with a greedily selected
feature threshold. A learned/model-internal router should use richer evidence,
but it needs regularization and cross-block validation instead of one-block
threshold selection.

Boundary: this remains a two-checkpoint inference-router diagnostic. It is a
larger held-out positive validation of the routing mechanism, not an
official/full benchmark and not a single-model result.
