# G_4kd02 Result Analysis

- Created: `2026-06-21 19:05:00 CST`
- Status: `smoke01 complete; not promoted`

## Smoke01 Plan

`G_4kd02_smoke01` tests whether branch-level Gaussian route-control supervision can break the shared-control interference seen in `G_4kd01_smoke04`.

Expected signal:

- high branch anchor should activate on A08-better crops;
- low branch anchor should activate on A15-better crops;
- non-target delta over-budget metrics should stay small;
- final A37-aligned evaluation must improve high/A08 without losing low/A15.

Blocking rule:

- Do not run formal long training until best/final A37-aligned GPU evaluation is recorded here.

## Preflight Result

- Status: `pass`
- Device: `cuda`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/best_val.pt`
- Trainable params: `37698`
- Trainable prefix: `gaussian_route_control`
- Preflight branch signal:
  - low-branch/A15 probability mean `0.7696662545204163`;
  - low-branch weighted anchor loss `0.0007020853227004409`;
  - high-branch/A08 probability mean `0.0` on this sampled A15-better crop.

## Training Result

- Train status: `exit_code=0`
- Physical GPU: `3`
- CUDA policy: single visible GPU, `TWODGAUSS_CUDA_MEMORY_FRACTION=0.35`, no CPU fallback.
- Final step: `40`
- Best step: `10`
- Best validation PSNR: `16.84168937420473`
- Best validation loss: `0.06276035433014233`
- Peak CUDA reserved: `3.091796875GiB`
- Branch anchor activation:
  - high/A08 branch active `13/40` steps;
  - low/A15 branch active `27/40` steps;
  - non-target delta over-budget weighted loss stayed `0.0`.

## A37-Aligned Evaluation

Best checkpoint:

| metric | G_4kd02 best | A37 | delta |
| --- | ---: | ---: | ---: |
| test PSNR | 17.888740401218694 | 17.889658453438816 | -0.0009180522201219787 |
| test SSIM | 0.7027190274000168 | 0.7026848953962326 | +0.000034132003784215215 |
| overall gap | 0.04180602540427511 | 0.042724077624394126 | -0.0009180522201190158 |
| high-route gap | -0.111354498048944 | -0.11159004428502844 | +0.0002355462360844468 |
| low-route gap | 0.15734817467600182 | 0.1591364853806252 | -0.0017883107046233748 |
| A08-oracle gap | -0.19803029212725382 | -0.19984098726138502 | +0.0018106951341312072 |
| A15-oracle gap | 0.2461110366348368 | 0.2493535773419097 | -0.0032425407070728884 |

Final checkpoint:

| metric | G_4kd02 final | A37 | delta |
| --- | ---: | ---: | ---: |
| test PSNR | 17.887757378098673 | 17.889658453438816 | -0.0019010753401431657 |
| test SSIM | 0.7027128818631172 | 0.7026848953962326 | +0.00002798646688462192 |
| overall gap | 0.04082300228425266 | 0.042724077624394126 | -0.0019010753401414657 |
| high-route gap | -0.11133933763390716 | -0.11159004428502844 | +0.0002507066511212852 |
| low-route gap | 0.1556121359067241 | 0.1591364853806252 | -0.003524349473901095 |
| A08-oracle gap | -0.19711797674249362 | -0.19984098726138502 | +0.0027230105188914 |
| A15-oracle gap | 0.24351346589962913 | 0.2493535773419097 | -0.0058401114422805545 |

## Interpretation

`G_4kd02_smoke01` proves branch-isolated Gaussian route-control supervision is wired and directionally stronger on high/A08 than the paired scalar-anchor probe. It does not break the bottleneck: the best checkpoint still spends low-route and A15-oracle margin, and the final checkpoint spends more.

The next probe should protect A15/low-route before extending training. The priority is not more steps; it is a hard anti-regression mechanism or checkpoint-selection gate that rejects low/A15 loss while preserving the new high/A08 movement.
