# A_4kd40 Result Analysis

时间: 2026-06-15 23:14 CST

## Verdict

A40 is deployment-complete and sets a new aggregate fast-probe PSNR record, but it is not the desired high-`input_std`/A08-tail breakthrough.

It improves slightly over A39 in overall test PSNR while keeping the low route locked, which is useful. The key failure is that high-route A18 alignment remains worse than A37 and still misses the `>-0.10 dB` target. Therefore A40 should be recorded as an aggregate numeric best plus a negative high-tail diagnostic.

## Setup

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/configs/a_4kd40_fast_4kdehaze_a37_high_route_balanced_teacher.toml`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Train log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/logs/train_20260615_225430.log`
- Best checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/checkpoints/best_val.pt`

A40 returns to the A37 anchor, keeps `gaussian_control_canvas` frozen, excludes low-route trainable prefixes, and enables high-`input_std` balanced sampling/crop targeting. With `batch_size=1`, the balanced sampler forms all-high training batches in the formal run.

## Training Outcome

- GPU: physical GPU3
- Preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/logs/train_20260615_225246.log`
- Formal exit code: `0`
- Best/final validation step: `120`
- Best val PSNR: `18.619183648392536`
- Val high-route delta abs mean: `0.015635613431222738`
- Val low-route delta abs mean: `0.0`

Validation rose monotonically:

| Step | Val PSNR | Val high delta | Val low delta |
|---:|---:|---:|---:|
| 10 | 18.57108618632744 | 0.01708358958363533 | 0.0 |
| 20 | 18.578070878216156 | 0.01687158620916307 | 0.0 |
| 30 | 18.584639812089108 | 0.016702666226774455 | 0.0 |
| 40 | 18.59407341575857 | 0.016419715965166688 | 0.0 |
| 50 | 18.59880541606789 | 0.016303756311535834 | 0.0 |
| 60 | 18.60227132599493 | 0.016188389007002115 | 0.0 |
| 70 | 18.6058992822906 | 0.016080607157200576 | 0.0 |
| 80 | 18.611465073858966 | 0.0158918091095984 | 0.0 |
| 90 | 18.61354156494786 | 0.01582374457735568 | 0.0 |
| 100 | 18.615089937162885 | 0.015764110116288067 | 0.0 |
| 110 | 18.616667968483842 | 0.015709660286083817 | 0.0 |
| 120 | 18.619183648392536 | 0.015635613431222738 | 0.0 |

## Best-Val Test Metrics

| Metric | A37 | A39 | A40 | A40 vs A37 |
|---|---:|---:|---:|---:|
| PSNR | 17.889658453438816 | 17.900369585886775 | 17.900562951956445 | +0.010904498517629 |
| SSIM | 0.7026848953962326 | 0.7017246654629707 | 0.7017474815249443 | -0.0009374138712883 |
| dark channel L1 | 0.14105536360293627 | 0.13971996370702983 | 0.1396497754007578 | -0.00140558820217847 |
| luminance L1 | 0.10456957183778286 | 0.10452312719076871 | 0.10450994782149792 | -0.00005962401628494 |

Evaluation paths:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/test_eval_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/test_task_metrics_best_val_gpu.json`

## A18 Record Alignment

| Metric | A37 | A39 | A40 | A40 vs A37 |
|---|---:|---:|---:|---:|
| overall gap vs A18 router | +0.042724077624394126 | +0.053435210072354966 | +0.0536285761420247 | +0.010904498517630574 |
| high-route gap vs A18 router | -0.11159004428502844 | -0.11858325626995542 | -0.1180115677202543 | -0.00642152343522586 |
| low-route gap vs A18 router | +0.1591364853806252 | +0.18320352678672946 | +0.18311149168725271 | +0.0239750063066275 |
| A08-oracle gap | -0.19984098726138502 | -0.23903555195244477 | -0.23833660113789967 | -0.03849561387651465 |
| A15-oracle gap | +0.2493535773419097 | +0.3025769703157029 | +0.3023396530841825 | +0.0529860757422728 |

Path:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/a18_record_alignment_best_val_gpu.json`

A40's aggregate gain is again dominated by the low/A15 side. The high-route gap is slightly better than A39 but still worse than A37, and it remains far from the `>-0.10 dB` target.

## Route Diagnostics

Path:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/metrics/input_std_route_regime_best_val_gpu.json`

Key values:

- Test route counts: high `43`, low `57`
- Overall selected delta mean: `0.0056926918867975474`
- High selected delta mean: `0.013238818341389645`
- Low selected delta mean: `0.0`
- High branch L1 vs A08 teacher: `0.013522875278668349`
- Low branch L1 vs A15 teacher: `0.014295937117646661`

The high branch moves closer to A08 teacher by L1 than A37, but this still does not translate into high-route PSNR alignment. Balanced high sampling mainly preserves or slightly improves the aggregate/low-side record while failing to recover the missing A08-tail fidelity.

## Decision

A40 is the new aggregate fast-probe numeric best, but it is not promoted as the high-route bottleneck breakthrough.

The next experiment should stop assuming more high-sample exposure alone solves the gap. The likely next productive move is to keep A37/A40 as anchors and use stricter high-route offline targets or per-sample A08-tail trust anchors, especially for the A08-oracle/high-route worst cases, while explicitly preventing the shared/base path from turning high improvements into low-route aggregate-only gains.
