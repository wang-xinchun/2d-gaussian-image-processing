# A_4kd41 Result Analysis

时间: 2026-06-16 11:00 CST

## Verdict

A41 is deployment-complete and scientifically useful, but it is not a high-`input_std`/A08-tail breakthrough.

It adds a default-off oracle-weighted high-branch loss and completes the full GPU3 train/eval chain. The mechanism slightly improves high-branch L1-to-A08 relative to A40, but the improvement does not convert into A18 high-route PSNR alignment. Aggregate PSNR is also slightly below A40. Therefore A41 should be recorded as a negative high-route probe, not promoted.

## Setup

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/configs/a_4kd41_fast_4kdehaze_a37_high_route_oracle_weighted.toml`
- Code change: `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Train log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/logs/train_20260615_233454.log`
- Best checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/checkpoints/best_val.pt`

A41 keeps the A37 anchor, freezes low route and `gaussian_control_canvas`, preserves A40 high-`input_std` balanced sampling, and weights high-branch teacher pressure by the current crop's A08-vs-A15 oracle advantage against GT.

## Training Outcome

- GPU: physical GPU3
- Preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/logs/train_20260615_233303.log`
- Formal exit code: `0`
- Best/final validation step: `120`
- Best val PSNR: `18.617112345875956`
- Val high-route delta abs mean: `0.015633893799968063`
- Val low-route delta abs mean: `0.0`

Validation rose monotonically but stayed slightly behind A40 at matched steps:

| Step | A40 Val PSNR | A41 Val PSNR | A41 Val high delta | A41 Val low delta |
|---:|---:|---:|---:|---:|
| 10 | 18.57108618632744 | 18.56986775670201 | 0.0170768712554127 | 0.0 |
| 20 | 18.578070878216156 | 18.57539290368002 | 0.01690997527912259 | 0.0 |
| 30 | 18.584639812089108 | 18.581606014089235 | 0.01675575670786202 | 0.0 |
| 40 | 18.59407341575857 | 18.592767298774664 | 0.01641333687119186 | 0.0 |
| 50 | 18.59880541606789 | 18.5977824195147 | 0.01627900768071413 | 0.0 |
| 60 | 18.60227132599493 | 18.601519461948342 | 0.01616345366463065 | 0.0 |
| 70 | 18.6058992822906 | 18.604527595787804 | 0.016061253817752003 | 0.0 |
| 80 | 18.611465073858966 | 18.60964765608021 | 0.015881563839502632 | 0.0 |
| 90 | 18.61354156494786 | 18.61187273787058 | 0.015808637961745264 | 0.0 |
| 100 | 18.615089937162885 | 18.613600120669805 | 0.015746736628934742 | 0.0 |
| 110 | 18.616667968483842 | 18.61492262502824 | 0.015697065251879395 | 0.0 |
| 120 | 18.619183648392536 | 18.617112345875956 | 0.015633893799968063 | 0.0 |

The oracle weights were active during training. Example final-step telemetry: weight mean `1.521048903465271`, A08-better fraction `1.0`, oracle advantage mean `0.00033687055110931396`, high hard mean `1.0`.

## Best-Val Test Metrics

| Metric | A37 | A40 | A41 | A41 vs A37 | A41 vs A40 |
|---|---:|---:|---:|---:|---:|
| PSNR | 17.889658453438816 | 17.900562951956445 | 17.900201657598004 | +0.010543204159187525 | -0.0003612943584414552 |
| SSIM | 0.7026848953962326 | 0.7017474815249443 | 0.7017551222443581 | -0.0009297731518745 | +0.0000076407194138 |
| dark channel L1 | 0.14105536360293627 | 0.1396497754007578 | 0.1396693666651845 | -0.00138599693775177 | +0.0000195912644267 |
| luminance L1 | 0.10456957183778286 | 0.10450994782149792 | 0.10450344946235418 | -0.00006612237542868 | -0.00000649835914374 |

Evaluation paths:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/test_eval_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/test_task_metrics_best_val_gpu.json`

## A18 Record Alignment

| Metric | A37 | A40 | A41 | A41 vs A37 | A41 vs A40 |
|---|---:|---:|---:|---:|---:|
| overall gap vs A18 router | +0.042724077624394126 | +0.0536285761420247 | +0.05326728178358334 | +0.010543204159189211 | -0.000361294358441365 |
| high-route gap vs A18 router | -0.11159004428502844 | -0.1180115677202543 | -0.11813195131188482 | -0.006541907026856378 | -0.00012038359163052403 |
| low-route gap vs A18 router | +0.1591364853806252 | +0.18311149168725271 | +0.182568457627533 | +0.023431972246907795 | -0.000543034059719727 |
| A08-oracle gap | -0.19984098726138502 | -0.23833660113789967 | -0.23811165869770295 | -0.03827067143631793 | +0.00022494244019671816 |
| A15-oracle gap | +0.2493535773419097 | +0.3023396530841825 | +0.3014789718231976 | +0.0521253944812879 | -0.0008606812609848746 |

Path:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/a18_record_alignment_best_val_gpu.json`

A41 does not beat A37 on the primary high-route target and remains far below the planned `>-0.10 dB` line. It is almost indistinguishable from A40 on high-route gap, with a tiny A08-oracle improvement that is too small and too localized to count as a breakthrough.

## Route Diagnostics

Path:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/metrics/input_std_route_regime_best_val_gpu.json`

Key values:

- Test route counts: high `43`, low `57`
- Overall selected delta mean: `0.005682019237428904`
- High selected delta mean: `0.013213998226578845`
- Low selected delta mean: `0.0`
- High branch L1 vs A08 teacher: `0.01348984414754912`
- Low branch L1 vs A15 teacher: `0.014252835606927411`

Compared with A40, high branch L1 vs A08 improves only by about `0.000033031131119229395`, while high selected delta is nearly unchanged. This is the important negative result: the oracle weighting nudges the teacher-distance diagnostic but does not change the route's PSNR behavior.

## Decision

A41 is accepted as engineering-complete but not promoted. The default-off oracle-weighting hook is useful infrastructure, but this specific crop-level reweighting is not the missing mechanism.

The next experiment should stop trying to extract a large high-route gain by scalar reweighting alone. The useful direction is stronger: offline A08-oracle/high-route worst-case target caching, per-sample trust anchors from saved A18/A36 records, or additional route-conditioned internal capacity that can change representation before the late route head.
