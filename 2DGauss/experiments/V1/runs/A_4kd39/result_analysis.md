# A_4kd39 Result Analysis

时间: 2026-06-15 22:48 CST

## Verdict

A39 is deployment-complete and becomes the best aggregate fast-probe PSNR checkpoint so far, but it does not break the intended high-route bottleneck.

It should not be interpreted as the high-`input_std`/A08-tail breakthrough. Its useful finding is mechanistic: making `gaussian_control_canvas` trainable from the A37 anchor improves the shared/base and low-route side, raising total PSNR, but it worsens the A18 high-route gap.

## Setup

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/configs/a_4kd39_fast_4kdehaze_a37_teacher_control_refine.toml`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Train log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/logs/train_20260615_222621.log`
- Best checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/checkpoints/best_val.pt`

A39 keeps A37's online native-config A08/A15 teachers and high-route branch loss, locks the low route, disables ordinary reconstruction/proxy losses, and adds `gaussian_control_canvas` to the A37 trainable set.

## Training Outcome

- GPU: physical GPU3
- Preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/logs/train_20260615_222110.log`
- Formal exit code: `0`
- Best/final validation step: `120`
- Best val PSNR: `18.62166244241818`
- Val high-route delta abs mean: `0.015583764696493745`
- Val low-route delta abs mean: `0.0`

Validation rose monotonically from step10 through step120, so A39 is not an early-overfit replay of A38.

## Best-Val Test Metrics

| Metric | A37 | A38 best | A39 best | A39 vs A37 |
|---|---:|---:|---:|---:|
| PSNR | 17.889658453438816 | 17.889819259555214 | 17.900369585886775 | +0.01071113244795838 |
| SSIM | 0.7026848953962326 | 0.7027300244569779 | 0.7017246654629707 | -0.000960229933261888 |
| dark channel L1 | 0.14105536360293627 | 0.14106974665075542 | 0.13971996370702983 | -0.0013353998959064405 |
| luminance L1 | 0.10456957183778286 | 0.10456683240830898 | 0.10452312719076871 | -0.00004644464701414459 |

Evaluation paths:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/metrics/test_eval_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/metrics/test_task_metrics_best_val_gpu.json`

## A18 Record Alignment

| Metric | A37 | A38 best | A39 best | A39 vs A37 |
|---|---:|---:|---:|---:|
| overall gap vs A18 router | +0.042724077624394126 | +0.04288488374079478 | +0.053435210072354966 | +0.01071113244796084 |
| high-route gap vs A18 router | -0.11159004428502844 | -0.1111094511726412 | -0.11858325626995542 | -0.00699321198492698 |
| low-route gap vs A18 router | +0.1591364853806252 | +0.15905604867549208 | +0.18320352678672946 | +0.02406704140610426 |
| A08-oracle gap | -0.19984098726138502 | -0.19848443524653703 | -0.23903555195244477 | -0.03919456469105975 |
| A15-oracle gap | +0.2493535773419097 | +0.2484957851003737 | +0.3025769703157029 | +0.0532233929737932 |

Path:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/metrics/a18_record_alignment_best_val_gpu.json`

A39 improves aggregate PSNR by pushing the A15/low side, not by solving the A08/high side. The planned high-route target was `>-0.10 dB`; A39 ends at `-0.11858325626995542 dB`, worse than A37.

## Route Diagnostics

Path:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd39/metrics/input_std_route_regime_best_val_gpu.json`

Key values:

- Test route counts: high `43`, low `57`
- Overall selected delta mean: `0.005716269994154572`
- High selected delta mean: `0.01329365114919668`
- Low selected delta mean: `0.0`
- High branch L1 vs A08 teacher: `0.013812246537485789`
- Low branch L1 vs A15 teacher: `0.01426901130757311`

Compared with A37, high branch L1 vs A08 teacher improves (`0.014976846487369648 -> 0.013812246537485789`) but high-route PSNR gets worse. The selected high delta also shrinks (`0.014578644511138284 -> 0.01329365114919668`). This suggests the control-canvas update changes the shared/base behavior in a way that helps low-route aggregate cases while diluting the useful A37 high-route correction.

## Decision

A39 is an aggregate numerical improvement and a useful mechanism diagnostic, but it is not promoted as the desired bottleneck breakthrough.

Next direction: do not keep expanding shared control-canvas trainability for this bottleneck. Return to the A37 anchor, keep low route locked, remove `gaussian_control_canvas` from the trainable set, and aim high-route pressure more directly through high-balanced sampling or a stricter A08/offline-target high-route objective.
