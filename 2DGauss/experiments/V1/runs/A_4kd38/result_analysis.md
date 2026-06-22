# A_4kd38 Result Analysis

## Verdict

A38 is deployment-complete and establishes a new tiny best PSNR number, but it is not a meaningful high-route bottleneck breakthrough.

It should not trigger medium training in this form. The useful finding is negative: from A37, standard reconstruction/proxy loss applied only to the high input-std decoder-head path gives a microscopic early gain, then steadily degrades validation and final high-route behavior.

## Setup

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/configs/a_4kd38_fast_4kdehaze_a37_standard_loss_high_only_short.toml`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/checkpoints/best_val.pt`
- Train log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/logs/train_20260615_125822.log`
- Best checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/checkpoints/best_val.pt`
- Final checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/checkpoints/final.pt`

A38 disables teacher loading and route branch loss, keeps natural sampling, and trains only:

- `input_std_route_decoder_heads.high_feature`
- `input_std_route_decoder_heads.high_head`

The low route, shared route encoder, Gaussian/control modules, and base decoder are frozen.

## Training Outcome

- GPU: physical GPU3
- Preflight log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/logs/train_20260615_125507.log`
- Formal exit code: `0`
- Best validation step: `10`
- Best val PSNR: `18.562811833496248`
- Final step: `120`
- Final val PSNR: `18.560370419887278`

Validation decayed monotonically after step10 while the high-route delta slowly increased. This is the important mechanism signal: merely pushing the high head with standard GT/proxy loss is not the missing A08-tail mechanism.

## Best-Val Test Metrics

| Metric | A37 | A38 best | Delta |
|---|---:|---:|---:|
| PSNR | 17.889658453438816 | 17.889819259555214 | +0.00016080611639779363 |
| SSIM | 0.7026848953962326 | 0.7027300244569779 | +0.000045129060745274785 |
| loss | 0.11911950781941413 | 0.11911884762346744 | -0.0000006601959466889795 |
| total loss | 0.12235063299536705 | 0.12234997287392617 | -0.0000006601214408846756 |

Evaluation paths:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/metrics/test_eval_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/metrics/test_task_metrics_best_val_gpu.json`

## A18 Record Alignment

| Metric | A37 | A38 best | Delta |
|---|---:|---:|---:|
| overall gap vs A18 router | +0.042724077624394126 | +0.04288488374079478 | +0.00016080611640065245 |
| high-route gap vs A18 router | -0.11159004428502844 | -0.1111094511726412 | +0.0004805931123872398 |
| low-route gap vs A18 router | +0.1591364853806252 | +0.15905604867549208 | -0.00008043670513310786 |
| A08-oracle gap | -0.19984098726138502 | -0.19848443524653703 | +0.0013565520148479948 |
| A15-oracle gap | +0.2493535773419097 | +0.2484957851003737 | -0.0008577922415359818 |

Alignment paths:

- Best-val: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/metrics/a18_record_alignment_best_val_gpu.json`
- Final: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/metrics/a18_record_alignment_final_gpu.json`

The high-route target was `>-0.10 dB`. A38 best remains at `-0.1111094511726412 dB`, so the bottleneck is not broken. Final is worse, with high-route gap `-0.11144634980848278 dB`.

## Route Diagnostics

- Route-regime path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd38/metrics/input_std_route_regime_best_val_gpu.json`
- Test route counts: high `43`, low `57`
- Selected delta mean: `0.006315981186926365`
- High selected delta mean: `0.01468832834168922`
- Low selected delta mean: `0.0`

The low route remains protected. The high route receives a slightly larger selected delta than A37, but the extra movement barely improves the A18 high-route gap and does not cross the scientific target.

## Interpretation

A38 answers the short-training question: do not continue this exact standard-loss high-head-only recipe into medium training.

The likely reason is that A37's improvement came from native A08/A15 teacher-compression pressure, not from ordinary GT reconstruction. GT loss on the selected high output is too weak and too entangled with the existing A37 compromise: it nudges the high head in a direction that marginally helps a few A08-oracle samples but quickly degrades validation and does not recover the missing A08-like high-route representation.

## Next Direction

The next high-probability run should return to route-aware teacher signal or add route-conditioned feature capacity, not extend A38.

Recommended A39 short/mid candidate:

- Initialize from A37 best-val, not A38 final.
- Keep low route locked.
- Use A08 teacher pressure on high route again, but only if GPU memory allows teacher loading.
- If GPU3 remains tight, use cached/offline A08 high-route targets or a teacher-free precomputed target dataset instead of online teachers.
- Move capacity earlier than the terminal high head: route-conditioned feature expert or Gaussian/control branch modulation is more plausible than another output-head-only standard-loss pass.
