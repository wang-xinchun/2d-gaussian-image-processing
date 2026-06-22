# A_4kd37 Result Analysis

时间: 2026-06-15 00:26:27 CST

## Summary

A37 is promoted as the new best fast-probe single-checkpoint/shared-backbone candidate.

It starts from A33, keeps the low route locked, and uses native-config A08/A15 teachers to run a high-route-only A36-teacher compression probe. The main result is a real aggregate breakthrough:

- Test PSNR: `17.889658453438816`
- Previous A36/A18 router line: `17.8470899392002` / `17.84693437581442`
- A37 vs A18 router overall gap: `+0.042724077624394126 dB`
- Standard-loss re-eval: `loss_mean=0.11911950781941413`, `total_loss_mean=0.12235063299536705`
- SSIM: `0.7026848953962326`

This is not just a validation-log artifact. The result is from full 100-sample test evaluation on GPU3, with A18-record alignment and route-regime diagnostics.

## What Changed

- Training script now supports native teacher configs through `loss.router_distillation_config_a` and `loss.router_distillation_config_b`.
- A37 loads A08 teacher with A08's own config and A15 teacher with A15's own config, avoiding student-config pollution.
- The final deployed training recipe disables ordinary reconstruction/proxy losses and keeps only high/A08 route branch supervision plus delta trust. This avoids low-route samples updating shared late modules.
- Low route is locked by excluding `input_std_route_decoder_heads.low_*` from trainable prefixes.

## Training

Formal training:

- Log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/logs/train_20260614_234511.log`
- Exit code: `0`
- Best step: `120`
- Best validation PSNR: `18.56340678071067`
- Validation high delta abs mean: `0.017279619527980686`
- Validation low delta abs mean: `0.0`

Preflight notes:

- First preflight OOMed at `batch_size=2`; deployment changed to `batch_size=1`, `grad_accum=8`.
- Second preflight showed ordinary losses could drag the low-route/base regime down.
- Final deployed loss mode is high-route-only branch distillation with `allow_proxy_only_loss=true`.

## Evaluation

Primary test eval:

- Path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/test_eval_best_val_gpu.json`
- PSNR: `17.889658453438816`
- Samples: `100`

Standard-loss re-eval, using the same checkpoint:

- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/configs/a_4kd37_eval_standard_loss.toml`
- Path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/test_eval_best_val_standard_loss_gpu.json`
- PSNR: `17.889658453438816`
- Loss: `0.11911950781941413`
- Auxiliary loss: `0.003231124826706946`
- Total loss: `0.12235063299536705`

Task metrics:

- Path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/test_task_metrics_best_val_gpu.json`
- SSIM: `0.7026848953962326`
- Edge magnitude L1: `0.1586235400289297`
- High frequency L1: `0.02159714322537184`
- Local contrast L1: `0.024814886646345256`
- Chroma L1: `0.02905867462977767`
- Dark channel L1: `0.14105536360293627`

## A18 Alignment

Path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/a18_record_alignment_best_val_gpu.json`

- Overall: `+0.042724077624394126 dB`
- Low route: `+0.1591364853806252 dB`
- High route: `-0.11159004428502844 dB`
- A15-oracle group: `+0.2493535773419097 dB`
- A08-oracle group: `-0.19984098726138502 dB`

This is a genuine aggregate breakthrough over A18/A36. The remaining caveat is high route: it improved from A33/A35 around `-0.1267 dB` to `-0.1116 dB`, but it still misses the planned `>-0.10 dB` target by about `0.0116 dB`.

## Route-Regime Diagnostics

Path: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/metrics/input_std_route_regime_best_val_gpu.json`

- Overall selected delta abs mean: `0.0062688171397894624`
- High-route selected delta abs mean: `0.014578644511138284`
- Low-route selected delta abs mean: `0.0`
- High branch L1 vs A08 teacher: `0.014976846487369648`
- Low branch L1 vs A15 teacher: `0.012233833717018888`
- High-route count: `43`
- Low-route count: `57`

The low route stayed locked at zero selected delta, while high route kept a useful but bounded correction.

## Decision

A37 is promoted as the new best single-checkpoint fast-probe candidate and an aggregate milestone. It beats A24, A33, A35, A36, and the A18 router on the current 100-sample fast-test PSNR protocol while also clearing SSIM and comparable loss requirements.

Boundary: A37 does not completely solve the original high-input-std/A08-tail gap. It narrows the high-route gap materially but leaves `-0.11159004428502844 dB` against A18 router and `-0.19984098726138502 dB` on the A08-oracle group. The next run should use A37 as the anchor and target only this remaining high-route/A08-oracle tail without giving back the low-route surplus.
