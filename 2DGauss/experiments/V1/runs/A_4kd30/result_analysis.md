# A_4kd30 Result Analysis

## Summary

A30 tested whether A29 hard high/low `input_std` route decoder heads could be made useful by adding a small route-delta trust region and staged branch pressure. It is engineering-complete, but it is not promoted.

The best checkpoint stayed at step 50, before branch supervision had any weight. Final validation did not recover after branch ramp-up.

## Configuration

- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/checkpoints/best_val.pt`
- Teacher A/high route: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher B/low route: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
- Main mechanism: `use_input_std_route_decoder_heads=true`
- Disabled mechanisms: old input-std residual, A27 terminal heads, A28 feature expert
- Fixed route prior: high if `input_std > 0.07026463374495506`
- Trust region: `input_std_route_delta_trust_weight=0.08`, budget `0.0015`
- Branch schedule: high/A08 `0.10`, low/A15 `0.03`, warmup `50`, ramp `150`
- Gate schedule: learned-gate weight `0.02`, ramp `50`
- Sampling: natural fast-probe sampling, no balanced sampler

## Checks

- `py_compile scripts/train_v1_t01.py`: passed after A30 training-loss changes and after route-info write-condition fix.
- TOML parse and checkpoint path checks: passed.
- Launcher `bash -n`: passed.
- GPU0/GPU3 dry-run gate: passed where memory allowed; formal training used physical GPU3.
- Preflight: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd30/logs/train_20260614_173122.log`, exit code `0`.
- Formal training: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd30/logs/train_20260614_173319.log`, exit code `0`.

## Validation

Best selection used `val_psnr`.

| step | val_psnr | val_loss | high_delta | low_delta |
| --- | ---: | ---: | ---: | ---: |
| 50 | 18.231969745567717 | 0.122691053673625 | 0.00007680406204599422 | 0.000059086885739816354 |
| 100 | 18.153006463219054 | 0.12401501465588809 | 0.00032506815259694124 | 0.00024370694925892166 |
| 150 | 18.164903592335143 | 0.12381048835813999 | 0.0008037611335748807 | 0.00043909458676353095 |
| 200 | 18.17848947550911 | 0.12355151783674956 | 0.0017518808599561453 | 0.0006247954122954979 |
| 250 | 18.160505904371178 | 0.12382980771362781 | 0.0026073359476868063 | 0.0008245093218283727 |

Interpretation: the trust region successfully made the selected best checkpoint conservative, but that conservative checkpoint is still below A24/A29. When route delta grows later, validation remains worse than step 50.

## Fast-Test Results

Standard fast-test best-val:

- PSNR: `17.75180602939797`
- Loss: `0.12060016021132469`
- Total loss: `0.1238312616199255`
- Checkpoint step: `50`

Task metrics best-val:

- SSIM: `0.7003354308009148`
- Dark-channel L1: `0.14298989839851856`
- Edge-magnitude L1: `0.15856851033866406`
- High-frequency L1: `0.02160121445544064`
- Local-contrast L1: `0.02471954238601029`
- Luminance L1: `0.10521231602877379`
- Chroma L1: `0.030323888827115297`
- Saturation L1: `0.05569658525288105`

## A18 Alignment

Against A18 two-checkpoint router records:

- Overall gap vs A18 router: `-0.0951283464164498 dB`
- Low-route gap vs A18 router: `+0.012763517901581372 dB`
- High-route gap vs A18 router: `-0.23814779446593295 dB`
- A08-oracle group gap vs router: `-0.27065612700046393 dB`
- A15-oracle group gap vs router: `+0.05439531852548818 dB`

Compared with A29:

- A29 PSNR was `17.76127769536133`; A30 is `17.75180602939797`.
- A29 high-route gap was `-0.236088142902285 dB`; A30 is `-0.23814779446593295 dB`.
- A29 low-route gap was `+0.027826703499809195 dB`; A30 is `+0.012763517901581372 dB`.

## Route-Regime Diagnostics

Best-val route-regime metrics:

- Overall selected delta abs mean: `0.0000578112014773069`
- High-route selected delta abs mean: `0.0000618680088908065`
- Low-route selected delta abs mean: `0.0000547508029022107`
- High branch L1 vs A08 teacher: `0.29880278470904326`
- Low branch L1 vs A15 teacher: `0.3636353517833509`

The best checkpoint is almost an identity-initialized route-head checkpoint. It does not use route decoder capacity enough to recover A08-like high-route behavior.

## Acceptance

- Beat A24 PSNR `17.790202882742463`: no.
- Reach hopeful target `17.82`: no.
- Approach A18 router `17.84693437641316`: no.
- Improve high-route gap to `>-0.10 dB`: no, A30 is `-0.23814779446593295 dB`.
- Keep low-route gap `>= +0.03 dB`: no, A30 is `+0.012763517901581372 dB`.
- Keep SSIM `>= 0.699`: yes, `0.7003354308009148`.
- Keep loss `<= 0.1205`: no, `0.12060016021132469`.

## Conclusion

A30 is a negative but useful mechanism diagnostic. It shows that route-delta trust-region and staged branch pressure are insufficient when applied only around the hard route decoder-head terminal path. The trust region prevents the most obvious harmful drift, but the best checkpoint retreats to tiny deltas and does not solve the high-`input_std` A08-tail gap.

Next mainline should stop treating this as a terminal-head stabilization problem. A better A31 direction is a more structural mechanism: freeze the shared A24 path and train only route heads in a high-route-focused stage, transplant or initialize the high route closer to A08 behavior, or move route specialization into an upstream Gaussian/control/decoder feature branch with explicit high-route reconstruction against ground truth and A08 teacher.
