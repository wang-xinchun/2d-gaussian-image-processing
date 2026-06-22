# V1/T01/A_4kd28 Result Analysis

## Summary

`A_4kd28` tested a route-conditioned `decoder_feature` expert after the A27 terminal-head failure. It completed GPU3 preflight, 400-step formal training, standard fast-test evaluation, task-specific diagnostics, A18-record alignment, and input-std route-regime analysis.

The engineering path works, but the scientific result is negative. A28 does not beat A24 and does not reduce the high-`input_std` / A08-tail gap.

## Training

- Formal train log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/logs/train_20260614_154636.log`
- Exit code: `0`
- Best checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/checkpoints/best_val.pt`
- Best-val step: `400`
- Best-val PSNR: `18.44844816281949`
- Best-val loss: `0.11907587297260762`

Validation improved monotonically, which made the run look promising on val. The independent fast-test result did not follow.

## Fast-Test Metrics

- Standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/test_eval_best_val_gpu.json`
- Task metrics: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/test_task_metrics_best_val_gpu.json`
- PSNR: `17.73849582669033`
- SSIM: `0.6989976793527604`
- Loss: `0.12039212018251419`
- Total loss: `0.12362300843000412`
- Edge magnitude L1: `0.1572890555113554`
- High-frequency L1: `0.021602761903777717`
- Local-contrast L1: `0.024325284799560906`
- Chroma L1: `0.030040878672152757`
- Dark-channel L1: `0.13908436059951781`

A28 fails the required A24 fast-test PSNR threshold `17.790202882742463`, fails the hoped-for `17.82`, and stays below the A18 router target `17.84693437641316`. SSIM also narrowly misses the `0.699` proxy floor.

## A18 Alignment

- Alignment file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/a18_record_alignment_best_val_gpu.json`
- Overall gap vs A18 router: `-0.1084385491240903 dB`
- Low-route gap vs A18 router: `+0.06555415143863476 dB`
- High-route gap vs A18 router: `-0.3390800359165398 dB`
- A15-oracle gap vs A18 router: `+0.24120325508008136 dB`
- A08-oracle gap vs A18 router: `-0.5188876236246397 dB`

The low-route side remains healthy, but the high-route side is worse than A27 and worse than the A24 diagnostic target. This confirms that feature-level route deltas did not restore A08-like high-tail preservation.

## Route-Regime Diagnostics

- Route-regime file: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd28/metrics/input_std_route_regime_best_val_gpu.json`
- Overall selected delta abs mean: `0.0011144706892082467`
- Low-route selected delta abs mean: `0.0013225519980647061`
- High-route selected delta abs mean: `0.0008386419774682889`
- High branch L1 vs A08 teacher: `0.015561815286271794`
- Low branch L1 vs A15 teacher: `0.01525487631494016`

The gate follows the fixed prior, but the actual selected route delta remains only about `1e-3`. A28 therefore behaves like another small correction on top of the shared path, not like a true high/low expert split.

## Decision

A28 is not promoted. It is a useful negative mechanism diagnostic:

- It proves that merely moving a small route delta from terminal RGB space into `decoder_feature` is still too weak.
- It strengthens the explanation that the missing capacity is not just feature placement, but effective independent route prediction capacity.
- The next run should stop using small deltas as the main route expression and instead give high/low routes hard independent final RGB decoder heads initialized from the A24 base head.

Proposed next run: `A_4kd29`, `input_std_route_decoder_heads`, where high/low route heads directly own the final RGB projection while keeping the fixed input-std prior, natural sampling, and A08/A15 branch supervision.
