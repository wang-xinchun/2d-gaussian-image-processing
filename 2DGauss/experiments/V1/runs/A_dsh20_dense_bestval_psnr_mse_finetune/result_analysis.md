# A_dsh20 Result Analysis

更新时间: 2026-06-07 01:01:01 CST

## Summary

A_dsh20 tested the main bottleneck exposed by A_dsh19: low-learning-rate finetuning improved L1/SSIM but lowered PSNR, suggesting that the objective and best-checkpoint selection were not aligned with the acceptance metric. This run loaded A_dsh18 `best_val.pt`, added an MSE-heavy term, lowered the Charbonnier weight, and selected the best checkpoint by full-validation `val_psnr`.

The result is a real PSNR breakthrough on the same full `488`-pair independent evaluation: A_dsh20 reaches `19.740802621108518` PSNR, beating A_dsh18 by `+0.2001867941792852 dB` and A_dsh19 by `+0.22491903748258573 dB`.

## Training

- Init checkpoint: A_dsh18 `best_val.pt`, checkpoint step `1800`
- Loss: `charbonnier_weight=0.25`, `mse_weight=4.0`, `gradient_weight=0.01`
- Selection metric: `val_psnr`
- Learning rate: `3e-5`
- Steps: `700`
- Train metric records: `141`
- Validation records: `7`
- Best/full final validation: step `700`, `val_loss=0.09079718149599207`, `val_psnr=19.740802516976924`, `val_batches=488`
- Peak CUDA reserved memory: about `39.0546875 GiB`

## Independent Eval

Full dense eval on `488` validation pairs:

- Input baseline: `17.904839041221802` PSNR, `0.4943777255362785` SSIM, `0.0961479383223064` L1
- A_dsh18 best-val: `19.540615826929233` PSNR, `0.5852153038331231` SSIM, `0.08023946645448259` L1
- A_dsh19 best-val: `19.515883583625932` PSNR, `0.5873642833598081` SSIM, `0.0799816856599703` L1
- A_dsh20 best-val: `19.740802621108518` PSNR, `0.5856158071670865` SSIM, `0.08071073212042512` L1
- A_dsh20 minus A_dsh18: `+0.2001867941792852 dB` PSNR, `+0.0004005036308080053` SSIM, `+0.0004712656660809151` L1
- A_dsh20 minus A_dsh19: `+0.22491903748258573 dB` PSNR, `-0.001748476192721578` SSIM, `+0.0007290464604548255` L1

## Group Signal

- A_dsh20 improves PSNR on `399/488` samples versus A_dsh18 and `397/488` samples versus A_dsh19.
- A_dsh20 improves SSIM on `301/488` samples versus A_dsh18 and `283/488` samples versus A_dsh19.
- A_dsh20 improves L1 on `230/488` samples versus A_dsh18 and `220/488` samples versus A_dsh19.
- Versus A_dsh18, PSNR improves most on `walking_diagonal_smooth_detail` (`+0.30027165905258774 dB`), then `bicycle_smooth` (`+0.1593187464022311 dB`), `impact_shake` (`+0.13626153685425968 dB`), and `quick_rotation` (`+0.08879183502689125 dB`).
- Versus A_dsh18, `portrait_handheld` regresses by `-0.12188217023502028 dB` PSNR and `-0.008783521751562754` SSIM.

## Decision

A_dsh20 is promoted to the current dense-view PSNR-best checkpoint. The bottleneck was not simply lack of finetuning; it was a metric-alignment issue. A PSNR-aligned MSE term plus `val_psnr` checkpoint selection breaks the PSNR plateau, while making the L1/SSIM tradeoff explicit.

Next experiments should try to keep the PSNR gain while recovering L1/SSIM, for example by sweeping a lower MSE weight or adding a secondary selection/reporting checkpoint for SSIM/L1. A_dsh20 should not be described as a complete video stabilization solution.
