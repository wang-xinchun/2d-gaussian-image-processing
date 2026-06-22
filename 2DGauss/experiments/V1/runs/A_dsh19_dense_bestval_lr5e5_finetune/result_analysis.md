# A_dsh19 Result Analysis

更新时间: 2026-06-07 00:03:26 CST

## Summary

A_dsh19 tested the suspected A_dsh18 bottleneck: after dense training, the remaining issue looked like a high-learning-rate plateau with small metric oscillation. The run loaded A_dsh18 `best_val.pt` at step `1800`, reset AdamW, lowered the learning rate to `5e-5`, and finetuned for `800` steps with full `488`-pair validation every `100` steps.

The result is mixed but useful: full-val loss, L1, and SSIM improve, while PSNR drops slightly. This means A_dsh19 is evidence that low-LR finetuning can improve perceptual/absolute-error behavior, but it does not solve the PSNR bottleneck.

## Training

- Init checkpoint: A_dsh18 `best_val.pt`, checkpoint step `1800`
- Learning rate: `5e-5`
- Steps: `800`
- Train metric records: `161`
- Validation records: `8`
- Best validation loss: step `800`, `val_loss=0.08194366997474285`, `val_psnr=19.515883491802068`, `val_batches=488`
- Peak CUDA reserved memory: about `39.0546875 GiB`

## Independent Eval

Full dense eval on `488` validation pairs:

- Input baseline: `17.904839041221802` PSNR, `0.4943777255362785` SSIM, `0.0961479383223064` L1
- A_dsh18 best-val: `19.540615826929233` PSNR, `0.5852153038331231` SSIM, `0.08023946645448259` L1
- A_dsh19 best-val: `19.515883583625932` PSNR, `0.5873642833598081` SSIM, `0.0799816856599703` L1
- A_dsh19 minus A_dsh18: `-0.02473224330330126 dB` PSNR, `+0.002148979526684913` SSIM, `-0.0002577807945122884` L1

## Group Signal

- A_dsh19 improves SSIM on `372/488` samples and L1 on `269/488` samples.
- A_dsh19 improves PSNR on `229/488` samples, so the mean PSNR drop is broad enough that A_dsh18 should remain PSNR-best.
- Motion deltas show PSNR gains for `impact_shake` and `quick_rotation`, but a larger PSNR drop for `portrait_handheld`.

## Decision

A_dsh19 is accepted as a safe completed low-LR finetune and a useful loss/SSIM improvement. It is not promoted as the dense-view PSNR-best checkpoint. Use A_dsh18 for PSNR-centered comparison and A_dsh19 for L1/SSIM-centered follow-up analysis.
