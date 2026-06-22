# A_dsh18 Result Analysis

更新时间: 2026-06-07 00:03:26 CST

## Summary

A_dsh18 completed the first full GPU3-only training on the duration-aware dense synthetic deshake view. Relative to the old 112-pair BVI-AOM training view, this run moves the main bottleneck from data scarcity to metric tradeoff and validation protocol: dense training is clearly useful, but the training-time `64`-batch validation was optimistic compared with independent full `488`-pair validation.

## Training

- Steps: `2000`
- Train metric records: `401`
- Validation records: `20`
- Best training-time validation: step `1800`, `val_loss=0.05253878780058585`, `val_psnr=22.82667685760712`, `val_batches=64`
- Final training-time validation: step `2000`, `val_loss=0.05280464925453998`, `val_psnr=22.771658398502467`, `val_batches=64`
- Peak CUDA reserved memory: about `39.0546875 GiB`

## Independent Eval

Full dense eval on `488` validation pairs:

- Input baseline: `17.904839041221802` PSNR, `0.4943777255362785` SSIM, `0.0961479383223064` L1
- Prediction: `19.540615826929233` PSNR, `0.5852153038331231` SSIM, `0.08023946645448259` L1
- Gain: `+1.6357767857074346 dB` PSNR, `+0.09083757829684459` SSIM

## Decision

A_dsh18 is accepted as the current dense-view PSNR-best paired-frame checkpoint. It should be cited with the independent `488`-pair eval metric (`19.5406` PSNR), not the training-time `64`-batch validation PSNR (`22.8267`).
