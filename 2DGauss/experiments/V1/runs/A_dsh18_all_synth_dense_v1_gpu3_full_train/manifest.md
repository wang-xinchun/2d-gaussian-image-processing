# A_dsh18 Manifest

- Full id: `V1/T01/A_dsh18_all_synth_dense_v1_gpu3_full_train`
- Status: completed and independently evaluated; current dense-view PSNR-best checkpoint.
- Purpose: train the base A_dsh15 model line on the duration-aware dense synthetic deshake view.
- Dataset root: `/home/ldj/wxc/4K_8K/all_synth_deshake_dense_v1`
- Train/val pairs: `1992` / `488`
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh18_all_synth_dense_v1_gpu3_full_train.toml`
- Run-local config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh18_all_synth_dense_v1_gpu3_full_train/configs/a_dsh18_all_synth_dense_v1_gpu3_full_train.toml`
- Training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_231350_33378.log`

## Training Result

- Completed `2000` steps with `train_exit_code=0`.
- Produced `401` train metric records and `20` training-time validation records.
- Saved `best_val.pt`, `final.pt`, `latest.pt`, and step checkpoints through `step_002000.pt`.
- Training-time best validation loss was at step `1800`: `val_loss=0.05253878780058585`, `val_psnr=22.82667685760712`, using `64` val batches.
- Final step `2000`: `val_loss=0.05280464925453998`, `val_psnr=22.771658398502467`, using `64` val batches.

## Independent Eval

- Independent full dense validation used all `488` val pairs.
- Input baseline: `17.904839041221802` PSNR / `0.4943777255362785` SSIM / `0.0961479383223064` L1.
- A_dsh18 best-val prediction: `19.540615826929233` PSNR / `0.5852153038331231` SSIM / `0.08023946645448259` L1.
- Mean gain over input: `+1.6357767857074346 dB` PSNR and `+0.09083757829684459` SSIM.

## Boundary

A_dsh18 is the current dense-view PSNR-best paired-frame checkpoint. Its training-time `64`-batch validation PSNR is optimistic relative to the independent full `488`-pair validation and should not be cited as the final dense metric.
