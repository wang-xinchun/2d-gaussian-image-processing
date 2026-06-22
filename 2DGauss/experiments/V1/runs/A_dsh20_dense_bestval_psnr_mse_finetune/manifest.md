# A_dsh20 Manifest

- Full id: `V1/T01/A_dsh20_dense_bestval_psnr_mse_finetune`
- Status: completed and independently evaluated; current dense-view PSNR-best checkpoint.
- Purpose: test whether the A_dsh19 bottleneck was caused by optimizing/selecting for Charbonnier/L1-style loss while the acceptance metric is PSNR.
- Dataset root: `/home/ldj/wxc/4K_8K/all_synth_deshake_dense_v1`
- Train/val pairs: `1992` / `488`
- Pending config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh20_dense_bestval_psnr_mse_finetune.toml`
- Run-local config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh20_dense_bestval_psnr_mse_finetune/configs/a_dsh20_dense_bestval_psnr_mse_finetune.toml`
- Training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260607_005058_778845.log`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh18_all_synth_dense_v1_gpu3_full_train/checkpoints/best_val.pt`

## Change From A_dsh19

- Added `loss.mse_weight = 4.0`.
- Reduced `loss.charbonnier_weight` to `0.25`.
- Kept `loss.gradient_weight = 0.01`.
- Set learning rate to `3e-5`.
- Set `training.best_selection_metric = "val_psnr"`.
- Used full `488`-pair validation every `100` steps.

## Training Result

- Completed `700` steps with `train_exit_code=0`.
- Produced `141` train metric records and `7` full validation records.
- Saved `best_val.pt`, `final.pt`, `latest.pt`, and step checkpoints through `step_000700.pt`.
- Best selection metric was `val_psnr`.
- Best and final checkpoint are both step `700`: `val_loss=0.09079718149599207`, `val_psnr=19.740802516976924`, `val_batches=488`.
- Peak CUDA reserved memory was about `39.0546875 GiB`.

## Independent Eval

- Independent full dense validation used all `488` val pairs.
- Input baseline: `17.904839041221802` PSNR / `0.4943777255362785` SSIM / `0.0961479383223064` L1.
- A_dsh20 best-val prediction: `19.740802621108518` PSNR / `0.5856158071670865` SSIM / `0.08071073212042512` L1.
- Mean gain over input: `+1.8359635798867182 dB` PSNR and `+0.09123808163080792` SSIM.

## Boundary

A_dsh20 is the current dense-view PSNR-best paired-frame checkpoint. It improves PSNR over A_dsh18 and A_dsh19, but it does not dominate every metric: L1 is worse than both A_dsh18 and A_dsh19, and SSIM is slightly above A_dsh18 but below A_dsh19. It remains a single-frame paired restoration baseline, not a complete long-video stabilization system.
