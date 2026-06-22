# A_dsh19 Manifest

- Full id: `V1/T01/A_dsh19_dense_bestval_lr5e5_finetune`
- Status: completed and independently evaluated; improves dense-view L1/SSIM but does not replace A_dsh18 as PSNR-best.
- Purpose: low-learning-rate finetune from the A_dsh18 best-val checkpoint to test whether the dense-run plateau can be pushed further.
- Dataset root: `/home/ldj/wxc/4K_8K/all_synth_deshake_dense_v1`
- Train/val pairs: `1992` / `488`
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh18_all_synth_dense_v1_gpu3_full_train/checkpoints/best_val.pt` at step `1800`
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/configs/a_dsh19_dense_bestval_lr5e5_finetune.toml`
- Run-local config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_dsh19_dense_bestval_lr5e5_finetune/configs/a_dsh19_dense_bestval_lr5e5_finetune.toml`
- Training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/deshake_ddp_0_3_pending/logs/train_deshake_gpu3_train_20260606_235245_310660.log`

## Training Result

- Completed `800` finetune steps with `train_exit_code=0`.
- Produced `161` train metric records and `8` full-validation records.
- Saved `best_val.pt`, `final.pt`, `latest.pt`, and step checkpoints through `step_000800.pt`.
- Best validation loss was at step `800`: `val_loss=0.08194366997474285`, `val_psnr=19.515883491802068`, using all `488` val pairs.

## Independent Eval

- Input baseline: `17.904839041221802` PSNR / `0.4943777255362785` SSIM / `0.0961479383223064` L1.
- A_dsh19 best-val prediction: `19.515883583625932` PSNR / `0.5873642833598081` SSIM / `0.0799816856599703` L1.
- Relative to A_dsh18 independent eval: `-0.02473224330330126 dB` PSNR, `-0.0002577807945122884` L1, `+0.002148979526684913` SSIM.

## Boundary

A_dsh19 is a useful low-LR plateau probe and is better for L1/SSIM. It should not be promoted as the dense-view PSNR baseline because A_dsh18 remains stronger on independent PSNR.
