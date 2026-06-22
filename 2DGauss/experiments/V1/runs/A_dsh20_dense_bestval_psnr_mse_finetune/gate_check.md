# A_dsh20 Gate Check

- [x] Added optional `loss.mse_weight` support to `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`.
- [x] Added `training.best_selection_metric` support and selected `best_val.pt` by `val_psnr`.
- [x] `py_compile` passed for `train_v1_t01.py`, `evaluate_deshake_checkpoint.py`, and `preflight_deshake_ddp_gpu0_3.py`.
- [x] CPU logic check passed for MSE loss contribution and metric-direction helpers.
- [x] Checkpoint compatibility check loaded A_dsh18 `best_val.pt` strictly with no missing or unexpected keys.
- [x] GPU3 dry-run preflight passed with `1992` train pairs and `488` val pairs.
- [x] Uses physical GPU3 only through `CUDA_VISIBLE_DEVICES=3`.
- [x] GPU0/1/2 were not used by this project.
- [x] Real training completed with `train_exit_code=0`.
- [x] Independent full dense validation completed on `488` val pairs.
- [x] Post-run cleanup found no A_dsh20 training/tmux/eval residue and GPU3 returned to idle.

Current rule: A_dsh20 is the dense-view PSNR-best checkpoint, while A_dsh19 remains the L1/SSIM-improving finetune reference.
