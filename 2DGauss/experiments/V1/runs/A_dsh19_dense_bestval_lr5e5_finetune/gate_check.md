# A_dsh19 Gate Check

- [x] Added `training.init_checkpoint` support to `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`.
- [x] `py_compile` passed for `train_v1_t01.py`, `evaluate_deshake_checkpoint.py`, and `preflight_deshake_ddp_gpu0_3.py`.
- [x] Checkpoint compatibility check loaded A_dsh18 `best_val.pt` strictly with no missing or unexpected keys.
- [x] GPU3 dry-run preflight passed with `1992` train pairs and `488` val pairs.
- [x] Uses physical GPU3 only through `CUDA_VISIBLE_DEVICES=3`.
- [x] GPU0/1/2 were not used by this project.
- [x] Real training completed with `train_exit_code=0`.
- [x] Independent full dense validation completed on `488` val pairs.
- [x] Post-run cleanup found no A_dsh19 training/tmux/eval residue and GPU3 returned to idle.

Current rule: A_dsh19 is a completed low-LR finetune that improves L1/SSIM, while A_dsh18 remains dense-view PSNR-best.
