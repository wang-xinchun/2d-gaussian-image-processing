# A_dsh18 Gate Check

- [x] Uses the current recommended dense dataset root `/home/ldj/wxc/4K_8K/all_synth_deshake_dense_v1`.
- [x] Data gate verified `1992` train pairs and `488` val pairs.
- [x] Uses physical GPU3 only through `CUDA_VISIBLE_DEVICES=3`.
- [x] GPU0/1/2 were not used by this project.
- [x] Uses `batch_size=24`, global batch `24`, `num_workers=0`, `512` crop, and `TWODGAUSS_CUDA_MEMORY_FRACTION=0.85`.
- [x] Real training completed with `train_exit_code=0`.
- [x] Checkpoints and train/validation metrics were produced.
- [x] Independent full dense validation completed on `488` val pairs.
- [x] Post-run cleanup found no A_dsh18 training/tmux residue and GPU3 returned to idle.

Current rule: A_dsh18 is the dense-view PSNR-best checkpoint. A_dsh19 improves L1/SSIM but does not replace A_dsh18 as PSNR-best.
