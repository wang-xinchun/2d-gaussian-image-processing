# A_b02_ddp_0_3_bs12 Gate Check

- [x] User authorized distributed training on physical GPU0 and GPU3.
- [x] Data cache exists under `/home/ldj/wxc/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511`.
- [x] Project venv has CUDA PyTorch.
- [x] Training script has DDP support with rank0-only metrics/checkpoints/validation.
- [x] Launcher checks free memory on GPU0 and GPU3 before launch.
- [x] Launcher now refuses formal launch by default unless `TWODGAUSS_ALLOW_UHDHAZE_CALIBRATION=1` is set, because this is not the current deshaking task.
- [ ] DDP smoke package must pass before formal calibration launch.

Current launch rule:
- Launch only if both GPU0 and GPU3 have at least `30000MiB` free memory.
- High utilization is recorded but not a hard blocker for this package, because the user's updated objective is to share under-filled GPUs while avoiding OOM.
- This package remains `UHD_haze` calibration only. It is not approved for corrected UHD video deshaking training.
