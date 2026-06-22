# A_b02_gpu0_bs22_r2 Gate Check

- [x] Local cache exists under `/home/ldj/wxc/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511`.
- [x] Project venv exists and has CUDA PyTorch installed.
- [x] GPU0 `batch_size=24` passed 3-step smoke and a longer partial training run through step 860.
- [x] This rerun reduces per-step batch from `24` to `22` to increase GPU0 memory headroom.
- [ ] Actual launch must pass the fresh GPU0 free-memory gate.

Current boundary:
- Use GPU0 only.
- Do not touch GPU1/2/3.
- Do not launch if GPU0 free memory drops below `42000MiB`.
- Utilization is recorded but not used as a hard block for this trial.
