# A_b02_gpu0_bs24 Gate Check

- [x] Local cache exists under `/home/ldj/wxc/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511`.
- [x] Project venv exists and has CUDA PyTorch installed.
- [x] GPU0 `batch_size=24` smoke passed with peak reserved memory about `37.619GB`.
- [x] Actual launch may share GPU0 when free memory is at least `42000MiB`.

Current boundary:
- GPU0 was no longer empty after the smoke; it was occupied by another user's `python -u trainval.py` jobs with high utilization.
- Do not launch if GPU0 free memory drops below the safety threshold; utilization is recorded but not used as a hard block for this trial.
