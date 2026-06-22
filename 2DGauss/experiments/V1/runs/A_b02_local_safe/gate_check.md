# Gate Check: V1/T01/A_b02_local_safe

Status: prepared only; not launched.

## Purpose

- [x] Use the locally transferred `UHD_haze` patch cache under `/home/ldj/wxc/2DGauss`.
- [x] Avoid overwriting historical `A_b02` artifacts by using a separate run id.
- [x] Enforce the shared-server constraint: only physical GPU `0` or `3` may be used.
- [x] Prefer physical GPU `3` by default, because current GPU0 utilization is already saturated.
- [x] Use a conservative launch envelope so the run does not crowd other users.

## Dataset

- [x] Cache root: `/home/ldj/wxc/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511`.
- [x] Train pairs: `17520`.
- [x] Validation pairs: `100`.
- [x] Test pairs: `231`.
- [x] Split file counts and input/gt pairing were verified in this server before this run package was prepared.

## Runtime Envelope

- [x] Batch size: `4`.
- [x] DataLoader workers: `2`.
- [x] Max steps: `1200`.
- [x] Wall-clock cap: `58` minutes.
- [x] Validation/checkpoint interval: `300` steps.
- [x] Validation batches per training-time validation: `50`.
- [x] AMP: disabled.

## GPU Safety

- [x] Python CUDA entrypoints now default to `CUDA_VISIBLE_DEVICES=3` when unset.
- [x] Python CUDA entrypoints reject `CUDA_VISIBLE_DEVICES` values outside physical GPU ids `0,3`.
- [x] Python CUDA entrypoints default to single-GPU visibility; `CUDA_VISIBLE_DEVICES=0,3` is rejected unless an explicit diagnostic override is set.
- [x] `launch_tmux.sh` defaults to physical GPU `3`.
- [x] `launch_tmux.sh` refuses launch unless the target GPU has at least `24576 MiB` free.
- [x] `launch_tmux.sh` refuses launch unless the target GPU utilization is at most `60%`.
- [x] `launch_tmux.sh` supports `TWODGAUSS_DRY_RUN=1` to check resources without launching training.

## Remaining Before Launch

- [ ] Re-check `nvidia-smi` immediately before launch.
- [ ] Run formal preflight under `CUDA_VISIBLE_DEVICES=3`.
- [ ] Optionally run a short smoke calibration with the same physical GPU and conservative batch.
- [ ] Launch only if the GPU check and preflight are clean.
