# Run Manifest: V1/T01/A_b02_local_safe

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_b02_local_safe`
- Full identifier: `V1/T01/A_b02_local_safe`

## Intent

This is a local-server, conservative `UHD_haze` patch-cache training package.
It is deliberately separate from historical `A_b02` so old completed metrics,
checkpoints, and acceptance notes remain untouched.

The current server is shared. The launch policy is therefore conservative:
use only physical GPU `0` or `3`, default to physical GPU `3`, and refuse to
launch when the target GPU is already busy or short on free memory.

## Dataset

- Cache root: `/home/ldj/wxc/2DGauss/datasets/cache/UHD_haze_patch512_p8_png_seed20260511`
- Train input: `datasets/cache/UHD_haze_patch512_p8_png_seed20260511/train/input`
- Train target: `datasets/cache/UHD_haze_patch512_p8_png_seed20260511/train/gt`
- Validation input: `datasets/cache/UHD_haze_patch512_p8_png_seed20260511/val/input`
- Validation target: `datasets/cache/UHD_haze_patch512_p8_png_seed20260511/val/gt`
- Test input: `datasets/cache/UHD_haze_patch512_p8_png_seed20260511/test/input`
- Test target: `datasets/cache/UHD_haze_patch512_p8_png_seed20260511/test/gt`
- Counts verified on this server: train `17520`, val `100`, test `231` paired samples.

## Conservative Runtime

- Batch size: `4`
- DataLoader workers: `2`
- Max steps: `1200`
- Wall-clock cap: `58` minutes
- Validation interval: `300`
- Checkpoint interval: `300`
- Max validation batches during training: `50`

## Launch

Prepared launch script:

```bash
bash experiments/V1/runs/A_b02_local_safe/launch_tmux.sh
```

Default behavior:

- Uses physical GPU `3`.
- Requires at least `24576 MiB` free on the target GPU.
- Requires target GPU utilization no higher than `60%`.
- Exposes only the selected physical GPU through `CUDA_VISIBLE_DEVICES`.
- Supports `TWODGAUSS_DRY_RUN=1` for resource-check-only execution.

To explicitly choose physical GPU `0`, the caller must set:

```bash
TWODGAUSS_PHYSICAL_GPU=0 bash experiments/V1/runs/A_b02_local_safe/launch_tmux.sh
```

This should only be done if GPU0 is genuinely available immediately before launch.
