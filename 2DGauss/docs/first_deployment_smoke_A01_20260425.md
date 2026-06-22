# First Deployment Smoke: V1/T01/A01

## Record Time

- Time: `2026-04-25 00:37:09 CST`
- Run: `V1/T01/A01`
- Dataset: `UHD-LL_down4`
- Goal: choose first-deployment batch size and step budget with GPU memory between `85G` and `93G`, and runtime around 1 hour.

## Smoke Method

The smoke calibration used real `UHD-LL_down4` training batches and ran forward, loss, backward, and optimizer steps with the current `V1` model.

Script:

```bash
python3 scripts/smoke_calibrate_v1_t01.py \
  --config experiments/V1/templates/T01/configs/a01_short_uhdll_down4.toml \
  --batch-size <B> \
  --crop-size 0 \
  --steps <N> \
  --num-workers 2
```

`crop-size 0` means full `UHD-LL_down4` images at `540x960`.

## Calibration Results

| Batch size | Crop | Steps | Peak reserved | Peak allocated | Avg step | Status |
|---:|---|---:|---:|---:|---:|---|
| 4 | full `540x960` | 2 | `12.834G` | `12.119G` | `0.453s` | ok |
| 16 | full `540x960` | 2 | `51.189G` | `48.332G` | `0.771s` | ok |
| 28 | full `540x960` | 2 | `89.453G` | `84.540G` | `1.090s` | ok |
| 29 | full `540x960` | 2 | `92.680G` | `87.570G` | `1.191s` | ok, close to upper bound |
| 28 | full `540x960` | 20 | `89.453G` | `84.540G` | `0.851s` | ok, stable |

## First Deployment Setting

Use `batch_size = 28`.

Rationale:

- It satisfies the target memory range with peak reserved memory around `89.453G`.
- It leaves safer headroom than batch `29`, which reaches about `92.680G`.
- The stable 20-step smoke estimates about `3877` steps in 55 minutes and `4229` steps in 60 minutes.

Initial calibrated setting:

```text
batch_size = 28
train_crop_size = 0
max_steps = 4000
wall_clock_minutes = 58
val_interval = 200
checkpoint_interval = 200
```

The wall-clock cap remains below 1 hour. If the estimated step speed holds, `4000` steps should finish around the 1-hour target; otherwise the `58` minute cap stops the run.

## Launch Adjustment

Before launch, the user requested the first deployment to run about 1 hour with duration no shorter than 1 hour if possible. The launch setting was adjusted to:

```text
batch_size = 28
train_crop_size = 0
max_steps = 4300
wall_clock_minutes = 62
val_interval = 200
checkpoint_interval = 200
```

This preserves the calibrated memory behavior while giving the run enough budget to reach approximately 1 hour.

## Formal Configured Preflight

After updating the A01 config, the formal training entrypoint preflight succeeded:

```text
batch_input_shape: (28, 3, 540, 960)
batch_target_shape: (28, 3, 540, 960)
loss_total: 0.44990119338035583
val_loss: 0.354649218916893
val_psnr: 8.261483954411192
```
