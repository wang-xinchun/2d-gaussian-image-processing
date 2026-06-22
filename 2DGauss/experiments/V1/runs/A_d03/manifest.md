# Run Manifest: V1/T01/A_d03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_d03`
- Full identifier: `V1/T01/A_d03`

## Current Status

Launched and aborted after step `500` due post-validation CUDA OOM. Static
checks, smoke calibration, and formal preflight had passed before launch.

## Purpose

`A_d03` is the first `UHD_deblur` branch after `A_c03` that adds a new strong
mechanism instead of only increasing batch size.

`A_c03` proved that the `>=80G` memory floor can be satisfied, but it did not
beat the current official baseline:

```text
A03 final.pt official full-image test PSNR = 25.1402
A_c03 final.pt official full-image test PSNR = 25.1250
```

Therefore `A_d03` introduces edge-guided detail recovery to target deblurring
edges and high-frequency structure directly.

## New Mechanism

The code adds a default-off high-resolution edge-guided refiner:

- input-side Sobel/Laplacian/highpass descriptors are computed at full
  resolution;
- those descriptors are projected into the decoder feature space;
- the branch fuses edge features, decoder features, and Gaussian attention;
- the branch predicts an additional bounded RGB residual with strength `0.25`;
- `laplacian_weight=0.02` adds a small high-frequency consistency training
  signal.

The old `A03/A_c03` path remains the default when `use_edge_guidance=false`.

## Protocol

Kept from `A_c03`:

- dataset: `UHD_deblur`;
- training crop: random `512x512`;
- internal validation: full-image `crop_size=0`;
- validation limit: first `20` train pairs;
- validation interval: `500`;
- checkpoint interval: `250`;
- official test reserved for post-training evaluation;
- high-VRAM requirement: training reserved memory must be at least `80G`.

Changed from `A_c03`:

- `use_edge_guidance=true`;
- `edge_guidance_blocks=2`;
- `edge_guidance_strength=0.25`;
- `laplacian_weight=0.02`;
- initial batch target `24` was rejected because the edge refiner increased
  activation memory enough to OOM;
- final batch target is `20`, which passes smoke and still satisfies the
  `>=80G` memory floor.

## VRAM Calibration

Rejected:

```text
batch_size=24
status=oom
max_memory_reserved_gb=94.2910

batch_size=23
status=oom
max_memory_reserved_gb=93.7734

batch_size=22
status=oom
max_memory_reserved_gb=92.9238

batch_size=21
status=oom
max_memory_reserved_gb=93.8066
```

Accepted:

```text
batch_size=20
status=ok
steps=20
max_memory_allocated_gb=90.9545
max_memory_reserved_gb=94.1211
avg_total_step_seconds=1.3975
estimated_steps_60min=2575
```

## Files

- Template config: `experiments/V1/templates/T01/configs/a_d03_short_uhddeblur_edge_guided_vram80.toml`
- Run-local config: `experiments/V1/runs/A_d03/configs/a_d03_short_uhddeblur_edge_guided_vram80.toml`
- Gate check: `experiments/V1/runs/A_d03/gate_check.md`
- Manifest: `experiments/V1/runs/A_d03/manifest.md`

## Required Before Launch

1. Launch with `experiments/V1/runs/A_d03/launch_tmux.sh`.
2. Confirm startup window through at least the first checkpoint and preferably
   the first full-image validation.

## Launch Authorization

```text
authorized_at: 2026-05-09 00:00 UTC
user_request: 请你针对问题，设计并部署实验。注意门槛和模板
```

## Deployment Outcome

```text
session: 2dgauss_a_d03_uhddeblur_edge_guided_vram80
log: experiments/V1/runs/A_d03/logs/train_20260509_011214.log
step_000250.pt: written
step_000500.pt: written
step 500 full-image validation: val_loss=0.0235864, val_psnr=28.5929
failure: CUDA OOM on the next backward pass after validation
diagnosis: post-validation CUDA reserved/cache fragmentation
successor: V1/T01/A_e03
```

## Gate Completion

Static checks:

```text
scripts/train_v1_t01.py: py_compile ok
scripts/smoke_calibrate_v1_t01.py: py_compile ok
scripts/evaluate_v1_t01.py: py_compile ok
template config full_id: V1/T01/A_d03
run-local config full_id: V1/T01/A_d03
use_edge_guidance: true
batch_size: 20
laplacian_weight: 0.02
launch_tmux.sh: bash -n ok
legacy A_c03 checkpoint compatibility: strict load ok
```

Formal preflight:

```text
preflight: ok
device: cuda
train_samples: 20
val_samples: 20
batch_input_shape: (20, 3, 512, 512)
batch_target_shape: (20, 3, 512, 512)
total_params: 231528
loss_total: 0.09860626608133316
loss_charbonnier: 0.09657278656959534
loss_gradient: 0.029611244797706604
loss_laplacian: 0.027645735070109367
val_batches: 20
val_loss: 0.048894323967397214
val_psnr: 23.718956728936703
```
