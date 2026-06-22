# Gate Check: V1/T01/A_e03

Status: launched; startup window passed through post-validation continuation.

## Purpose

- [x] Address the `A_d03` deployment failure.
- [x] Preserve the same edge-guided detail recovery mechanism.
- [x] Keep the `>=80G` training-memory floor.
- [x] Add the template-side CUDA cache release after full-image validation.
- [x] Launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce
  fragmentation risk.

## Required Run Identity

- [x] Architecture version: `V1`.
- [x] Training template version: `T01`.
- [x] Training run version: `A_e03`.
- [x] Full run identifier: `V1/T01/A_e03`.
- [x] Run folder: `experiments/V1/runs/A_e03`.
- [x] Run folder is independent and does not overwrite `A_d03`.

## Dataset And Protocol

- [x] Dataset: `UHD_deblur`.
- [x] Training split: `train/input_new -> train/gt_new`, offset `100`.
- [x] Internal validation: first `20` train pairs, full-image `crop_size=0`.
- [x] Official test: `test/input300 -> test/gt300`, reserved for post-training.
- [x] Training crop: random `512x512`.
- [x] Batch size: `20`.
- [x] Max steps: `1500`.
- [x] Validation interval: `500`.
- [x] Checkpoint interval: `250`.

## Mechanism

- [x] `use_edge_guidance=true`.
- [x] `edge_guidance_blocks=2`.
- [x] `edge_guidance_strength=0.25`.
- [x] `laplacian_weight=0.02`.
- [x] Total params expected: `231528`.

## Gate Items

- [x] Static checks completed.
- [x] Formal preflight completed.
- [x] Launch script checked.
- [x] Explicit deployment command recorded.
- [x] Startup window confirmed.

## Gate Completion

Static checks:

```text
scripts/train_v1_t01.py: py_compile ok
scripts/smoke_calibrate_v1_t01.py: py_compile ok
scripts/evaluate_v1_t01.py: py_compile ok
launch_tmux.sh: bash -n ok
template full_id: V1/T01/A_e03
run-local full_id: V1/T01/A_e03
batch_size: 20
use_edge_guidance: true
laplacian_weight: 0.02
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
loss_total: 0.16460011899471283
loss_charbonnier: 0.16262266039848328
loss_gradient: 0.02879663184285164
loss_laplacian: 0.02688193880021572
val_batches: 20
val_loss: 0.08276902846992015
val_psnr: 19.91969840620233
```

## Launch And Startup

```text
session: 2dgauss_a_e03_uhddeblur_edge_guided_cache_safe
log: experiments/V1/runs/A_e03/logs/train_20260509_012535.log
launch_env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Startup-window observations:

```text
step 1 loss_total=0.1559442
step 20 loss_total=0.0464724
step 250 checkpoint written: step_000250.pt and latest.pt
step 500 full-image validation: val_loss=0.0236564, val_psnr=28.5723
step 500 checkpoint written: step_000500.pt and latest.pt
post-validation continuation confirmed through step 590
observed GPU memory: about 97.1GB used
```
