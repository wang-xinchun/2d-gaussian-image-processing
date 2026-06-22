# Gate Check: V1/T01/A_ai03

## Branch Identity

- Run: `V1/T01/A_ai03`
- Config: `experiments/V1/runs/A_ai03/configs/a_ai03_short_uhddeblur_shared_detail_gate_probe.toml`
- Mechanism: shared-prior feature refinement with decoder-detail gate

## Static Gate

- [x] Python compile/import check passed
- [x] Config parser accepts `gaussian_feature_refinement_use_decoder_detail_gate`

## Smoke / Preflight Gate

- [x] Formal preflight passed
- [x] True `6x720` smoke calibration passed
- [x] Reserved VRAM inside `85.0-93.0G`

Formal preflight:

```text
device = cuda
batch_input_shape = (6, 3, 720, 720)
batch_target_shape = (6, 3, 720, 720)
total_params = 305208
val_samples = 20
val_loss = 0.14137270227074622
val_psnr = 14.743460159597817
```

True `6x720` smoke:

```text
status = ok
max_memory_reserved_gb = 90.95703125
max_memory_allocated_gb = 90.63743114471436
avg_total_step_seconds = 2.0054664835333824
estimated_steps_target_wall_clock = 1735
within_target_reserved_memory_window = true
```

## Launch Gate

- [x] No conflicting active tmux training session
- [x] GPU availability checked
- [x] Training launched
- [x] Startup metrics writing

Launch:

```text
session = 2dgauss_a_ai03_uhddeblur_shared_detail_gate_probe
log = experiments/V1/runs/A_ai03/logs/train_20260519_102725.log
startup_gpu_memory_used = 96010 MiB
startup_gpu_utilization = 99%
first_metric_step = 1
first_metric_loss_total = 0.22921311855316162
```

## Post-Hoc Gate

- [x] Training finished cleanly
- [x] `final.pt` written
- [x] Endpoint checkpoint relation checked
- [x] Same-protocol official full-image test completed

Post-hoc:

```text
final_step = 2500
elapsed_minutes = 39.654192618529
endpoint_relation = step_002500.pt == latest.pt == final.pt by model weights
official_test_samples = 300
official_test_crop_size = 0
official_test_device = cuda
official_test_loss_mean = 0.03668969906090448
official_test_psnr_mean = 25.267052413067724
```

## Initial Decision

Gate passed and the short run was launched. The low cold-start preflight/step-1 signal should be watched, but early training loss recovered quickly by step 30, so the branch remains worth carrying to the first validation unless it becomes unstable.

## Final Decision

Run completed cleanly, but this exact decoder-detail gate is not a promotion candidate. It is slightly above `A_ae03`, but below `A_ah03`, `A_z03`, and frozen `M04/A_y03`.
