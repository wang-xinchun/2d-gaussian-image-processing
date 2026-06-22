# Run Manifest: V1/T01/A_e03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_e03`
- Full identifier: `V1/T01/A_e03`

## Status

Launched and startup-window accepted. This run is the cache-safe redeployment
of the `A_d03` edge-guided mechanism after `A_d03` OOMed immediately after the
first full-image validation.

## Gate Completion

Static checks and formal preflight passed. Preflight summary:

```text
device: cuda
train_samples: 20
val_samples: 20
total_params: 231528
loss_total: 0.16460011899471283
loss_laplacian: 0.02688193880021572
val_loss: 0.08276902846992015
val_psnr: 19.91969840620233
```

## Rationale

`A_d03` reached step `500`, wrote `step_000500.pt`, and completed full-image
validation with PSNR `28.5929`, but failed on the next backward pass with CUDA
OOM. The failure log showed a large reserved-but-unallocated CUDA block, so the
fix is runtime hygiene rather than a mechanism change.

`A_e03` keeps the same model and loss settings, but uses the updated T01
template, which releases CUDA cache after validation, and launches with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Files

- Template config: `experiments/V1/templates/T01/configs/a_e03_short_uhddeblur_edge_guided_cache_safe.toml`
- Run-local config: `experiments/V1/runs/A_e03/configs/a_e03_short_uhddeblur_edge_guided_cache_safe.toml`
- Gate check: `experiments/V1/runs/A_e03/gate_check.md`
- Manifest: `experiments/V1/runs/A_e03/manifest.md`
- Training log: `experiments/V1/runs/A_e03/logs/train_20260509_012535.log`

## Launch Status

```text
session: 2dgauss_a_e03_uhddeblur_edge_guided_cache_safe
config: experiments/V1/templates/T01/configs/a_e03_short_uhddeblur_edge_guided_cache_safe.toml
log: experiments/V1/runs/A_e03/logs/train_20260509_012535.log
launch_env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Startup-window observations:

```text
step 1 loss_total=0.1559442
step 20 loss_total=0.0464724
step 250 checkpoint written
step 500 full-image validation val_loss=0.0236564 val_psnr=28.5723
step 500 checkpoint written
post-validation continuation confirmed through step 590
observed GPU memory about 97.1GB used
```

## Final Runtime Status

`A_e03` finished by reaching `max_steps=1500`.

```text
final_step: 1500
final_elapsed_minutes: 27.624958618481955
final_train_loss_total: 0.03862513601779938
final_validation_step: 1500
final_val_batches: 20
final_val_loss: 0.02255534678697586
final_val_psnr: 28.761768145576514
tmux_session_after_finish: closed
training_process_after_finish: none
gpu_memory_after_finish: 0 MiB used
```

## Acceptance Evaluation Summary

Training/runtime acceptance passed. `A_e03` preserved the `>=80G` deployment
gate, crossed the `A_d03` post-validation failure point, and finished without
OOM, NaN, checkpoint write failure, or data-loader failure.

Post-hoc full-image evaluation:

```text
step_001500.pt full-val: loss_mean=0.02255534678697586, psnr_mean=28.761768145576514
step_001500.pt official test: loss_mean=0.03485536935583999, psnr_mean=25.21410079036176
final.pt full-val: loss_mean=0.02255534678697586, psnr_mean=28.761768145576514
final.pt official test: loss_mean=0.03485536935583999, psnr_mean=25.21410079036176
```

Comparison to the previous official baseline:

```text
A03 final.pt official full-image test PSNR = 25.1402293880564
A_e03 final.pt official full-image test PSNR = 25.21410079036176
delta = +0.07387140230535955
```

Technical note:

```text
step_001500.pt and final.pt have identical model weights; their file hashes differ because the checkpoint package contains non-model content.
```

Conclusion:

```text
A_e03 is accepted as a completed high-VRAM edge-guided short-training experiment and becomes the new official UHD_deblur baseline, but the measured gain over A03 is modest rather than a staged meaningful jump.
```

Result analysis:

```text
experiments/V1/runs/A_e03/result_analysis.md
```
