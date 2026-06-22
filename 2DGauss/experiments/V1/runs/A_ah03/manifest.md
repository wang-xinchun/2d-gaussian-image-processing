# Run Manifest: V1/T01/A_ah03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ah03`
- Full identifier: `V1/T01/A_ah03`

## Current Status

Smoke calibration and formal preflight have passed, and this branch is the completed short probe in the next probe family after `A_ae03`. It kept the accepted `M04/A_y03` short envelope fixed, but changed edge refinement so it could be controlled rather than acting only as a late residual helper. Training finished cleanly at `step=2500`, and the final official test has been recorded.

## Purpose

`A_ah03` exists to test a role-separation hypothesis:

> if the current bottleneck is not more edge information but better module coordination, can edge refinement become useful only when it is allowed to read the shared control surface?

## Experimental Question

Can a control-gated edge refiner improve the current shared-control surface while staying inside the same short-tier runtime envelope?

## Chosen Mechanism

`A_ah03` keeps:

```text
use_gaussian_decoder_guidance = true
use_gaussian_feature_refinement = true
use_edge_weighted_reconstruction = true
use_edge_guidance = true
use_directional_frequency_guidance = true
use_train_inference_consistency = true
use_gaussian_shared_prior = true
```

and changes:

```text
edge_guidance_use_control_gate = true
```

The edge refiner now receives control information from the shared prior, so it can act as a coordinated residual helper instead of a standalone late edge booster.

## Baseline For Comparison

- Primary frozen baseline: `M04/A_y03`
- Current strongest completed short continuation: `V1/T01/A_z03`
- Local shared-control result to beat: `V1/T01/A_ae03`
- Direct edge-aware negative reference: `V1/T01/A_ab03`

## Planned Protocol

Kept from the current short envelope:

- dataset: `UHD_deblur`
- effective training split: train pairs after the first `40`
- internal validation: first `20` train pairs
- validation crop: full image, `crop_size=0`
- official test: `300` full-image pairs, post-hoc only
- edge-guided refinement
- directional/frequency guidance
- `TLC`-style train-inference consistency
- decoder guidance `strength = 0.15`
- feature refinement `strength = 0.10`
- edge-weighted reconstruction `strength = 0.02`
- short-tier wall-clock target

Changed from nearby short branches:

- edge refinement becomes control-gated
- the branch still keeps shared prior enabled
- the branch does not introduce a new cue source

## Planned Runtime Envelope

- Tier target: short
- Wall-clock target: `58` minutes
- Max steps target: `2500`
- Accepted batch-size / context geometry: `6x720`
- Gradient accumulation target: `1`
- Train supervision size: `512`
- Validation interval target: `500`
- Checkpoint interval target: `250`
- Validation batches target: `20`
- Target reserved VRAM window: `85.0-93.0G`

## Verification To Be Performed

Accepted/planned items:

1. branch-local smoke calibration on the true config: passed;
2. formal `train_v1_t01.py --preflight` on the true config: passed;
3. clean short-budget training completion: passed;
4. checkpoint-endpoint equality verification: passed;
5. same-protocol official full-image test on `final.pt`: passed.

## Calibration Notes

Branch-local smoke on the true `A_ah03` config shows:

- `6x720`: `reserved_gb=91.826171875`, accepted
- `avg_total_step_seconds=1.9132`
- `estimated_steps_58min=1818`

Formal preflight on the final config shows:

- `batch_input_shape=(6, 3, 720, 720)`
- `batch_target_shape=(6, 3, 720, 720)`
- `total_params=305256`
- `val_psnr=23.6768`

Launch status:

- tmux: completed; session `2dgauss_a_ah03_uhddeblur_edge_control_gate_probe` has exited
- log: `experiments/V1/runs/A_ah03/logs/train_20260518_234142.log`
- startup GPU: `memory.used≈94347 MiB`, `utilization.gpu≈97%`
- first metric: `step=1`, `loss_total=0.1084759533405304`
- first validation: `step=500`, `val_psnr=28.527268073041363`, `val_loss=0.024733865913003684`, `val_batches=20`
- second validation: `step=1000`, `val_psnr=28.61423288220579`, `val_loss=0.024983534589409828`, `val_batches=20`
- final validation: `step=2500`, `val_psnr=29.039882709180723`, `val_loss=0.022765929251909255`, `val_batches=20`
- latest checked training metric: `step=2500`, `elapsed_minutes=36.139669915040336`, `loss_total=0.044364772737026215`
- latest checked GPU during training: `memory.used≈94315 MiB`, `utilization.gpu≈97%`
- checkpoints present at latest check: `step_000250.pt`, `step_000500.pt`, `step_000750.pt`, `step_001000.pt`, `step_001250.pt`, `step_001500.pt`, `step_001750.pt`, `step_002000.pt`, `step_002250.pt`, `step_002500.pt`, `latest.pt`, `final.pt`
- official test: `checkpoint_step=2500`, `samples=300`, `loss_mean=0.03634987864177674`, `psnr_mean=25.345361164304613`

## Working Hypothesis

If the bottleneck is module role confusion, then edge refinement should improve only when it is made dependent on the shared control surface rather than being allowed to free-run as a late residual patch.
