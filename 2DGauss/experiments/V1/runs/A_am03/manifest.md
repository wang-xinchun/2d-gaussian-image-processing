# Run Manifest: V1/T01/A_am03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_am03`
- Full identifier: `V1/T01/A_am03`

## Current Status

Completed and closed out. Not a promotion candidate.

## Purpose

`A_am03` is the next strong-mechanism mainline after `A_aj03`. It asks whether adding an explicit competitive budget to the structured role router makes the shared control surface more selective without losing the cross-module coordination signal.

## Mechanism

```text
use_gaussian_control_router = true
gaussian_control_router_use_decoder_state = true
gaussian_control_router_use_edge_state = true
edge_guidance_use_control_gate = true
gaussian_control_router_use_competitive_role_gate = true
gaussian_control_router_role_gate_temperature = 0.75
gaussian_feature_refinement_use_intervention_demand = false
```

The router uses a competitive softmax gate across decoder, feature, and edge roles, then applies role-specific deltas under a shared prior.

## Baselines

- Frozen medium baseline: `M04/A_y03`, official test `26.272695829299355`
- Current short leader: `A_z03`, official test `25.385710820365173`
- Structured-router predecessor: `A_aj03`, official test `25.2308542283191`
- Best completed structured control: `A_ah03`, official test `25.345361164304613`
- Negative single-gate diagnostic: `A_ai03`, official test `25.267052413067724`

## Protocol

- Dataset: `UHD_deblur`
- Train split: after first `40` pairs
- Internal validation: first `20` train pairs, full image
- Official test: post-hoc, `300` full-image pairs
- Geometry: `batch_size=6`, `train_context_size=688`, supervision crop `512`
- Budget: `max_steps=2500`, `wall_clock_minutes=58`
- Reserved VRAM target: `85.0-93.0G`

## Gate Status

- Static py_compile: passed
- Config parser/small forward: passed
- Parameter count checked: `347979`
- Formal preflight: passed at `6x688`
- True `6x688` smoke: passed
- Reserved VRAM: `92.77734375G`, inside `85.0-93.0G`
- Launch: completed as `2dgauss_a_am03_uhddeblur_structured_competitive_router_probe`

## Training Result

- Training log: `experiments/V1/runs/A_am03/logs/train_20260519_145711.log`
- Finish: clean, `step=2500`
- Final train metric: `loss_total=0.030509477481245995`
- Validation trajectory:
  - `step=500`, `val_psnr=28.450075673367785`, `val_loss=0.02533218963071704`
  - `step=1000`, `val_psnr=28.207115233777547`, `val_loss=0.025883095990866422`
  - `step=1500`, `val_psnr=28.602079030904918`, `val_loss=0.023843193985521795`
  - `step=2000`, `val_psnr=28.587588177783367`, `val_loss=0.0239897514693439`
  - `step=2500`, `val_psnr=28.439035481471315`, `val_loss=0.02499520471319556`
- Checkpoint endpoint: `step_002500.pt`, `latest.pt`, and `final.pt` share identical model weights.

## Official Evaluation

- Accepted final checkpoint: `checkpoints/final.pt`
- Same-protocol official test: `samples=300`, `split=test`, `crop_size=0`, `device=cuda`
- Final result: `psnr_mean=25.169950648899793`, `loss_mean=0.03711430704065909`
- Best-validation diagnostic checkpoint: `step_001500.pt`
- Diagnostic result: `psnr_mean=25.105170630019582`, `loss_mean=0.03717060950274269`

## Final Decision

`A_am03` is a clean engineering completion but a negative mechanism result. The competitive role gate is trainable, yet its official final result is below `A_aj03`, `A_ai03`, `A_ah03`, `A_z03`, and frozen `M04/A_y03`. The `step_001500.pt` diagnostic is also worse than final, so the validation peak is not a useful early-stop breakthrough. Do not promote, replay, or tune only the competitive temperature as the next mainline.
