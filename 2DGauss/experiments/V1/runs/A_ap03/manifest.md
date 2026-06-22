# Run Manifest: V1/T01/A_ap03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_ap03`
- Full identifier: `V1/T01/A_ap03`

## Current Status

Completed and closed out. Not a promotion candidate.

## Purpose

`A_ap03` is the next mainline after the negative `A_am03` result. It keeps the structured-control router but removes hard role competition as the default behavior. The question is whether role-specific control can become stable if each module keeps the shared prior as a common base and only receives centered residual role deltas, with competition delayed and limited to high-demand regions.

## Mechanism

```text
use_gaussian_control_router = true
gaussian_control_router_use_competitive_role_gate = false
gaussian_control_router_use_role_residual_centering = true
gaussian_control_router_use_demand_conditioned_competition = true
gaussian_control_router_role_delta_strength = 0.42
gaussian_control_router_role_competition_start = 0.45
gaussian_control_router_role_competition_strength = 0.30
gaussian_control_router_role_competition_floor = 0.45
```

The router first behaves as a cooperative residual role controller. Role deltas are centered across decoder, feature, and edge outputs, so the shared prior remains the common control language and role branches only express relative residual specialization. A learned demand map then gradually introduces a floored soft competition after training progress passes `0.45`, avoiding the early global role suppression that likely hurt `A_am03`.

## Baselines

- Frozen medium baseline: `M04/A_y03`, official test `26.272695829299355`
- Current short leader: `A_z03`, official test `25.385710820365173`
- Best completed structured control: `A_ah03`, official test `25.345361164304613`
- Structured-router predecessor: `A_aj03`, official test `25.2308542283191`
- Negative competitive-router result: `A_am03`, official test `25.169950648899793`

## Protocol

- Dataset: `UHD_deblur`
- Train split: after first `40` pairs
- Internal validation: first `20` train pairs, full image
- Official test: post-hoc, `300` full-image pairs
- Initial geometry: `batch_size=6`, `train_context_size=696`, supervision crop `512`
- Retuned geometry after 696 preflight OOM: `batch_size=6`, `train_context_size=688`, supervision crop `512`
- Retuned launch geometry after 688 smoke exceeded reserved-memory window: `batch_size=6`, `train_context_size=680`, supervision crop `512`
- Budget: `max_steps=2500`, `wall_clock_minutes=58`
- Reserved VRAM target: `85.0-93.0G`

## Gate Status

- Static py_compile: passed before branch creation
- Config parser/small forward: passed at `6x688` and `6x680`
- Parameter count: `348700`
- Formal preflight: passed at `6x680`, `val_psnr=17.340130889173448`
- True smoke calibration: passed at `6x680`
- Reserved VRAM: `91.56640625G`, inside `85.0-93.0G`
- Average smoke step time: `2.008628693098823s`
- Estimated steps in 58 minutes: `1732`
- Launch: completed as `2dgauss_a_ap03_uhddeblur_demand_residual_soft_router_probe`

## Training Result

- Training log: `experiments/V1/runs/A_ap03/logs/train_20260519_165513.log`
- Finish: clean, `step=2500`
- Final train metric: `loss_total=0.02709762565791607`
- Validation trajectory:
  - `step=500`, `val_psnr=28.512026467667624`, `val_loss=0.025119276158511637`
  - `step=1000`, `val_psnr=28.665201331008205`, `val_loss=0.02411575373262167`
  - `step=1500`, `val_psnr=28.306285066358477`, `val_loss=0.025638479460030793`
  - `step=2000`, `val_psnr=28.602542903189573`, `val_loss=0.024406322557479144`
  - `step=2500`, `val_psnr=28.759788226347787`, `val_loss=0.023809175845235585`
- Checkpoint endpoint: `step_002500.pt`, `latest.pt`, and `final.pt` share identical model weights.

## Official Evaluation

- Accepted final checkpoint: `checkpoints/final.pt`
- Same-protocol official test: `samples=300`, `split=test`, `crop_size=0`, `device=cuda`
- Final result: `psnr_mean=25.233562564509878`, `loss_mean=0.03691591784358025`

## Final Decision

`A_ap03` is a partial improvement over `A_am03` and a real positive short probe, but it still falls short of `A_ah03` and `A_z03`, so it is not a promotion candidate. The demand-conditioned residual soft router appears more stable than hard competitive routing, yet it does not fully solve the cross-split selectivity gap.

## Decision Rule

Launch only after formal preflight and true smoke are inside the reserved-memory window. Treat this branch as the active mainline if it is stable and its validation trajectory recovers toward or above `A_ah03`; if it underperforms `A_aj03`, the structured-router family likely needs a different control schedule or architecture rather than more local role-gate tweaks.
