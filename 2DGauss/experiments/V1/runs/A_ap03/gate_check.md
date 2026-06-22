# Gate Check: V1/T01/A_ap03

## Branch Identity

- Run: `V1/T01/A_ap03`
- Config: `experiments/V1/runs/A_ap03/configs/a_ap03_short_uhddeblur_demand_residual_soft_router_probe.toml`
- Mechanism: demand-conditioned residual soft router

## Static Gate

- [x] Config exists
- [x] Python compile/import check passed before branch creation
- [x] Config parser/small forward passed at `6x688`
- [x] Parameter count checked: `348700`
- [x] Retuned `6x680` formal preflight confirmation passed: `val_psnr=17.340130889173448`

## Smoke / Preflight Gate

- [x] Formal preflight passed at `6x688`
- [x] True smoke calibration passed at `6x680`
- [x] Reserved VRAM inside `85.0-93.0G`: `91.56640625G`
- [x] Average smoke step time recorded: `2.008628693098823s`
- [x] Estimated steps in 58 minutes recorded: `1732`

## Launch Gate

- [x] No conflicting active tmux training session
- [x] Training launched
- [x] Startup metrics writing
- [x] Training finished cleanly at `step=2500`
- [x] `step_002500.pt`, `latest.pt`, and `final.pt` have identical model weights

## Official Result Gate

- [x] Final checkpoint official test completed
- [x] `final.pt`: `psnr_mean=25.233562564509878`, `loss_mean=0.03691591784358025`, `samples=300`, `crop_size=0`, `device=cuda`

## Initial Decision

Prepared. This branch directly tests the post-`A_am03` bottleneck diagnosis: hard competition failed, so role control should remain cooperative early and become only demand-conditioned and floored later.

## Final Decision

Completed but not a promotion candidate. `A_ap03` improved over `A_am03`, which validates the residual soft-routing direction, but it still does not overtake `A_ah03` or `A_z03`.
