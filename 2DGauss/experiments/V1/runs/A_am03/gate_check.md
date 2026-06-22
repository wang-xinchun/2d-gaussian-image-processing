# Gate Check: V1/T01/A_am03

## Branch Identity

- Run: `V1/T01/A_am03`
- Config: `experiments/V1/runs/A_am03/configs/a_am03_short_uhddeblur_structured_competitive_router_probe.toml`
- Mechanism: structured role router with competitive gate

## Static Gate

- [x] Config exists
- [x] Small CPU forward passed
- [x] Parameter count checked: `347979`
- [x] Python compile/import check passed

## Smoke / Preflight Gate

- [x] Formal preflight passed at `6x688`
- [x] True `6x688` smoke calibration passed
- [x] Reserved VRAM inside `85.0-93.0G`: `92.77734375G`

## Launch Gate

- [x] No conflicting active tmux training session at launch
- [x] Training launched
- [x] Startup metrics writing
- [x] Training finished cleanly at `step=2500`
- [x] `step_002500.pt`, `latest.pt`, and `final.pt` have identical model weights

## Official Result Gate

- [x] Final checkpoint official test completed
- [x] `final.pt`: `psnr_mean=25.169950648899793`, `loss_mean=0.03711430704065909`, `samples=300`, `crop_size=0`, `device=cuda`
- [x] Best-validation checkpoint diagnostic completed
- [x] `step_001500.pt`: `psnr_mean=25.105170630019582`, `loss_mean=0.03717060950274269`, `samples=300`, `crop_size=0`, `device=cuda`

## Final Decision

Completed but rejected as a mainline breakthrough. Competitive gating did not improve selectivity relative to `A_aj03`; it underperformed `A_aj03`, `A_ai03`, `A_ah03`, and `A_z03` on the same official full-image test protocol. The best-validation checkpoint was also worse than the final checkpoint on test, so this is not an early-stop salvage case.
