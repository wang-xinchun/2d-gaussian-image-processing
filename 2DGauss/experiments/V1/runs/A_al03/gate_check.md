# Gate Check: V1/T01/A_al03

## Branch Identity

- Run: `V1/T01/A_al03`
- Config: `experiments/V1/runs/A_al03/configs/a_al03_short_uhddeblur_router_feature_demand_probe.toml`
- Mechanism: full structured router plus intervention-demand feature refinement

## Static Gate

- [x] Config exists
- [x] Small CPU forward passed
- [x] Parameter count checked: `364251`
- [x] Python compile/import check passed

## Smoke / Preflight Gate

- [ ] Formal preflight passed
- [ ] True `6x696` smoke calibration passed
- [ ] Reserved VRAM inside `85.0-93.0G`

## Launch Gate

- [x] No conflicting active tmux training session
- [ ] Training launched
- [ ] Startup metrics writing

## Initial Decision

Prepared. Run after `A_aj03` if the full router is stable but still not selective enough.
