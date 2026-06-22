# Gate Check: V1/T01/A_aj03

## Branch Identity

- Run: `V1/T01/A_aj03`
- Config: `experiments/V1/runs/A_aj03/configs/a_aj03_short_uhddeblur_structured_role_router_probe.toml`
- Mechanism: full structured role router

## Static Gate

- [x] Config exists
- [x] Small CPU forward passed
- [x] Parameter count checked: `347979`
- [x] Python compile/import check passed

## Smoke / Preflight Gate

- [x] Formal preflight passed
- [x] True `6x696` smoke calibration passed
- [x] Reserved VRAM inside `85.0-93.0G`

## Launch Gate

- [x] No conflicting active tmux training session
- [x] Training launched
- [x] Startup metrics writing

## Initial Decision

Prepared. Initial `6x720` preflight exceeded available VRAM in the edge-refiner path; `6x704` preflight passed but smoke reserved `93.8086G`, slightly above the `93G` ceiling. The accepted strong-mechanism scout geometry is reduced to `6x696`. Launch only after true-config formal preflight and smoke pass at `6x696`.
