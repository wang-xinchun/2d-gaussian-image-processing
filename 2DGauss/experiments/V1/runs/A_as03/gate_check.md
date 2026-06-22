# Gate Check: V1/T01/A_as03

## Branch Identity

- Run: `V1/T01/A_as03`
- Config: `experiments/V1/runs/A_as03/configs/a_as03_short_uhddeblur_control_canvas_closed_loop_probe.toml`
- Mechanism family: supervised Gaussian control canvas / closed-loop feedback

## Static Gate

- [x] Config exists
- [x] Python compile/import check passed
- [x] Model forward/aux check passed
- [x] Parameter count checked: `372409`

## Smoke / Preflight Gate

- [x] Formal preflight passed at `6x640`, `val_psnr=23.350940153924366`
- [x] True smoke calibration passed at `6x640`
- [x] Reserved VRAM inside `85.0-93.0G`: `91.5078125G`
- [x] Average smoke step time recorded: `2.049535014977058s`
- [x] Estimated steps in 58 minutes recorded: `1697`

## Launch Gate

- [x] No conflicting active tmux training session
- [x] Training launched only after gates pass
- [x] Startup metrics writing

## Result Gate

- [x] Training finished cleanly
- [x] Checkpoint endpoint verified
- [x] Same-protocol official test completed
- [x] Task-specific diagnostics generated
- [x] Mechanism panels generated
