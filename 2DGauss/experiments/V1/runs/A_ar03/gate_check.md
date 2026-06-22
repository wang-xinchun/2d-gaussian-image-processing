# Gate Check: V1/T01/A_ar03

## Branch Identity

- Run: `V1/T01/A_ar03`
- Config: `experiments/V1/runs/A_ar03/configs/a_ar03_short_uhddeblur_control_canvas_aux_probe.toml`
- Mechanism family: supervised Gaussian control canvas / closed-loop feedback

## Static Gate

- [x] Config exists
- [x] Python compile/import check passed
- [x] Model forward/aux check passed
- [x] Parameter count checked: `372313`

## Smoke / Preflight Gate

- [x] Formal preflight passed at `6x640`, `val_psnr=23.929713178947054`
- [x] True smoke calibration passed at `6x640`
- [x] Reserved VRAM inside `85.0-93.0G`: `91.490234375G`
- [x] Average smoke step time recorded: `2.0045870194832482s`
- [x] Estimated steps in 58 minutes recorded: `1736`

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
