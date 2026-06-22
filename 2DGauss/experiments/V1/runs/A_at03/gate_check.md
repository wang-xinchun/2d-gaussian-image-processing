# Gate Check: V1/T01/A_at03

## Branch Identity

- Run: `V1/T01/A_at03`
- Config: `experiments/V1/runs/A_at03/configs/a_at03_short_uhddeblur_canvas_split_closed_loop_probe.toml`
- Mechanism family: supervised Gaussian control canvas / closed-loop feedback

## Static Gate

- [x] Config exists
- [x] Python compile/import check passed
- [x] Model forward/aux check passed
- [x] Parameter count checked: `372409`

## Smoke / Preflight Gate

- [x] Formal preflight passed at `6x640`, `val_psnr=23.674274706421937`
- [x] True smoke calibration passed at `6x640`
- [x] Reserved VRAM inside `85.0-93.0G`: `91.5078125G`
- [x] Average smoke step time recorded: `2.136026594787836s`
- [x] Estimated steps in 58 minutes recorded: `1629`

## Launch Gate

- [ ] No conflicting active tmux training session
- [ ] Training launched only after gates pass
- [ ] Startup metrics writing

## Result Gate

- [ ] Training finished cleanly
- [ ] Checkpoint endpoint verified
- [ ] Same-protocol official test completed
- [ ] Task-specific diagnostics generated
- [ ] Mechanism panels generated
