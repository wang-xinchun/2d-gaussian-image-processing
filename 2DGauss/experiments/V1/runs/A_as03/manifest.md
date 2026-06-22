# Run Manifest: V1/T01/A_as03

## Identity

- Architecture: `V1`
- Training template: `T01`
- Training run: `A_as03`
- Full identifier: `V1/T01/A_as03`

## Current Status

Completed. Geometry remained `6x640`; the run was launched after gates passed, finished cleanly, and has been fully evaluated.

## Purpose

`A_as03` is part of the post-`A_aq03` strong-mechanism short-probe family. The family tests whether a supervised Gaussian control canvas and closed-loop feedback can break the flat-control bottleneck observed in `A_aq03`.

## Mechanism

- `A_ar03`: control-canvas auxiliary supervision only; feedback strength is zero, so it tests whether the canvas itself becomes structured without changing restoration features.
- `A_as03`: control-canvas auxiliary supervision plus target-conditioned closed-loop feedback into decoder features.
- `A_at03`: explicit frequency split plus structure hint plus control-canvas closed-loop feedback, testing whether split responsibility and supervised canvas are complementary.

This run uses config: `experiments/V1/runs/A_as03/configs/a_as03_short_uhddeblur_control_canvas_closed_loop_probe.toml`.

## Baselines

- Frozen medium baseline: `M04/A_y03`, official test `26.272695829299355`
- Current short leader: `A_z03`, official test `25.385710820365173`
- Latest split probe: `A_aq03`, official test `25.281006597978177`, attention std `≈0.00017`

## Gate Status

- [x] Static py_compile
- [x] Config parse/model forward
- [x] Formal preflight at `6x640`
- [x] True smoke calibration inside the memory window: `91.5078125G`
- [x] Launch after gates passed
- [x] Training finished cleanly
- [x] Checkpoint endpoint verified
- [x] Same-protocol official test completed
- [x] Task-specific diagnostics generated
- [x] Mechanism panels generated

## Decision Rule

A run is useful only if it improves the control-canvas story and remains competitive in official fidelity. Promotion evidence requires same-protocol official test, task-specific diagnostics, and mechanism panels showing non-flat control/canvas maps without ringing, halo, over-sharpening, checkerboard artifacts, or color shift.
