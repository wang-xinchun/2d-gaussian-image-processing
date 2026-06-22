# Manifest: V1/T01/A_4kd17

- Created: `2026-06-11 CST`
- Run: `V1/T01/A_4kd17`
- Purpose: Test A08-teacher consistency as a targeted replacement for the lower-value A16 input-identity protection idea.
- Baseline: `A_4kd15` no-haze-guard EMA995 finetune from `A_4kd08` best-val.
- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Teacher checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
- Dataset: `4KDehaze_local_90_5_5_fast_probe_512`
- Template config: `/home/ldj/wxc/2DGauss/experiments/V1/templates/T01/configs/a_4kd17_fast_4kdehaze_a08_best_no_haze_guard_ema995_teacher_consistency_finetune.toml`
- Run-local config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd17/configs/a_4kd17_fast_4kdehaze_a08_best_no_haze_guard_ema995_teacher_consistency_finetune.toml`

## Hypothesis

`A_4kd15` improves many hard samples but damages the high-A08-PSNR tail. The measured per-sample relation is strongly negative between `A_4kd15 - A_4kd08` PSNR delta and `A_4kd08` PSNR, so the next useful mechanism is not input identity protection. `A_4kd17` anchors the student to the frozen `A_4kd08` teacher where the teacher is already close to GT, while leaving high-error regions mostly supervised by the ordinary restoration loss.

## Narrow Variables

- Keep `A_4kd15` architecture, optimizer, EMA, LR schedule, no haze guards, and `val_psnr` checkpoint selection.
- Add `loss.teacher_consistency_weight = 0.05`.
- Use `teacher_consistency_threshold = 0.08`, `teacher_consistency_softness = 0.02`.
- Do not add MSE in this run; isolate teacher anchoring before reintroducing PSNR-heavy loss.

## Success Criteria

- Strict PSNR supersede target: CPU test PSNR must exceed `A_4kd08` (`17.789133503142946`).
- Tail target: watched high-A08 samples such as `94_94000077/80/73/84/9` should recover relative to `A_4kd15`.
- Preserve useful A15 effect: hard-sample wins should not collapse back to pure `A_4kd08`.
