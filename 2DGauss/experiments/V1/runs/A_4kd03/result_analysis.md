# A_4kd03 Result Analysis

## Verdict

`A_4kd03` is a useful negative probe, not an accepted candidate. It confirmed
that a predicted restoration-demand control canvas can be supervised and
diagnosed, but hard-replacing the main Gaussian attention with that early
canvas hurts the fast-probe trajectory.

## Evidence

- Preflight passed after reducing runtime to `batch_size=2`, `grad_accum=4`.
- Formal training used physical GPU3 only and stayed around `11.2GiB` peak
  reserved memory.
- Step `100`: `val_loss=0.2597135889157653`,
  `val_psnr=12.1511608526603`.
- Step `200`: `val_loss=0.22252442486584187`,
  `val_psnr=13.527238149476778`.
- `A_4kd02` reached `val_psnr=14.261986201609318` at step `100` and
  `17.529067811582326` at step `800` on the same fast-probe view.

## Control Diagnostics

- Preflight `val_control_canvas_corr=0.00371266288235347`.
- Step `100` `val_control_canvas_corr=0.16863191392738372`,
  `val_control_canvas_std=0.0023506595322396605`.
- Step `200` `val_control_canvas_corr=0.1268156026955694`,
  `val_control_canvas_std=0.0064538781507872045`.
- Target variation is much larger:
  `val_control_target_std=0.10203479953110219`.

## Interpretation

The model starts to learn a weak demand direction, but the control canvas is
still close to a low-variance field. Sharing that field by replacing the main
attention path is too destructive: it dampens the original Gaussian guidance
before the control map is mature enough to carry the role.

## Next Step

Use a conservative blended shared-control probe: keep the original Gaussian
attention as the main signal and mix in a small fraction of the control canvas,
with lower feedback strength and lower budget.
