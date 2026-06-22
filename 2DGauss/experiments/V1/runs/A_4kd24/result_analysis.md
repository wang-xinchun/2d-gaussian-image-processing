# Result Analysis: V1/T01/A_4kd24

- Run: `V1/T01/A_4kd24`
- Completed: `2026-06-13 CST`
- Purpose: Convert the A18/A19/A21 two-checkpoint `input_std` router signal into a structural single-checkpoint dual-route residual head.
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/configs/a_4kd24_fast_4kdehaze_a15_input_std_dual_route_residual.toml`
- Training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/logs/train_20260613_191503.log`

## Setup

`A_4kd24` starts from the `A_4kd15` best-val checkpoint and adds a default-off,
zero-initialized route residual module near the terminal residual path.

```text
input_std >  0.07026463374495506 -> high route -> A_4kd08 teacher
input_std <= 0.07026463374495506 -> low route  -> A_4kd15 teacher
```

Unlike `A_4kd23`, `router_distillation_weight=0.0`; the two teachers are used
for high/low branch supervision only. The route losses are
`input_std_route_gate_weight=0.04` and `input_std_route_branch_weight=0.08`.

## Checks

- `python -m py_compile` for changed Python files: passed.
- venv tensor smoke for route forward keys/shape/zero-init behavior: passed.
- A24 TOML parse and path existence checks: passed.
- `bash -n` for A24 launchers: passed.
- GPU3 memory-gate dry run: passed.
- GPU3 preflight: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/logs/train_20260613_191253.log`, exit code `0`.
- Formal GPU3 training: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/logs/train_20260613_191503.log`, exit code `0`, completed step `400`.
- Best-val GPU standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/metrics/test_eval_best_val_gpu.json`.
- Best-val GPU task eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/metrics/test_task_metrics_best_val_gpu.json`.
- Final GPU standard eval: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/metrics/test_eval_final_gpu.json`.

## Training Trajectory

Validation improved monotonically at each validation checkpoint:

| Step | val_psnr | val_loss | Selection |
| ---: | ---: | ---: | --- |
| 100 | 18.2476802772934 | 0.12256812375038863 | best |
| 150 | 18.25239846933429 | 0.12248050704598427 | best |
| 200 | 18.24856095114244 | 0.12252095226198435 | no |
| 250 | 18.279412377125 | 0.12197740454226733 | best |
| 300 | 18.31380097682363 | 0.121388069242239 | best |
| 350 | 18.329543660956986 | 0.12109325230121612 | best |
| 400 | 18.349088096641417 | 0.1207316406071186 | best |

The route branch was active, not silently disabled. At step `400`,
`train_route_high_delta_abs_mean=0.0016153835458680987`,
`train_route_low_delta_abs_mean=0.0034981542266905308`, and
`loss_input_std_route_branch_weighted=0.004345999099314213`.

## Fast-Test Results

All rows use the same 100-sample crop512 fast-test protocol.

| Candidate | Checkpoint | PSNR | SSIM | loss | total_loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| A08 single | 700 | 17.789133503142946 | 0.6947606813907623 | 0.1228142374008894 | 0.12604619041085244 |
| A15 single | 300 | 17.781017099305146 | 0.6972798517346382 | 0.12060526199638844 | 0.12383682064712048 |
| A18 two-checkpoint router | n/a | 17.84693437641316 | 0.6992742219567298 | 0.12033024244010448 | 0.12356196045875549 |
| A23 best-val single | 50 | 17.78064128762362 | 0.6971213883161544 | 0.12059733912348747 | 0.12382887728512287 |
| A24 best-val single | 400 | 17.790202882742463 | 0.7000000587105751 | 0.12000417917966842 | 0.1232352689653635 |
| A24 final single | 400 | 17.558779176036488 | n/a | 0.12264519166201353 | 0.1258754152804613 |

## Interpretation

`A_4kd24` is a real single-checkpoint mechanism breakthrough relative to the
previous conversion attempts, but it is not yet a complete conversion of the
A18 two-checkpoint router.

- Versus `A_4kd23` best-val: `+0.009561595118843 dB` test PSNR, better SSIM,
  better loss, and better total loss.
- Versus `A_4kd15`: `+0.009185783437317 dB` test PSNR and better loss/SSIM.
- Versus `A_4kd08`: `+0.001069379599517 dB` test PSNR and much better SSIM/loss.
- Versus `A_4kd18` router: `-0.056731493670697 dB` test PSNR, but better SSIM
  and lower loss/total loss.

The main positive evidence is that structural routing finally exceeded the best
single-checkpoint PSNR points instead of collapsing to A15. The remaining gap is
that the route residual is still a small correction head on an A15 backbone; it
does not fully preserve the A08 high-`input_std` expert behavior that the
two-checkpoint router gets for free.

## Decision

Keep `A_4kd24` as the new best single-checkpoint fast-probe candidate and as the
first positive structural conversion result. Do not claim it supersedes the A18
two-checkpoint router on PSNR.

The next mechanism should be stronger than a small terminal route residual:
either a larger route-conditioned expert path, route-specific terminal decoder
heads, balanced high/low `input_std` sampling, or a single checkpoint that
stores two internal experts and gates between them.
