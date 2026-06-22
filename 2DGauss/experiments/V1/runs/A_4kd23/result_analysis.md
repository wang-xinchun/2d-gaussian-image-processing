# Result Analysis: V1/T01/A_4kd23

- Run: `V1/T01/A_4kd23`
- Completed: `2026-06-12 CST`
- Purpose: Convert the A18/A19/A21 two-checkpoint `input_std` router into one student checkpoint through constrained output distillation.
- Config: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/configs/a_4kd23_fast_4kdehaze_a15_router_distill_input_std_single_model.toml`
- Training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd23/logs/train_20260612_225300.log`

## Setup

`A_4kd23` starts from the `A_4kd15` best-val checkpoint and keeps the A15
single-model training recipe, then adds one default-off training branch:

```text
input_std <= 0.07026463374495506 -> A_4kd15 teacher
input_std  > 0.07026463374495506 -> A_4kd08 teacher
```

The router distillation branch uses `router_distillation_weight=0.03`,
`router_distillation_softness=0.005`, and strict teacher loading. Teacher A is
`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`
at step `700`; teacher B is
`/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`
at step `300`.

## Checks

- `python -m py_compile scripts/train_v1_t01.py`: passed.
- Run-local TOML parse and checkpoint existence checks: passed.
- CPU tensor smoke for `compute_router_distillation_loss`: passed.
- `bash -n` for launchers: passed.
- GPU3 memory-gate dry run: passed with free memory above `12288MiB`, even when GPU util was high under the user's current memory-first rule.
- Formal preflight: passed, `train_exit_code=0`, strict init and both strict teacher loads had no missing or unexpected keys.
- Formal training: completed `300` steps, `train_exit_code=0`.

## Training Trajectory

Validation selected the earliest checkpoint:

| Step | val_psnr | val_loss | Selection |
| ---: | ---: | ---: | --- |
| 50 | 18.256583743058567 | 0.12251933488994837 | best |
| 100 | 18.23805755625296 | 0.12280667465180159 | no |
| 150 | 18.211043605486207 | 0.12323430690914393 | no |
| 200 | 18.17693582810965 | 0.12380122907459735 | no |
| 250 | 18.160365789371642 | 0.12404881782829762 | no |
| 300 | 18.154180648966136 | 0.12409969229251146 | no |

The router-distillation diagnostics were active during training. At step `300`,
`loss_router_distillation=0.041278231889009476` and weighted router loss was
`0.0012383469147607684`, confirming that the branch was not silently disabled.

## Fast-Test Results

All rows use the same 100-sample crop512 fast-test protocol.

| Candidate | Checkpoint | PSNR | SSIM | loss | total_loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| A08 single | 700 | 17.789133503142946 | 0.6947606813907623 | 0.1228142374008894 | 0.12604619041085244 |
| A15 single | 300 | 17.781017099305146 | 0.6972798517346382 | 0.12060526199638844 | 0.12383682064712048 |
| A18 two-checkpoint router | n/a | 17.84693437641316 | 0.6992742219567298 | 0.12033024244010448 | 0.12356196045875549 |
| A23 best-val single | 50 | 17.78064128762362 | 0.6971213883161544 | 0.12059733912348747 | 0.12382887728512287 |
| A23 final single | 300 | 17.035365402315897 | n/a | 0.13115204714238643 | 0.13438382618129252 |

## Interpretation

`A_4kd23` is an engineering success but a negative scientific conversion result.
The default-off router-distillation infrastructure works, loads both teachers
strictly, produces diagnostics, trains on GPU3, and writes standard checkpoints
and metrics. However, the best single checkpoint only matches the A15 operating
point within noise and does not preserve the A18 two-checkpoint router gain:

- A23 best-val vs A15: `-0.000375811681526 dB` PSNR, slightly lower SSIM, and
  slightly lower loss.
- A23 best-val vs A08: `-0.008492215519326 dB` PSNR.
- A23 best-val vs A18 router: `-0.06629308878954 dB` PSNR.
- A23 final is much worse than best-val, so later training drifts away from the
  useful A15-like point.

The likely issue is that weak output distillation from an A15-initialized
student does not create or preserve a separate high-input-std A08 branch. The
student remains mostly A15-like early, then degrades with continued fine-tuning.

## Decision

Keep the router-distillation code path as reusable default-off infrastructure,
but do not promote `A_4kd23` as a candidate and do not claim it converted the
A18/A19/A21 milestone into a single checkpoint. The next conversion attempt
should use a stronger structural constraint: hard per-sample router labels,
balanced high/low `input_std` batches, residual/correction distillation, a
learned gating head, dual-head/internal routing, or an A08-preserving branch
rather than weak whole-output distillation into an A15-only student.
