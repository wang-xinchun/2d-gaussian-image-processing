# A_4kd27 Result Analysis

## Summary

`A_4kd27` implements the planned high/low `input_std` route terminal heads from the `A_4kd24` best-val checkpoint. The engineering path is complete: default-off model/config support, A27 config and launchers, static checks, smoke checks, GPU3 preflight, 400-step training, standard eval, task metrics eval, A18-record alignment, and per-regime route diagnostics all completed.

The scientific result is negative for promotion. A27 improves several proxy/task metrics, but it does not beat A24 on fast-test PSNR and it worsens the high-route gap against the A18 two-checkpoint router.

## Implementation

- Init checkpoint: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd24/checkpoints/best_val.pt`, non-strict.
- Teacher A/high route: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd08/checkpoints/best_val.pt`.
- Teacher B/low route: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd15/checkpoints/best_val.pt`.
- Route rule: high route when `input_std > 0.07026463374495506`, low route otherwise.
- New mechanism: `use_input_std_route_terminal_heads=true`; old `use_input_std_route_residual=false`.
- Terminal heads: shared route encoder, high expert, low expert, high RGB head, low RGB head, learned gate head.
- Route strength: `0.60`; terminal blocks: `2`.
- Branch losses: high/A08 `0.14`, low/A15 `0.06`; whole-output router distillation remains `0.0`.
- Sampling: natural fast-probe sampling; no A25/A26 balanced sampler.

Changed code/config paths:

- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`
- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/__init__.py`
- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
- `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`
- `/home/ldj/wxc/2DGauss/scripts/evaluate_a18_record_alignment_v1_t01.py`
- `/home/ldj/wxc/2DGauss/scripts/evaluate_input_std_route_regimes_v1_t01.py`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/configs/a_4kd27_fast_4kdehaze_a24_route_terminal_heads.toml`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/launch_tmux.sh`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/wait_and_launch_a_4kd27_memory_gated_gpu_safe.sh`

## Checks

- `py_compile` passed for `config.py`, `__init__.py`, `restorer.py`, `train_v1_t01.py`, `evaluate_a18_record_alignment_v1_t01.py`, and `evaluate_input_std_route_regimes_v1_t01.py`.
- TOML parse and checkpoint path checks passed.
- Launcher `bash -n` passed.
- Forward smoke passed: terminal aux keys exist, shapes are correct, and zero-init deltas do not disturb the baseline.
- Default-disabled smoke passed: old configs do not emit terminal-head aux keys.
- Teacher branch-loss smoke passed: high and low branch losses are nonzero and gradients reach the route heads.
- First GPU3 preflight with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.30` failed because the process memory cap was too tight, not because GPU3 was full.
- Second GPU3 preflight with `TWODGAUSS_CUDA_MEMORY_FRACTION=0.50` passed.

## Training

- Formal training log: `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/logs/train_20260614_131135.log`
- Exit code: `0`
- Device: physical GPU3
- Steps: `400`
- Best checkpoint: step `400`
- Best validation: `val_psnr=18.41533773497595`, `val_loss=0.11968508075922728`

Validation progression:

| Step | Val PSNR | Val Loss |
| ---: | ---: | ---: |
| 50 | 18.334242075288238 | 0.12100242651998996 |
| 100 | 18.352142568918897 | 0.12070204943418503 |
| 150 | 18.382412528203925 | 0.12019331585615874 |
| 200 | 18.363547696607224 | 0.12052278086543083 |
| 250 | 18.3611512027818 | 0.12056647941470146 |
| 300 | 18.356097289229027 | 0.12068005725741386 |
| 350 | 18.37806915952928 | 0.12031993735581636 |
| 400 | 18.41533773497595 | 0.11968508075922728 |

## Fast-Test Metrics

Primary eval output:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/metrics/test_eval_best_val_gpu.json`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/metrics/test_task_metrics_best_val_gpu.json`

| Candidate | PSNR | SSIM | Loss | Total Loss |
| --- | ---: | ---: | ---: | ---: |
| A24 best single checkpoint | 17.790202882742463 | 0.7000000587105751 | 0.12000417917966842 | 0.1232352689653635 |
| A27 best-val | 17.7513865188403 | 0.700952832698822 | 0.1202681017667055 | 0.12349890783429146 |
| A18 two-checkpoint router | 17.84693437641316 | 0.6992742219567298 | 0.12033024244010448 | 0.12356196045875549 |

A27 passes the SSIM and loss proxy thresholds, but fails the primary PSNR promotion gate.

Task/proxy metrics:

| Metric | A27 |
| --- | ---: |
| edge_magnitude_l1_mean | 0.15703101120889187 |
| high_frequency_l1_mean | 0.021589383380487562 |
| local_contrast_l1_mean | 0.024256876949220896 |
| dark_channel_l1_mean | 0.1394224874675274 |
| chroma_l1_mean | 0.02999501518905163 |
| saturation_l1_mean | 0.05453904043883085 |

These proxy metrics are mostly slightly better than A24, but the PSNR drop means A27 did not solve the high-fidelity tail.

## A18 Alignment

Alignment output:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/metrics/a18_record_alignment_best_val_gpu.json`

| Group | Count | A27 PSNR | A18 Router PSNR | Gap vs Router | Gap vs Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 100 | 17.7513865188403 | 17.84693437581442 | -0.09554785697412059 | -0.13216927624623692 |
| Low route | 57 | 16.621413377989295 | 16.558633271774358 | +0.06278010621493545 | +0.005668035439059631 |
| High route | 43 | 19.249257891596283 | 19.55468235093729 | -0.3054244593410088 | -0.3148838522011649 |
| A15-oracle | 54 | 16.492842893725346 | - | +0.2100120739571167 | +0.2024795944573628 |
| A08-oracle | 46 | 19.22880729614916 | - | -0.45424864545861654 | -0.5250179505504626 |

The low route remains healthy: `+0.06278 dB` vs A18 router, satisfying the `>= +0.03 dB` requirement. The high route is the failure point: `-0.30542 dB` vs A18 router, worse than A24's `-0.224697 dB` and far below the `>-0.10 dB` target.

## Route-Regime Diagnostics

Route-regime output:

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd27/metrics/input_std_route_regime_best_val_gpu.json`

| Group | Count | PSNR | Loss | Gate Mean | Prior Mean | High Delta Abs | Low Delta Abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 100 | 17.7513865188403 | 0.1202681017667055 | 0.43046327508986 | 0.42865418231150215 | 0.0007448780169943348 | 0.0011624186136759817 |
| Low route | 57 | 16.621413377989295 | 0.13678760748160512 | 0.08474766123190261 | 0.04450152798111511 | 0.0007453924708738269 | 0.001151348803225055 |
| High route | 43 | 19.249257891596283 | 0.0983701523306758 | 0.8887374609015709 | 0.9378797938657362 | 0.0007441960665029149 | 0.0011770925484597683 |

Branch-to-teacher L1:

| Group | High Branch vs A08 | Low Branch vs A15 | Selected Branch vs Teacher |
| --- | ---: | ---: | ---: |
| Overall | 0.018603432322852314 | 0.013385437512770295 | 0.014530827612616122 |
| Low route | 0.02185701047838257 | 0.01471208993410855 | 0.01471208993410855 |
| High route | 0.01429054965156802 | 0.011626851744949818 | 0.01429054965156802 |

The learned gate follows the fixed prior reasonably well, so the failure is not primarily route selection. The deltas remain tiny, especially the high selected delta around `0.000744`, which indicates the terminal heads still behave like a weak late correction rather than a full A08-like high-route expert.

## Acceptance

| Criterion | Target | A27 | Result |
| --- | ---: | ---: | --- |
| Beat A24 PSNR | > 17.790202882742463 | 17.7513865188403 | Fail |
| Hopeful target | >= 17.82 | 17.7513865188403 | Fail |
| A18 milestone | close to or beat 17.84693437641316 | 17.7513865188403 | Fail |
| High-route gap | > -0.10 dB | -0.3054244593410088 | Fail |
| Low-route gap | >= +0.03 dB | +0.06278010621493545 | Pass |
| SSIM | >= 0.699 | 0.700952832698822 | Pass |
| Loss | <= 0.1205 | 0.1202681017667055 | Pass |

## Interpretation

A27 is useful because it cleanly tests the theory gap, but it does not close it. The result says:

1. Fixed `input_std` routing is still a good prior; the gate/prior diagnostics are not the bottleneck.
2. Low-route/A15-like behavior remains preservable inside one checkpoint.
3. High-route/A08-tail preservation is still not recovered.
4. A terminal head after the shared Gaussian/control/decoder/base-residual path is still too late and too weak, even when upgraded from A24's small delta head to higher-capacity terminal experts.

The likely scientific explanation is that high-`input_std` fidelity depends on upstream representation and control decisions, not only the final RGB projection. By the time A27 applies high/low terminal heads, the shared path has already compressed the sample into a mixed A24-like feature state. The high terminal branch can imitate A08 in local L1 terms, but it cannot recover the A08 checkpoint's full internal behavior on high-fidelity tail samples.

## Next Direction

Do not promote A27 and do not keep sweeping terminal-head weights as the mainline. Keep A24 as the current best single-checkpoint fast-probe candidate, and keep A18 as the two-checkpoint PSNR target.

The next high-probability mechanism is `A_4kd28`: move route conditioning earlier, before the final RGB residual:

- route-conditioned feature expert inserted on `decoder_feature`;
- or route-conditioned Gaussian/control branch;
- still keep fixed `input_std` prior and branch supervision;
- preserve natural sampling;
- report the same per-regime/A18-alignment diagnostics.
