# G_4kd01 Result Analysis

- Created: `2026-06-21 16:06:13 CST`
- Status: `G_4kd01_smoke03` evaluated; formal training remains blocked.

## Current Outcome

`G_4kd01` has completed three short GPU3 smoke probes:

- `G_4kd01_smoke01`: valid CUDA mechanism probe, tiny aggregate/low/A15 gain, no A08 breakthrough.
- `G_4kd01_smoke02`: strict-positive A08/hard-case pressure moved high/A08 but spent low/A15.
- `G_4kd01_smoke03`: added mixed hard cases and smoke01 trust anchor, but still spent low/A15.

Formal long training remains blocked. The next run must change the low/A15 preservation mechanism instead of lengthening smoke03.

## 2026-06-21 CPU Gate Result

The first implementation gate passed on CPU:

- `GaussianRouteControl` is implemented as pre-RGB Gaussian-conditioned route-specific feature control.
- A37 best-val loads into `G_4kd01` with only expected `gaussian_route_control.*` missing keys.
- Unexpected keys are `[]`.
- Zero-init output equivalence vs A37 passed:
  - `output_diff.max_abs=0.0`;
  - `output_diff.mean_abs=0.0`;
  - `gaussian_route_control_delta.max_abs=0.0`.
- Parameter policy is clean:
  - total params: `206104`;
  - trainable params: `37698`;
  - frozen params: `168406`;
  - only `gaussian_route_control` is trainable.

This means the new architecture is now safely inserted around A37 without disturbing the initial function. It still has no training result.

## G_4kd01_smoke01 Result

- Status: `success`
- Run ID: `G_4kd01_smoke01`
- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke01_grscn_gpu_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/launch_smoke01_gpu3.sh`
- Log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/smoke01_20260621_164709.log`
- Exit code: `0`
- Scope:
  - physical GPU3 only;
  - `max_steps=20`;
  - `batch_size=1`;
  - `grad_accum=4`;
  - `max_val_batches=24`;
  - trainable prefix only `gaussian_route_control`;
  - A37 best-val initialization.
- GPU gate:
  - GPU3 free memory `46580MiB`;
  - launch-time GPU3 utilization `0%`;
  - in-run GPU3 memory used about `5216MiB`;
  - peak CUDA reserved `2.80859375GiB`;
  - GPU0 avoided at `91%` utilization;
  - GPU1/GPU2 forbidden.
- Training result:
  - final step `20`;
  - best step `20`;
  - best val PSNR `16.847573189834453`;
  - best val loss `0.1390131477576991`;
  - checkpoint `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke01/checkpoints/best_val.pt`.
- Trainable policy:
  - total params `206104`;
  - trainable params `37698`;
  - frozen params `168406`;
  - trainable prefix only `gaussian_route_control`.
- Mechanism signal:
  - first train `train_route_high_delta_abs_mean=0.011202372610569`;
  - final train `train_route_high_delta_abs_mean=0.010699759237468243`;
  - final val `val_route_high_delta_abs_mean=0.017351550166495144`;
  - this confirms the new Gaussian route-control path was active during CUDA training.

## Smoke01 Interpretation

`G_4kd01_smoke01` is an execution and mechanism-success result, not a promotion result. The absolute validation PSNR is not comparable to full A37/A40 runs because this probe used only `20` steps and `24` validation batches.

## G_4kd01_smoke02 Deployment Plan

- Status: `approved for GPU3 short training`
- Run ID: `G_4kd01_smoke02`
- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke02_a08_hard_positive_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/launch_smoke02_gpu3.sh`
- Evaluation launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/evaluate_smoke02_gpu3.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke01/checkpoints/best_val.pt`
- Scope:
  - physical GPU3 only;
  - `max_steps=60`;
  - `batch_size=1`;
  - `grad_accum=2`;
  - `max_val_batches=24`;
  - trainable prefix only `gaussian_route_control`.
- Bottleneck target:
  - add strict-positive A08 oracle anchor with `input_std_route_a08_oracle_anchor_weight=0.10`;
  - reuse A46 low-route A08-positive hard-case cache;
  - sample hard positives with probability `0.65`;
  - keep low/A15 protection through main reconstruction and low-demand identity loss.

Promotion remains blocked unless A37-aligned evaluation shows high/A08 movement without repeating A46 low/A15 collapse.

## G_4kd01_smoke02 Training Result

- Status: `success`
- Run ID: `G_4kd01_smoke02`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/smoke02_20260621_173307.log`
- Exit code: `0`
- Final step: `60`
- Best step: `10`
- Best val PSNR: `16.84425906344786`
- Best val loss: `0.05294540027777354`
- Peak CUDA reserved: `3.0546875GiB`
- Trainable policy: total params `206104`, trainable params `37698`, trainable prefix only `gaussian_route_control`.
- A08 anchor activation: `19/60` train steps had positive strict A08-anchor probability; active probability mean `0.6627349257469177`; active weighted anchor loss mean `0.0011802482829233142`.
- Sampler: A46 hard-positive cache loaded with `20` usable hard cases and `hard_case_fraction=0.65`.

## G_4kd01_smoke02 A37-Aligned Evaluation

- Best-checkpoint eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/eval_smoke02_20260621_173745.log`
- Final-checkpoint eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/eval_smoke02_final_20260621_174341.log`
- Exit code: `0` for both evaluations.
- Metrics:
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/metrics/test_task_metrics_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/metrics/a18_record_alignment_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/metrics/input_std_route_regime_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/metrics/test_task_metrics_final_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/metrics/a18_record_alignment_final_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/metrics/input_std_route_regime_final_gpu.json`

Compared with A37 and smoke01:

| Metric | A37 | smoke01 best | smoke02 best | smoke02 final | smoke02 best delta vs A37 | smoke02 final delta vs A37 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test PSNR | `17.889658453438816` | `17.8901531659889` | `17.889345890095033` | `17.887252440269894` | `-0.000312563343783` | `-0.002406013168923` |
| test SSIM | `0.7026848953962326` | `0.7027058997750282` | `0.7027057281136513` | `0.7026950472593307` | `+0.000020832717419` | `+0.000010151863098` |
| overall router gap | `+0.042724077624394126` | `+0.04321879017447685` | `+0.0424115142806121` | `+0.04031806445547233` | `-0.000312563343782` | `-0.002406013168922` |
| high-route router gap | `-0.11159004428502844` | `-0.11152356441357741` | `-0.11139881941583521` | `-0.11118622204609782` | `+0.000191224869193` | `+0.000403822238931` |
| low-route router gap | `+0.1591364853806252` | `+0.15995425065318444` | `+0.1584438712796864` | `+0.15461077181630595` | `-0.000692614100939` | `-0.004525713564319` |
| A08-oracle router gap | `-0.19984098726138502` | `-0.19986915378072753` | `-0.1986544551187663` | `-0.1961123874526143` | `+0.001186532142619` | `+0.003728599808771` |
| A15-oracle router gap | `+0.2493535773419097` | `+0.2502937053955769` | `+0.24776400673193444` | `+0.24172178274754613` | `-0.001589570609975` | `-0.007631794594364` |

Required hard-case task metrics:

| File | smoke01 PSNR | smoke02 best PSNR | smoke02 final PSNR |
| --- | ---: | ---: | ---: |
| `94_9400006.jpg` | `18.145732951997076` | `18.15056230711764` | `18.16204581286837` |
| `94_94000065.jpg` | `19.114467118712138` | `19.119427433403928` | `19.130995877250605` |
| `94_94000081.jpg` | `18.857186970213093` | `18.86163836046725` | `18.87214961247682` |

Route-regime diagnostics stayed anchored to the frozen A37 route prior:

- smoke02 best selected delta abs mean: `0.006268110405653715`;
- smoke02 best high delta abs mean: `0.01730584815144539`;
- smoke02 best low delta abs mean: `0.0`;
- smoke02 final selected delta abs mean: `0.006267857467755675`;
- smoke02 final high delta abs mean: `0.017301723705604673`;
- smoke02 final low delta abs mean: `0.0`.

Interpretation:

- Positive: strict-positive A08/hard-case pressure produced the first clear G-line movement on high-route and A08-oracle. The final checkpoint improves A08-oracle by `+0.003728599808771` vs A37.
- Negative: the same pressure spends low-route and A15-oracle surplus; final low-route is `-0.004525713564319` vs A37 and final A15-oracle is `-0.007631794594364` vs A37.
- Decision: do not promote and do not run formal long training from smoke02. The next probe must preserve the A08/hard-positive signal while adding a low/A15 preservation mechanism or a checkpoint-selection metric that penalizes low/A15 loss.

## G_4kd01_smoke03 Deployment Plan

- Status: `approved for GPU3 short training`
- Run ID: `G_4kd01_smoke03`
- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke03_a08_hard_positive_trust_anchor_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/launch_smoke03_gpu3.sh`
- Evaluation launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/evaluate_smoke03_gpu3.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/best_val.pt`
- Scope:
  - physical GPU3 only;
  - `max_steps=40`;
  - `batch_size=1`;
  - `grad_accum=2`;
  - `max_val_batches=24`;
  - trainable prefix only `gaussian_route_control`.
- Bottleneck target:
  - keep strict-positive A08 oracle anchor but reduce weight from `0.10` to `0.08`;
  - add smoke01 best as a strict-positive trust anchor with weight `0.06`;
  - use a mixed low-route hard-case cache with `20` A08-positive and `20` A15-protective samples;
  - preserve low/A15 as a first-class gate, not a post-hoc note.

Smoke03 is promotable only if it keeps the A08/high gain while restoring low-route and A15-oracle to at least A37.

## G_4kd01_smoke03 Training Result

- Status: `success`
- Run ID: `G_4kd01_smoke03`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/smoke03_20260621_180303.log`
- Exit code: `0`
- Final step: `40`
- Best step: `10`
- Best val PSNR: `16.844470528732668`
- Best val loss: `0.06827673812707265`
- Peak CUDA reserved: `3.03515625GiB`
- Trainable policy: total params `206104`, trainable params `37698`, trainable prefix only `gaussian_route_control`.
- A08 anchor activation: `12/40` train steps had positive strict A08-anchor probability; all-step probability mean `0.18741101771593094`; max probability `0.7209430932998657`.
- Trust-anchor activation: `21/40` train steps had positive smoke01-trust probability; all-step probability mean `0.2661998376250267`; max probability `0.5161881446838379`.
- Summary JSON: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke03/metrics/smoke03_a37_aligned_summary.json`.

## G_4kd01_smoke03 A37-Aligned Evaluation

- Best-checkpoint eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/eval_smoke03_best_val_20260621_180850.log`
- Final-checkpoint eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/eval_smoke03_final_20260621_181213.log`
- Exit code: `0` for both evaluations.
- Metrics:
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke03/metrics/test_task_metrics_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke03/metrics/a18_record_alignment_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke03/metrics/input_std_route_regime_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke03/metrics/test_task_metrics_final_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke03/metrics/a18_record_alignment_final_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke03/metrics/input_std_route_regime_final_gpu.json`

Compared with A37, smoke01, and smoke02:

| Metric | A37 | smoke01 best | smoke02 best | smoke03 best | smoke03 final | smoke03 best delta vs A37 | smoke03 final delta vs A37 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test PSNR | `17.889658453438816` | `17.8901531659889` | `17.889345890095033` | `17.888992833916543` | `17.888069211342366` | `-0.000665619522273` | `-0.001589242096451` |
| test SSIM | `0.7026848953962326` | `0.7027058997750282` | `0.7027057281136513` | `0.7026935306191444` | `0.7026983806490898` | `+0.000008635222912` | `+0.000013485252857` |
| overall router gap | `+0.042724077624394126` | `+0.04321879017447685` | `+0.0424115142806121` | `+0.04205845810212274` | `+0.041134835527945786` | `-0.000665619522271` | `-0.001589242096448` |
| high-route router gap | `-0.11159004428502844` | `-0.11152356441357741` | `-0.11139881941583521` | `-0.11145541481812737` | `-0.11131750483667309` | `+0.000134629466901` | `+0.000272539448355` |
| low-route router gap | `+0.1591364853806252` | `+0.15995425065318444` | `+0.1584438712796864` | `+0.15786716925248687` | `+0.15614274141704423` | `-0.001269316128138` | `-0.002993743963581` |
| A08-oracle router gap | `-0.19984098726138502` | `-0.19986915378072753` | `-0.1986544551187663` | `-0.19887362622538687` | `-0.19767744286230152` | `+0.000967361035998` | `+0.002163544399084` |
| A15-oracle router gap | `+0.2493535773419097` | `+0.2502937053955769` | `+0.24776400673193444` | `+0.24729690030703835` | `+0.24456751711963792` | `-0.002056677034871` | `-0.004786060222272` |

Route-regime diagnostics:

- smoke03 best selected delta abs mean: `0.006268389886245131`;
- smoke03 best high delta abs mean: `0.017305808244273066`;
- smoke03 best low delta abs mean: `0.0`;
- smoke03 final selected delta abs mean: `0.006268032351508737`;
- smoke03 final high delta abs mean: `0.0173039114754647`;
- smoke03 final low delta abs mean: `0.0`.

Interpretation:

- Positive: smoke03 preserves the smoke02 observation that the G-RSCN route-control path can move high-route and A08-oracle in the desired direction.
- Positive: the smoke01 trust anchor did activate (`21/40` steps), so the protection plumbing is real rather than dead code.
- Negative: the protection is too weak. Best checkpoint already loses low-route by `-0.001269316128138` and A15-oracle by `-0.002056677034871` vs A37; final loses low-route by `-0.002993743963581` and A15-oracle by `-0.004786060222272`.
- Negative: aggregate PSNR also drops below A37, while high/A08 gains remain small.
- Decision: do not promote and do not run formal long training from smoke03. The bottleneck is now narrower: high/A08 pressure works, but low/A15 preservation must be enforced more directly than a weak global trust anchor.

Next required change before another short run:

- do not simply lengthen smoke03 or increase `max_steps`;
- add a paired hard-positive/hard-negative objective or sampler where low/A15-protective samples receive direct preservation pressure, not only conditional smoke01 trust;
- add checkpoint selection that rejects A15-oracle or low-route regression before selecting by val PSNR;
- keep the run as `G_4kd01_smoke04` or move to `G_4kd02` only if the mechanism itself changes.

## G_4kd01_smoke04 Deployment Plan

- Status: `approved for GPU3 short training`
- Run ID: `G_4kd01_smoke04`
- Config: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke04_a08_a15_paired_anchor_short.toml`
- Launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/launch_smoke04_gpu3.sh`
- Evaluation launcher: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/evaluate_smoke04_gpu3.sh`
- Parent checkpoint: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/best_val.pt`
- Scope:
  - physical GPU3 only;
  - `max_steps=40`;
  - `batch_size=1`;
  - `grad_accum=2`;
  - `max_val_batches=24`;
  - trainable prefix only `gaussian_route_control`.
- Mechanism change vs smoke03:
  - added a symmetric strict-positive A15 oracle anchor in `/home/ldj/wxc/2DGauss/scripts/train_v1_t01.py`;
  - keep strict-positive A08 oracle anchor weight `0.08`;
  - add strict-positive A15 oracle anchor weight `0.08`;
  - disable smoke01 trust anchor (`0.0`) so the paired teacher anchors give a cleaner signal;
  - keep the mixed hard-case cache from smoke03.
- CUDA preflight:
  - `device=cuda`;
  - trainable prefix only `gaussian_route_control`;
  - A15 anchor active on the sampled low/A15 crop with probability mean `0.7696662545204163`;
  - A08 anchor inactive on that crop, correctly reflecting A15 teacher advantage.

Smoke04 is promotable only if it preserves the smoke02 high/A08 signal while returning low-route and A15-oracle to at least A37.

## G_4kd01_smoke04 Training Result

- Status: `success, not promoted`
- Run ID: `G_4kd01_smoke04`
- Train log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/smoke04_20260621_183609.log`
- Exit code: `0`
- Final step: `40`
- Best step: `10`
- Best val PSNR: `16.84318164737956`
- Best val loss: `0.06970801282053192`
- Peak CUDA reserved: `3.0546875GiB`
- Trainable policy: total params `206104`, trainable params `37698`, trainable prefix only `gaussian_route_control`.
- A08 anchor activation: `13/40` train steps had positive strict A08-anchor probability; all-step probability mean `0.2004557117819786`; max probability `0.7269479632377625`.
- A15 anchor activation: `27/40` train steps had positive strict A15-anchor probability; all-step probability mean `0.5397373855113983`; max probability `0.9935572743415833`.
- Summary JSON: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke04/metrics/smoke04_a37_aligned_summary.json`.

## G_4kd01_smoke04 A37-Aligned Evaluation

- Best-checkpoint eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/eval_smoke04_best_val_20260621_183945.log`
- Final-checkpoint eval log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/eval_smoke04_final_20260621_184352.log`
- Exit code: `0` for both evaluations.
- Metrics:
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke04/metrics/test_task_metrics_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke04/metrics/a18_record_alignment_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke04/metrics/input_std_route_regime_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke04/metrics/test_task_metrics_final_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke04/metrics/a18_record_alignment_final_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke04/metrics/input_std_route_regime_final_gpu.json`

Compared with A37, smoke02, and smoke03:

| Metric | A37 | smoke02 best | smoke03 best | smoke04 best | smoke04 final | smoke04 best delta vs A37 | smoke04 final delta vs A37 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test PSNR | `17.889658453438816` | `17.889345890095033` | `17.888992833916543` | `17.888755382824197` | `17.887806965601595` | `-0.000903070614619` | `-0.001851487837222` |
| test SSIM | `0.7026848953962326` | `0.7027057281136513` | `0.7026935306191444` | `0.7027023512125016` | `0.7026802280545235` | `+0.000017455816269` | `-0.000004667341709` |
| overall router gap | `+0.042724077624394126` | `+0.0424115142806121` | `+0.04205845810212274` | `+0.04182100700977646` | `+0.040872589787172856` | `-0.000903070614618` | `-0.001851487837221` |
| high-route router gap | `-0.11159004428502844` | `-0.11139881941583521` | `-0.11145541481812737` | `-0.1115325943744442` | `-0.11162197987872982` | `+0.000057449910584` | `-0.000031935593701` |
| low-route router gap | `+0.1591364853806252` | `+0.1584438712796864` | `+0.15786716925248687` | `+0.157508811562785` | `+0.15591235286846786` | `-0.001627673817840` | `-0.003224132512157` |
| A08-oracle router gap | `-0.19984098726138502` | `-0.1986544551187663` | `-0.19887362622538687` | `-0.19859082195515507` | `-0.19814425160702606` | `+0.001250165306230` | `+0.001696735654359` |
| A15-oracle router gap | `+0.2493535773419097` | `+0.24776400673193444` | `+0.24729690030703835` | `+0.24661626872064404` | `+0.24447952875260157` | `-0.002737308621266` | `-0.004874048589308` |

Required hard-case task metrics:

| File | smoke04 best PSNR | smoke04 final PSNR |
| --- | ---: | ---: |
| `94_9400006.jpg` | `18.151972861281696` | `18.15618902087743` |
| `94_94000065.jpg` | `19.120726784467465` | `19.124622542908764` |
| `94_94000081.jpg` | `18.862864997416942` | `18.86669996612578` |

Route-regime diagnostics:

- smoke04 best selected delta abs mean: `0.006268117036670446`;
- smoke04 best high delta abs mean: `0.01730520751327276`;
- smoke04 best low delta abs mean: `0.0`;
- smoke04 final selected delta abs mean: `0.006268504643812776`;
- smoke04 final high delta abs mean: `0.017304132459685206`;
- smoke04 final low delta abs mean: `0.0`.

Interpretation:

- Positive: the new A15 strict-positive anchor is live (`27/40` active steps), and the existing A08 anchor remains live (`13/40` active steps).
- Positive: best checkpoint keeps a tiny high-route gain and improves A08-oracle by `+0.001250165306230` vs A37.
- Negative: the paired scalar anchors still spend low/A15 surplus. Best checkpoint loses low-route by `-0.001627673817840` and A15-oracle by `-0.002737308621266` vs A37; final loses low-route by `-0.003224132512157` and A15-oracle by `-0.004874048589308`.
- Negative: final also loses high-route vs A37, so longer same-recipe training is moving away from the promotion boundary.
- Decision: do not promote and do not run formal long training from smoke04. The bottleneck is not anchor activation; it is shared control-channel interference. The next probe should split or constrain route-specific Gaussian control so A08-positive updates cannot overwrite A15-protective behavior.

Next required change before another short run:

- do not simply increase A15 weight or lengthen smoke04;
- add route-conditional gradient isolation, branch-specific control masking, or a paired anti-regression term that compares the current selected output against the smoke02/smoke01 low/A15-preserving output on A15-positive cases;
- make checkpoint selection reject low-route or A15-oracle regression before val PSNR;
- keep GPU policy: physical GPU0/3 only, single visible GPU, memory fraction `0.35`, no CPU fallback.

## G_4kd01_smoke01 A37-Aligned Evaluation

- Status: `success`
- Log: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/logs/eval_smoke01_20260621_170423.log`
- Exit code: `0`
- Metrics:
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke01/metrics/test_task_metrics_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke01/metrics/a18_record_alignment_best_val_gpu.json`
  - `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01_smoke01/metrics/input_std_route_regime_best_val_gpu.json`

Compared with A37 on the same 100-sample 512-crop test evaluation:

| Metric | A37 | G_4kd01_smoke01 | Delta |
| --- | ---: | ---: | ---: |
| test PSNR | `17.889658453438816` | `17.8901531659889` | `+0.000494712550084` |
| test SSIM | `0.7026848953962326` | `0.7027058997750282` | `+0.0000210043787956` |
| overall router gap | `+0.042724077624394126` | `+0.04321879017447685` | `+0.000494712550083` |
| high-route router gap | `-0.11159004428502844` | `-0.11152356441357741` | `+0.000066479871451` |
| low-route router gap | `+0.1591364853806252` | `+0.15995425065318444` | `+0.000817765272559` |
| A08-oracle router gap | `-0.19984098726138502` | `-0.19986915378072753` | `-0.000028166519343` |
| A15-oracle router gap | `+0.2493535773419097` | `+0.2502937053955769` | `+0.000940128053667` |

Hard-case task metrics:

| File | PSNR | SSIM | Loss |
| --- | ---: | ---: | ---: |
| `94_9400006.jpg` | `18.145732951997076` | `0.7848045229911804` | `0.10305830091238022` |
| `94_94000065.jpg` | `19.114467118712138` | `0.7969244718551636` | `0.09197874367237091` |
| `94_94000081.jpg` | `18.857186970213093` | `0.7872453927993774` | `0.09525853395462036` |

Route-regime signal:

- Overall selected delta abs mean: `0.006268339483067393`, effectively matching A37's frozen route-head anchor.
- High-route selected delta abs mean: `0.014577533681552077`.
- Low-route selected delta abs mean: `0.0`.
- Gate/prior means remain equal to the input-std prior, so the run tested pre-RGB Gaussian control more than learned boundary movement.

Interpretation:

- Positive: the CUDA eval is valid; aggregate PSNR/SSIM, high-route, low-route, and A15-oracle moved slightly in the right direction.
- Negative: A08-oracle did not improve; it is essentially flat and slightly worse than A37.
- Decision: do not promote to formal training. The next deployable probe should keep the same G-line mechanism but add a stronger A08/hard-positive pressure or hard-case sampler, still as a short run.

Next deployable step:

- deploy `G_4kd01_smoke02` or `G_4kd02` as another short run, not formal long training;
- keep the G-RSCN pre-RGB Gaussian route-control mechanism;
- add stronger A08/hard-positive pressure or a hard-case sampler;
- keep low/A15 protection as a gate because smoke01's clearest positive movement was low-route/A15 stability.

## Required Analysis After First Valid Run

The first valid result analysis must report:

- test PSNR and SSIM;
- A18 alignment overall/high/low/A08-oracle/A15-oracle gaps;
- route-regime diagnostics;
- hard-positive and hard-negative boundary sample outcomes;
- parameter count vs A37 and A36 reference;
- whether high/A08 improved without low/A15 collapse.

## Promotion Boundary

Aggregate PSNR alone cannot promote this run.

`G_4kd01` is promotable only if it improves high-route and A08-oracle behavior over A37 while preserving low-route and A15-oracle surplus.
