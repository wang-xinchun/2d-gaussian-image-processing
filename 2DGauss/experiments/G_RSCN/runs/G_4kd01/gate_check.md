# G_4kd01 Gate Check

- Created: `2026-06-21 16:06:13 CST`
- Status: `SMOKE04 COMPLETE - not promoted; formal training remains blocked`

This file is the launch authority for `G_4kd01`. Until every required item below is marked `PASS`, formal training is not allowed.

## Naming Gate

- PASS: New architecture line uses `experiments/G_RSCN/runs/G_4kd01`.
- PASS: Run ID is `G_4kd01`, not `A_4kd48` or any other A-line ID.
- PASS: Historical alias `G/B_4kd01` is documentation-only; training artifacts must use `G_4kd01`.

## Mechanism Gate

- PASS: Implemented `GaussianRouteControl` default-off module in `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`.
- PASS: Preserved `GaussianParameterHead`.
- PASS: Preserved `TextureAwareGaussianRenderer`.
- PASS: Preserved Gaussian attention/control injection in decoder/refinement/control path.
- PASS: Added route-specific high/low Gaussian-conditioned feature control before RGB residual generation.
- PASS: Added Gaussian-aware boundary router initialized from A18 `input_std` prior.
- PASS: Added exactly one first-stage Gaussian-conditioned route adapter before RGB head.
- PASS: Verified route gate is not the only trainable route mechanism; high/low control paths are trainable.
- PASS: No generic non-Gaussian adapter or non-Gaussian replacement is used as the mainline.

## Equivalence Gate

- PASS: Expected-missing load from A37 best-val.
- PASS: Recorded expected new missing keys: all missing keys are under `gaussian_route_control.*`.
- PASS: Recorded unexpected keys as `[]`.
- PASS: Fixed CPU smoke batch output max/mean absolute difference vs A37.
- Required tolerance before smoke/formal:
  - max absolute output difference: `<= 1e-6` if exact zero-init path is intended;
  - otherwise a written exception must explain the nonzero default behavior.
- PASS: Initial route gate equals A18 input-std prior within clamp tolerance.

Evidence:

- Script: `/home/ldj/wxc/2DGauss/scripts/check_g_4kd01_a37_equivalence.py`.
- Metric JSON: `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/metrics/a37_zero_init_equivalence_cpu.json`.
- Result:
  - `pass=true`;
  - `output_diff.max_abs=0.0`;
  - `gaussian_route_control_delta.max_abs=0.0`;
  - `gaussian_route_control_gate_prior_diff.max_abs=1.0013580322265625e-05`;
  - `unexpected_keys=[]`.

## Parameter Gate

- Reference A37 total params: `168406`.
- Reference A37 trainable params: `76643`.
- PASS: G_4kd01 total params: `206104`.
- PASS: G_4kd01 trainable params: `37698`.
- PASS: Increase vs A37: `37698`.
- Default first-run budget: total params must stay below `1.35x` A37 unless a new decision entry approves an exception.
- PASS: Current total parameter ratio vs A37 is about `1.224x`, within the `1.35x` budget.
- PASS: Trainable prefix list:
  - `gaussian_route_control`
- PASS: Confirmed forbidden first-stage prefixes are not trainable:
  - `input_std_route_decoder_heads.gate_head` alone;
  - terminal/output/post adapters alone;
  - `gaussian_parameter_head` replacement;
  - `TextureAwareGaussianRenderer` replacement.

Parameter policy evidence:

- `total_parameter_count=206104`;
- `trainable_parameter_count=37698`;
- `frozen_parameter_count=168406`;
- `trainable_parameter_prefixes=["gaussian_route_control"]`;
- A37 route decoder heads remain enabled as a frozen output anchor.

## Data And Evaluation Gate

- PASS: Config uses the existing `4KDehaze_local_90_5_5_fast_probe_512` view for first fast-probe comparability.
- PASS: Evaluation plan requires A37/A18/A36/A40/A46/A47 alignment diagnostics.
- PASS: Defined hard-positive and hard-negative list in `/home/ldj/wxc/2DGauss/experiments/G_RSCN/runs/G_4kd01/hard_case_validation_list.md`.
- PASS: Included nominal-low/A08-oracle files:
  - `94_9400006.jpg`;
  - `94_94000065.jpg`;
  - `94_94000081.jpg`.

## GPU Gate

- PASS: Approved physical GPUs are only `0` and `3`.
- PASS: Default physical GPU is `3`.
- PASS: GPU1 and GPU2 are forbidden for this project.
- PASS: Multi-GPU visibility is forbidden for first G_4kd01 training.
- PASS: `CUDA_VISIBLE_DEVICES` must contain exactly one approved physical GPU.
- PASS: Default launch requires `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- PASS: The launch template refuses formal training unless explicit confirmation variables are set.
- PASS: Fresh `nvidia-smi` preflight at `2026-06-21 16:43 CST` selected physical GPU3:
  - GPU3 free memory: `46580MiB`;
  - GPU3 utilization: `15%`;
  - GPU0 avoided because utilization was `91%`;
  - GPU1/GPU2 remain forbidden.
- Required first-run GPU policy:
  - `TWODGAUSS_PHYSICAL_GPU` in `{0,3}`;
  - `TWODGAUSS_MIN_FREE_MIB >= 8192` for smoke/preflight;
  - `TWODGAUSS_MAX_GPU_UTIL <= 20` by default for formal training;
  - `TWODGAUSS_CUDA_MEMORY_FRACTION <= 0.35` unless a smoke calibration justifies more;
  - `num_workers <= 1` for the first G_4kd01 probe;
  - no DDP and no `CUDA_VISIBLE_DEVICES=0,3` for G_4kd01 first run.

## Current Decision

`G_4kd01` is implemented and CPU-equivalence checked. `G_4kd01_smoke01` is approved as a short single-GPU smoke run on physical GPU3 only.

Smoke01 completion:

```text
run id = G_4kd01_smoke01
exit code = 0
final step = 20
best step = 20
best val_psnr = 16.847573189834453
best val_loss = 0.1390131477576991
peak cuda reserved = 2.80859375 GiB
checkpoint = experiments/G_RSCN/runs/G_4kd01_smoke01/checkpoints/best_val.pt
```

Smoke01 A37-aligned evaluation:

```text
eval exit code = 0
test samples = 100
test psnr = 17.8901531659889
test ssim = 0.7027058997750282
overall router gap = +0.04321879017447685
high-route router gap = -0.11152356441357741
low-route router gap = +0.15995425065318444
A08-oracle router gap = -0.19986915378072753
A15-oracle router gap = +0.2502937053955769
```

Interpretation: smoke01 is a valid CUDA mechanism probe with a tiny aggregate/low/A15 and high-route uptick vs A37, but it is not a formal breakthrough because A08-oracle is essentially flat and slightly below A37.

Do not run formal training until smoke results are analyzed and this file is updated again:

```text
mechanism gate = PASS
equivalence gate = PASS
parameter gate = PASS
data/evaluation gate = PASS
GPU gate = PASS
```

Smoke02 completion:

```text
run id = G_4kd01_smoke02
config = experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke02_a08_hard_positive_short.toml
physical GPU = 3
max_steps = 60
batch_size = 1
grad_accum = 2
max_val_batches = 24
trainable_parameter_prefixes = ["gaussian_route_control"]
parent checkpoint = experiments/G_RSCN/runs/G_4kd01_smoke01/checkpoints/best_val.pt
A08 oracle anchor weight = 0.10
A08 oracle anchor require positive advantage = true
A46 hard-positive cache = experiments/V1/runs/A_4kd46/metrics/input_std_route_oracle_hard_cases_train_limit128.json
hard-case sample probability = 0.65
train exit code = 0
final step = 60
best step = 10
best val_psnr = 16.84425906344786
best val_loss = 0.05294540027777354
peak cuda reserved = 3.0546875 GiB
checkpoint best = experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/best_val.pt
checkpoint final = experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/final.pt
```

Smoke02 A37-aligned evaluation:

```text
best checkpoint:
  test PSNR = 17.889345890095033
  test SSIM = 0.7027057281136513
  overall router gap = +0.0424115142806121
  high-route router gap = -0.11139881941583521
  low-route router gap = +0.1584438712796864
  A08-oracle router gap = -0.1986544551187663
  A15-oracle router gap = +0.24776400673193444

final checkpoint:
  test PSNR = 17.887252440269894
  test SSIM = 0.7026950472593307
  overall router gap = +0.04031806445547233
  high-route router gap = -0.11118622204609782
  low-route router gap = +0.15461077181630595
  A08-oracle router gap = -0.1961123874526143
  A15-oracle router gap = +0.24172178274754613
```

Interpretation: smoke02 is a useful partial signal, not a promotion. The strict-positive A08/hard-positive pressure can move high/A08 in the correct direction, but the movement spends low/A15 surplus. Do not launch formal training from this recipe.

Approved next:

```text
run id = G_4kd01_smoke03
config = experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke03_a08_hard_positive_trust_anchor_short.toml
physical GPU = 3
max_steps = 40
batch_size = 1
grad_accum = 2
max_val_batches = 24
trainable_parameter_prefixes = ["gaussian_route_control"]
parent checkpoint = experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/best_val.pt
A08 oracle anchor weight = 0.08
trust anchor weight = 0.06
trust anchor checkpoint = experiments/G_RSCN/runs/G_4kd01_smoke01/checkpoints/best_val.pt
mixed hard-case cache = experiments/G_RSCN/runs/G_4kd01/metrics/input_std_route_mixed_hard_cases_smoke03_train_limit128.json
mixed hard-case samples = 20 A08-positive + 20 A15-protective
hard-case sample probability = 0.60
```

Smoke03 promotion check:

```text
must improve A08-oracle vs A37
must keep high-route at or above A37
must keep low-route at or above A37
must keep A15-oracle at or above A37
aggregate PSNR alone cannot promote
```

Smoke03 completion and decision:

```text
run id = G_4kd01_smoke03
train exit code = 0
final step = 40
best step = 10
best val_psnr = 16.844470528732668
best val_loss = 0.06827673812707265
peak cuda reserved = 3.03515625 GiB
checkpoint best = experiments/G_RSCN/runs/G_4kd01_smoke03/checkpoints/best_val.pt
checkpoint final = experiments/G_RSCN/runs/G_4kd01_smoke03/checkpoints/final.pt
```

Smoke03 A37-aligned evaluation:

```text
best checkpoint:
  test PSNR = 17.888992833916543
  test SSIM = 0.7026935306191444
  overall router gap = +0.04205845810212274
  high-route router gap = -0.11145541481812737
  low-route router gap = +0.15786716925248687
  A08-oracle router gap = -0.19887362622538687
  A15-oracle router gap = +0.24729690030703835

final checkpoint:
  test PSNR = 17.888069211342366
  test SSIM = 0.7026983806490898
  overall router gap = +0.041134835527945786
  high-route router gap = -0.11131750483667309
  low-route router gap = +0.15614274141704423
  A08-oracle router gap = -0.19767744286230152
  A15-oracle router gap = +0.24456751711963792
```

Interpretation: smoke03 is not promotable. It keeps the smoke02 high/A08 movement but still gives back low-route and A15-oracle surplus. The smoke01 trust anchor activated during training, but it only slowed the regression; it did not satisfy the promotion gate. Do not launch formal training from smoke03 and do not simply lengthen this recipe.

Smoke04 approval:

```text
run id = G_4kd01_smoke04
config = experiments/G_RSCN/runs/G_4kd01/configs/g_4kd01_smoke04_a08_a15_paired_anchor_short.toml
physical GPU = 3
max_steps = 40
batch_size = 1
grad_accum = 2
max_val_batches = 24
trainable_parameter_prefixes = ["gaussian_route_control"]
parent checkpoint = experiments/G_RSCN/runs/G_4kd01_smoke02/checkpoints/best_val.pt
A08 oracle anchor weight = 0.08
A15 oracle anchor weight = 0.08
trust anchor weight = 0.0
mixed hard-case cache = experiments/G_RSCN/runs/G_4kd01/metrics/input_std_route_mixed_hard_cases_smoke03_train_limit128.json
hard-case sample probability = 0.60
```

Smoke04 CUDA preflight:

```text
preflight = ok
device = cuda
trainable_parameter_prefixes = ["gaussian_route_control"]
trainable params = 37698
total params = 206104
init checkpoint = smoke02 best, strict load with no missing/unexpected keys
A15 anchor active on sampled crop: probability_mean = 0.7696662545204163
A15 anchor weighted loss = 0.0007020853227004409
A08 anchor inactive on sampled crop because A15 teacher is better there
```

Smoke04 is a short diagnostic run only. It is promotable only if it keeps A08/high movement while restoring low-route and A15-oracle to at least A37.

Smoke04 completion and decision:

```text
run id = G_4kd01_smoke04
train exit code = 0
final step = 40
best step = 10
best val_psnr = 16.84318164737956
best val_loss = 0.06970801282053192
peak cuda reserved = 3.0546875 GiB
checkpoint best = experiments/G_RSCN/runs/G_4kd01_smoke04/checkpoints/best_val.pt
checkpoint final = experiments/G_RSCN/runs/G_4kd01_smoke04/checkpoints/final.pt
A08 anchor active steps = 13/40
A15 anchor active steps = 27/40
```

Smoke04 A37-aligned evaluation:

```text
best checkpoint:
  test PSNR = 17.888755382824197
  test SSIM = 0.7027023512125016
  overall router gap = +0.04182100700977646
  high-route router gap = -0.1115325943744442
  low-route router gap = +0.157508811562785
  A08-oracle router gap = -0.19859082195515507
  A15-oracle router gap = +0.24661626872064404

final checkpoint:
  test PSNR = 17.887806965601595
  test SSIM = 0.7026802280545235
  overall router gap = +0.040872589787172856
  high-route router gap = -0.11162197987872982
  low-route router gap = +0.15591235286846786
  A08-oracle router gap = -0.19814425160702606
  A15-oracle router gap = +0.24447952875260157
```

Interpretation: smoke04 is not promotable. The paired A08/A15 strict-positive teacher anchors both activated, and A08-oracle improved vs A37, but low-route and A15-oracle still regressed below A37. Final also gives back the high-route gate. Do not launch formal training from smoke04 and do not repeat scalar paired-anchor tuning as the next bottleneck step.

Still forbidden:

```text
formal long training
multi-GPU/DDP
physical GPU1/GPU2
CUDA_VISIBLE_DEVICES=0,3
CPU fallback for smoke04 or the next G-line probe
promoting aggregate-only or high/A08-only movement
formal training from smoke03 or any same-recipe extension
formal training from smoke04 or any same scalar-anchor extension
```
