# Result Analysis: V1/T01/A_am03

Status: completed and rejected as a mainline breakthrough.

## Question

`A_am03` tested whether the structured role router becomes more selective when decoder, feature, and edge roles share an explicit competitive softmax budget. The intended mechanism was to keep the `A_aj03` structured-control signal but make role allocation sharper through:

```text
gaussian_control_router_use_competitive_role_gate = true
gaussian_control_router_role_gate_temperature = 0.75
```

## Completion

- Training finished cleanly at `step=2500`.
- Training log: `experiments/V1/runs/A_am03/logs/train_20260519_145711.log`.
- Final train metric: `loss_total=0.030509477481245995`.
- `step_002500.pt`, `latest.pt`, and `final.pt` have identical model weights.

## Validation Trajectory

| step | val_loss | val_psnr |
| ---: | ---: | ---: |
| 500 | 0.02533218963071704 | 28.450075673367785 |
| 1000 | 0.025883095990866422 | 28.207115233777547 |
| 1500 | 0.023843193985521795 | 28.602079030904918 |
| 2000 | 0.0239897514693439 | 28.587588177783367 |
| 2500 | 0.02499520471319556 | 28.439035481471315 |

The internal validation peak occurs at `step=1500`, then softens by the final checkpoint. This looked like a possible schedule or early-stop issue, so the best-validation checkpoint was tested separately on the official test split as a diagnostic.

## Official Full-Image Test

Accepted final checkpoint:

```text
checkpoint = experiments/V1/runs/A_am03/checkpoints/final.pt
split = test
samples = 300
crop_size = 0
device = cuda
loss_mean = 0.03711430704065909
psnr_mean = 25.169950648899793
```

Best-validation diagnostic checkpoint:

```text
checkpoint = experiments/V1/runs/A_am03/checkpoints/step_001500.pt
split = test
samples = 300
crop_size = 0
device = cuda
loss_mean = 0.03717060950274269
psnr_mean = 25.105170630019582
```

## Comparison

| run | official test PSNR | delta vs A_am03 final |
| --- | ---: | ---: |
| `A_am03 final.pt` | 25.169950648899793 | 0 |
| `A_am03 step_001500.pt` | 25.105170630019582 | -0.06478001888021012 |
| `A_aj03 final.pt` | 25.2308542283191 | -0.06090357941930691 |
| `A_ai03 final.pt` | 25.267052413067724 | -0.09710176416793104 |
| `A_ah03 final.pt` | 25.345361164304613 | -0.17541051540482044 |
| `A_z03 final.pt` | 25.385710820365173 | -0.21576017146538007 |
| `M04/A_y03 final.pt` | 26.272695829299355 | -1.1027451803995625 |

## Interpretation

The competitive role gate is trainable but does not solve the core bottleneck. It underperforms the non-competitive structured router `A_aj03`, the decoder-detail single-gate diagnostic `A_ai03`, the best control-gated edge branch `A_ah03`, and the short leader `A_z03`.

The `step_001500.pt` diagnostic is especially important: the best internal validation point is worse than final on the official test split. Therefore the problem is not simply that the final checkpoint overtrained past a good model. The validation peak is not a transferable early-stop signal.

The root bottleneck after `A_am03` is more specific than "router not trainable": the router can train, but hard competition across roles does not produce stable cross-split restoration behavior. It likely suppresses complementary role cooperation too early or too globally, so decoder/detail, feature correction, and edge correction stop acting as coordinated helpers and become a brittle allocation game.

## Decision

Do not promote `A_am03`. Do not replay it unchanged. Do not spend the next mainline on temperature-only tuning of the same competitive gate.

The next useful mainline should keep the structured-control lesson but replace hard role competition with a more stable mechanism, such as stage-aware soft routing, residual role priors, delayed competition, or a controller that constrains role deltas without forcing decoder/feature/edge roles to compete at every spatial location from the start.
