# Retrospective: A37-A41 High-Route Bottleneck

时间: 2026-06-16 13:34:36 CST

## Executive Summary

这几轮不是“没有突破”。A37 已经是明确的 aggregate breakthrough：它把 A33/A36 的专家互补信号压进一个更紧凑的 A33-like 单模型路径，并在当前 100-sample fast-test 上超过 A18/A36 router line。

真正没有突破的是更窄的目标：high-`input_std` / A08-tail fidelity。A37 已经把 high-route gap 从 A33/A35 约 `-0.1267 dB` 推到 `-0.11159004428502844 dB`，但后续 A39/A40/A41 没有继续向 `>-0.10 dB` 推进，反而稳定落在约 `-0.118 dB`。这说明我们现在卡住的不是路由信号、不是真假 teacher、不只是训练曝光，而是压缩/共享结构无法同时保住 A08-tail 与 A15/low-side 红利。

## What We Know

### A36/A37 已经回答的问题

A36 证明 fixed `input_std` router 与 A08/A15 互补性是真实的。只要 checkpoint 内部保留完整双 backbone route capacity，就能基本复现 A18 router：

- A36 test PSNR: `17.8470899392002`
- A18 record router: `17.84693437581442`
- A36 high-route gap: `-0.00019952460737376242 dB`
- A36 low-route gap: `+0.00042343678412253124 dB`

A37 进一步证明这条路可以被部分压缩到共享模型：

- A37 test PSNR: `17.889658453438816`
- A37 overall gap vs A18 router: `+0.042724077624394126 dB`
- A37 low-route gap: `+0.1591364853806252 dB`
- A37 high-route gap: `-0.11159004428502844 dB`
- A37 selected low delta: `0.0`

所以项目不是停在 A18/A36 的 two-backbone diagnostic。A37 是真正的 compressed/shared-backbone aggregate milestone。

### A39/A40/A41 新暴露的问题

后续三个 run 的共同特点是：aggregate/low-side 继续好看，但 high-route 不进反退。

| Run | Test PSNR | Overall gap | High-route gap | Low-route gap | A08-oracle gap | A15-oracle gap |
|---|---:|---:|---:|---:|---:|---:|
| A37 | 17.889658453438816 | +0.042724077624394126 | -0.11159004428502844 | +0.1591364853806252 | -0.19984098726138502 | +0.2493535773419097 |
| A39 | 17.900369585886775 | +0.053435210072354966 | -0.11858325626995542 | +0.18320352678672946 | -0.23903555195244477 | +0.3025769703157029 |
| A40 | 17.900562951956445 | +0.0536285761420247 | -0.1180115677202543 | +0.18311149168725271 | -0.23833660113789967 | +0.3023396530841825 |
| A41 | 17.900201657598004 | +0.05326728178358334 | -0.11813195131188482 | +0.182568457627533 | -0.23811165869770295 | +0.3014789718231976 |

关键读法：

- A40 是 aggregate numeric best，但 high-route 不如 A37。
- A39/A40/A41 比 A37 多出来的约 `+0.0105 dB` aggregate gain，主要来自 low/A15 side。
- A08-oracle group 在 A39/A40/A41 全部比 A37 更差约 `0.038 dB`。
- 这不是单个 run 的随机波动，而是三种不同干预都指向同一个 failure mode。

## Why A39/A40/A41 Did Not Break Through

### 1. Shared capacity helps aggregate, but it can dilute high-route fidelity

A39 让 `gaussian_control_canvas` 可训练，aggregate 提升到 `17.900369585886775`，但 high-route gap 从 A37 `-0.11159004428502844` 退到 `-0.11858325626995542`。这说明共享控制容量会优先服务总体训练分布和 low/A15 side，而不是自动补 A08-tail。

这不是说 control canvas 没用；它对 aggregate 有用。但它不是 high-route 瓶颈的正确杠杆。

### 2. More high-route exposure is insufficient

A40 冻结 `gaussian_control_canvas`，加强 high-`input_std` balanced sampling/crop targeting，结果成为当前 aggregate best：

- A40 test PSNR: `17.900562951956445`
- A40 low-route gap: `+0.18311149168725271`

但 high-route gap 仍是 `-0.1180115677202543`，没有比 A37 好。说明“多喂高样本”不能让当前共享结构自动学出 A08-like representation。

### 3. Teacher L1 变好，不等于 route PSNR 变好

A39/A40/A41 都把 high branch L1 vs A08 teacher 压低了，但 high-route PSNR 没有跟着变好：

| Run | High selected delta | High branch L1 vs A08 | High-route gap |
|---|---:|---:|---:|
| A37 | 0.014578644511138284 | 0.014976846487369648 | -0.11159004428502844 |
| A39 | 0.01329365114919668 | 0.013812246537485789 | -0.11858325626995542 |
| A40 | 0.013238818341389645 | 0.013522875278668349 | -0.1180115677202543 |
| A41 | 0.013213998226578845 | 0.01348984414754912 | -0.11813195131188482 |

这条证据很重要。我们之前自然会认为“branch output 更像 A08 teacher，高路由就该更好”。但数据说不是这样。当前 L1 口径可能压到了低频/平均误差，或者压的是 branch output 本身，却没有让最终 selected route 在关键样本上恢复 A08 的高保真结构。

### 4. Crop-level oracle reweighting 太浅

A41 的机制是合理的：当前 crop 上 A08 比 A15 更接近 GT 时，加大 high-branch A08 teacher loss。它确实让 high branch L1 vs A08 从 A40 `0.013522875278668349` 降到 `0.01348984414754912`。

但变化只有 `0.000033031131119229395`，high-route gap 反而从 A40 `-0.1180115677202543` 到 A41 `-0.11813195131188482`。这说明标量 reweight 不能改变模型表达瓶颈；它只是稍微改了 loss 权重，没有给 high route 新的表示通道。

### 5. Worst cases 揭示了 router-route 与 oracle-route 的错位

A37/A40/A41 的 worst-gap 列表里，很多最差样本按 A18 fixed threshold 被归为 `low_b`，但 `a15_minus_a08` 是负数，表示 A08 实际更强。例如：

- `94_9400006.jpg`: route `low_b`, input_std `0.058188`, `a15_minus_a08=-0.302716`
- `94_94000065.jpg`: route `low_b`, input_std `0.064002`, `a15_minus_a08=-0.310023`
- `94_94000081.jpg`: route `low_b`, input_std `0.058379`, `a15_minus_a08=-0.276238`

这说明 high-tail 问题不只发生在 threshold 以上的 high-route 样本。A08-oracle tail 横跨一部分 nominal low route。仅对 high-`input_std` route 加压，会漏掉这些 “low route by rule, A08 better by oracle” 的 hard cases。

这解释了为什么 A40/A41 的 high-balanced sampling 没有解决 A08-oracle gap。

## The Core Bottleneck

现在的瓶颈可以更精确地表述为：

> 我们已经把 A08/A15 互补性的一部分压进了 A37，但当前共享结构与训练目标会把新增容量优先转化为 low/A15 side 或 aggregate gain；它缺少能在 A08-oracle hard cases 上保持高保真结构的 route-conditioned internal representation。

这句话里有三个关键信息：

- 不是 router signal 不存在。A18/A19/A21/A36 已经证明 signal 稳。
- 不是 teacher 加载或训练工程问题。A37-A41 native teachers、preflight、eval chain 都闭环了。
- 不是再多一点 high samples 或再调 branch loss 权重能解决。A40/A41 已经给出负证据。

## What We Should Stop Doing

1. 不要把 aggregate PSNR 小涨当作 high-route 突破。

A39/A40/A41 都能维持或推高整体分，但 high-route 目标没有动。这类结果要记录，但不能当主线胜利。

2. 不要继续只扫 scalar loss weights。

A41 已经证明更聪明的 scalar reweight 也只能微调 teacher L1。没有新表示容量时，loss 权重很难创造 A08-tail representation。

3. 不要只围绕 fixed high-`input_std` route 采样。

Worst-case 里有不少 `input_std` 低于 threshold、但 A08 比 A15 强的样本。下一步要从 A08-oracle / saved-record target 出发，而不是只从 threshold high route 出发。

4. 不要继续让 shared/base 无约束地吸收 high pressure。

A39/A40 显示 high pressure 很容易变成 low-side/aggregate redirection。需要更强的 per-sample trust anchor 或 route-conditioned capacity，防止高路由训练被共享路径平均掉。

## Recommended Next Experiments

### A42: A08-oracle hard-case target cache

目的：不再只按 `input_std > threshold` 定义 high target，而是从 A18 saved records 找出 A08-oracle/hard negative gap 样本，建立离线 target list/cache。

建议：

- 从 A18 records 选取 `a15_minus_a08 <= 0` 或 student gap 最差样本。
- 覆盖 nominal high route 和 nominal low route 中的 A08-better cases。
- 对这些样本施加 explicit A08 teacher anchor 或 GT+teacher mixed target。
- 验收主指标不只看 overall PSNR，而看 A08-oracle gap 和 high-route gap。

### A43: per-sample trust anchor around A37

目的：A37 是当前 high-route 更好的压缩 anchor。后续训练不应允许 high/A08-oracle hard cases 比 A37 退化。

建议：

- 保存 A37 对 test/val hard-case records 的 outputs 或 PSNR margins。
- 训练时对 hard-case target 设置 hinge/trust loss：新模型不能低于 A37 prediction 或 A18/A36 selected expert 太多。
- 对 low/A15 side 保持当前优势，但不让它通过牺牲 A08-tail 换整体分。

### A44: route-conditioned internal capacity, not late scalar correction

目的：给 A08-tail 一个真实表示通道，而不是继续用 late route head 小 delta。

候选结构：

- route-conditioned decoder feature/refinement branch；
- route-conditioned Gaussian/control branch，但必须按 route 分离或加 per-sample anchor，避免 A39 式 shared-control redirection；
- partial dual-expert blocks only for the last decoder/refinement stages，向 A36 靠近但控制参数量。

验收线：

- high-route gap 必须优于 A37 `-0.11159004428502844`，最好越过 `>-0.10 dB`。
- A08-oracle gap 不应继续退到 A40/A41 的 `~ -0.238 dB`。
- low-route selected delta 仍保持 `0.0` 或可解释的受控值。
- aggregate PSNR 不必刷新 A40 才算科学正例；先把 high-route 打穿更重要。

## Updated Interpretation of Current Best Runs

- A37: 当前最重要的 compressed/shared milestone。Aggregate 已突破，high-route anchor 仍最好。
- A40: 当前 aggregate numeric best。可作为整体分数记录，但不是 high-route anchor。
- A41: 有用的负例和工具性代码。保留默认关闭 oracle weighting，但不要沿这个方向继续小扫。
- A36: 机制上界与路线证明。它告诉我们只要 route-specific capacity 足够，A08/A15 complementarity 可以被保住。

## Bottom Line

接下来真正该打的不是“再涨一点 PSNR”，而是“让 A08-oracle/high-route hard cases 不再被共享压缩吞掉”。A37 已经证明 compressed model 能赢 overall；A39-A41 则证明 overall gain 可以和 high-tail regression 同时发生。

所以下一步的实验设计必须把 promotion criterion 改成 high-tail-first：先看 high-route gap、A08-oracle gap、worst-gap list，再看 aggregate PSNR。否则我们会继续做出看起来涨分、但对核心瓶颈没有帮助的 run。
