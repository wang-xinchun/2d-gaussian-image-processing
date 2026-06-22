# 当前模型架构与突破实验分析报告

- 时间：`2026-06-16 15:10:54 CST`
- 项目根目录：`/home/ldj/wxc/2DGauss`
- 范围：当前 V1/T01 模型详细架构、A08-A41 突破实验数据、结果分析、瓶颈判断与下一步突破口。
- 当前主线任务：`4KDehaze_local_90_5_5_fast_probe_512` 上的 paired 4K 图像去雾/超高清图像恢复。

## 0. 总结先行

当前项目不是一个纯 2D Gaussian 渲染器。我们的主模型 `CrossResolutionGaussianRestorer` 本质是：

```text
紧凑 CNN 图像恢复 backbone
+ 低分辨率 Gaussian attention/control prior
+ 多个可插拔结构控制、特征修正、路由专家模块
```

最终图像仍然是：

```text
restored = input + RGB residual
```

Gaussian 模块并不直接渲染 RGB 图像。它负责预测结构注意力图、控制图和 affine 调制信号，用来指导 CNN backbone 在哪些区域增强、保细节或修复。

当前核心科学结论已经比较清楚：

- A08 与 A15 不是谁完全优于谁，而是互补。
- A15 更适合一批低 `input_std` / 困难去雾样本。
- A08 更适合保护一批高保真 A08-tail 样本。
- A18/A19/A21 证明固定 `input_std` router 能稳定利用这种互补。
- A36 证明只要 checkpoint 内部有完整双 backbone 容量，A08/A15 的 route 行为几乎可以完整复现。
- A37 证明这种互补可以被部分压进一个更紧凑的共享模型，并且在 aggregate PSNR 上突破 A18/A36 router line。
- A39/A40/A41 进一步证明：整体 PSNR 继续涨，不等于 high-route/A08-tail 瓶颈被突破。

现在真正卡住的点不是“有没有路由信号”，也不是“teacher 能不能加载”，而是：

```text
如何在单个紧凑/共享模型里保住 A08-oracle 高保真 hard cases，
同时不丢掉 A15/low-side 的整体收益。
```

当前定位：

- `A_4kd37`：最重要的 compressed/shared milestone；aggregate 已突破，且仍是 compressed 模型里 high-route 更好的 anchor。
- `A_4kd40`：当前 aggregate numeric best；但不是 high-route 突破。
- `A_4kd36`：机制上界；证明 route-specific full capacity 足够。
- `A_4kd41`：有价值负例；oracle weighting 让 high branch L1 微降，但 high-route PSNR 没改善。

下一步必须 high-tail-first：先看 high-route gap、A08-oracle gap、worst-gap list，再看整体 PSNR。

## 1. 当前模型详细架构

### 1.1 主要代码入口

核心代码路径：

- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/restorer.py`
- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/models/gaussian.py`
- `/home/ldj/wxc/2DGauss/src/uhd_gaussian/config.py`

当前代表性配置：

- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd37/configs/a_4kd37_fast_4kdehaze_a33_high_route_a36_teacher_compress.toml`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd40/configs/a_4kd40_fast_4kdehaze_a37_high_route_balanced_teacher.toml`
- `/home/ldj/wxc/2DGauss/experiments/V1/runs/A_4kd41/configs/a_4kd41_fast_4kdehaze_a37_high_route_oracle_weighted.toml`

主模型类：

```text
CrossResolutionGaussianRestorer
```

### 1.2 总体前向流程

当前模型前向路径可以概括为：

```text
input RGB
  -> pad_to_multiple(lowres_scale)
  -> highres_stem：全分辨率 CNN 特征
  -> optional explicit_frequency_split
  -> pixel_unshuffle：低分辨率分支
  -> lowres_proj + TextureFrequencyExtractor + lowres_fusion
  -> GaussianParameterHead
  -> TextureAwareGaussianRenderer -> 单通道 attention_map
  -> attention_map 调制 highres high-pass feature
  -> LowRankAffineModulator
  -> concat(modulated highres, upsampled lowres)
  -> decoder feature
  -> Gaussian control/guidance/refinement 等可选模块
  -> input_std route decoder heads 等路由模块
  -> RGB residual
  -> restored = input + residual
```

这说明 Gaussian 是结构先验和控制器，不是 RGB 输出端。

### 1.3 基础 CNN 恢复 backbone

全分辨率分支：

- `highres_stem = Conv2d(in_channels, width, 3x3) + StackedResidualBlocks`
- 用于保留原图分辨率下的空间细节和高频结构。

低分辨率分支：

- 使用 `pixel_unshuffle`，当前 `lowres_scale=4`。
- `lowres_proj = Conv2d(in_channels * 16, width, 1x1) + StackedResidualBlocks`
- `TextureFrequencyExtractor` 从低分辨率 RGB 与 feature 中抽取纹理/频率信息。
- `lowres_fusion` 融合低分辨率语义、雾结构和纹理信息。

融合与解码：

```text
modulated highres feature
+ upsampled lowres feature
-> Conv2d(width * 2 -> width, 1x1)
-> StackedResidualBlocks(width, decoder_blocks)
-> Conv2d(width -> out_channels, 3x3)
-> RGB residual
```

### 1.4 Gaussian prior 细节

`GaussianParameterHead` 输出两类内容：

- `params`：每个低分辨率 cell 的 Gaussian-like 参数。
- `coeffs`：给 `LowRankAffineModulator` 使用的低秩 affine 系数。

当前代表性参数：

- `gaussian_atoms = 4`
- `gaussian_param_dim = 10`
- `max_frequency = 4.0`
- `max_texture_gain = 1.0`
- `affine_rank = 4`

`TextureAwareGaussianRenderer` 将 10 维参数解释为 amplitude、位置、sigma、方向、texture gain、frequency、phase、spectral damping 等，渲染成单通道 `attention_map`。随后：

```text
highpass = highres_feature - avg_pool(highres_feature)
modulated = highres_feature + attention_map * highpass
modulated = LowRankAffineModulator(modulated, coeffs)
```

所以 Gaussian 的实际作用是：指导哪些区域需要增强高频、控制结构修复强度、为 decoder feature 提供结构先验。

### 1.5 当前启用的结构控制模块

A37/A40/A41 代表性启用项：

- `use_gaussian_decoder_guidance = true`
- `gaussian_decoder_guidance_strength = 0.08`
- `use_gaussian_feature_refinement = true`
- `gaussian_feature_refinement_strength = 0.08`
- `use_gaussian_control_canvas = true`
- `gaussian_control_canvas_strength = 0.06`
- `gaussian_control_canvas_stage = "pre_guidance"`
- `gaussian_control_canvas_share_to_guidance = true`
- `gaussian_control_canvas_share_to_guidance_blend = 0.25`
- `gaussian_control_canvas_target_mode = "restoration_demand"`
- `gaussian_control_canvas_feedback_mode = "additive"`
- `gaussian_control_canvas_feedback_budget = 0.20`

这些模块都在 shared path 中工作。A39 证明，让 shared `gaussian_control_canvas` 可训练可以提高 aggregate PSNR，但会把收益更多转到 low/A15 side，并没有修复 high-route。

### 1.6 input_std 路由机制

固定路由阈值来自 A18：

```text
threshold = 0.07026463374495506
```

当前语义：

```text
input_std > threshold  -> high route，A08-like target
input_std <= threshold -> low route，A15-like target
```

A37/A40/A41 当前 soft gate：

```text
softness = 0.003
direction = "gt"
```

路由模块演化：

- A24 `InputStdRouteResidualHead`：末端 high/low residual delta，是第一个正向 single-checkpoint 结构转化，但太晚、太小。
- A27 `InputStdRouteTerminalHeads`：独立 high/low terminal heads，工程成立但 high-route 失败。
- A28 `InputStdRouteFeatureExpert`：在 decoder feature 上做 route-conditioned expert，仍未解决 high-route。
- A29-A41 `InputStdRouteDecoderHeads`：当前主力路由结构。
- A35 `InputStdRoutePostAdapter`：输出后 adapter，边际增益极小。

当前关键结构 `InputStdRouteDecoderHeads`：

```text
shared = Conv1x1([decoder_feature, image, prior_map]) + residual blocks

high_feature = decoder_feature + strength * high_feature_expert(shared)
low_feature  = decoder_feature + strength * low_feature_expert(shared)

high_residual = high_head(high_feature)
low_residual  = low_head(low_feature)

residual = gate * high_residual + (1 - gate) * low_residual
```

其中 high/low head 初始化自 base decoder head。A37/A40/A41 锁定 low route，只训练 high/shared late modules。

### 1.7 当前模型规模与训练参数

A37/A40/A41 代表性 backbone 设置：

- `width = 32`
- `lowres_scale = 4`
- `highres_blocks = 2`
- `lowres_blocks = 4`
- `decoder_blocks = 2`
- `affine_rank = 4`
- `gaussian_atoms = 4`
- `gaussian_param_dim = 10`

A37/A40/A41 代表性 trainable prefixes：

```text
input_std_route_decoder_heads.shared
input_std_route_decoder_heads.high_feature
input_std_route_decoder_heads.high_head
gaussian_feature_refinement
gaussian_decoder_guidance
decoder.1
```

A41 preflight 记录：

```text
trainable params = 76643 / 168406
```

这说明当前 compact/compressed run 并不是双完整 backbone，而是在共享模型上用有限 late/shared/high 模块尝试恢复 A08-like high-route 行为。

## 2. 数据与评估口径

当前 active fast-probe view：

- 数据集：`4KDehaze_local_90_5_5_fast_probe_512`
- train：`512`
- val：`100`
- test：`100`
- crop size：`512`

完整 local 4KDehaze split：

- train：`14045`
- val：`780`
- test：`781`

注意：fast-probe 很适合快速机制发现，但不是最终泛化结论。A18/A19/A21/A22 的价值在于用不同 offset/block 验证了 router 信号不是单一小块偶然。

核心评估指标：

- standard fast-test PSNR/SSIM/loss；
- A18 record alignment：
  - overall gap；
  - high-route gap；
  - low-route gap；
  - A08-oracle gap；
  - A15-oracle gap；
- route-regime diagnostics：
  - selected delta；
  - high selected delta；
  - low selected delta；
  - high branch L1 vs A08；
  - low branch L1 vs A15。

## 3. 路由信号证据

### 3.1 A08/A15 互补

基础 checkpoint：

- A08 fast-test PSNR：`17.789133503142946`
- A15 fast-test PSNR：`17.781017099305146`

A15 在不少样本上更好，但会伤害 A08 high-fidelity tail。因此不能简单做全局 finetune、平均或单一路径蒸馏。

### 3.2 A18 fixed router

A18 规则：

```text
input_std <= 0.07026463374495506 -> A15
input_std  > 0.07026463374495506 -> A08
```

结果：

- selected router PSNR：`17.84693437581442`
- choose A15：`57 / 100`
- binary oracle PSNR：`17.883555795086536`

A18 证明 fast-test 上存在稳定可利用的条件互补。

### 3.3 A19/A21 跨块验证

A19 offset100：

- fixed A18 rule：`15.913737915082573`
- A08：`15.87689249278197`
- A15：`15.885454203061169`
- binary oracle：`15.928149885150932`

A21 offset200：

- fixed A18 rule：`14.64236451612018`
- A08：`14.61942131606054`
- A15：`14.570231632564433`
- binary oracle：`14.662330739170043`

结论：固定 `input_std` rule 不是 A18 单块偶然。

### 3.4 A22 router 分析

A22 说明更复杂的自由 router 容易过拟合：

- fixed input-std best：`15.761350330784339`
- oracle agreement：`0.77`
- choose A15：`204 / 400`
- richer logistic/all-common 方案更低。

结论：`input_std` 应作为强先验保留，不应把主线转成自由 learned router。瓶颈是 compact model 能否表达两套行为。

## 4. 关键实验数据总表

### 4.1 标准 fast-test PSNR

说明：A37/A39/A40/A41 的部分 standard eval 文件中 `loss=0.0` 是因为这些训练配置刻意关闭普通 reconstruction/proxy loss，用于 teacher-compression 诊断。A37 另有 standard-loss 复评：`loss_mean=0.11911950781941413`，`total_loss_mean=0.12235063299536705`。

| Run | 核心机制 | Test PSNR | 结论 |
|---|---|---:|---|
| A08 | high-fidelity/detail-protected baseline | 17.789133503142946 | A08-tail 强，但全局不稳 |
| A15 | A08 finetune / better proxy baseline | 17.781017099305146 | low-std 多样本更好，伤害 A08-tail |
| A18 | two-checkpoint fixed input-std router | 17.84693437581442 | 路由信号证明 |
| A23 | whole-output router distillation | 17.78064128762362 | 失败，退回 A15-like |
| A24 | late input-std route residual | 17.790202882742463 | 首个正向 single-checkpoint 结构转化 |
| A27 | route terminal heads | 17.7513865188403 | high route 失败 |
| A28 | feature expert | 17.73849582669033 | high route 失败 |
| A29 | hard decoder heads | 17.76127769536133 | 容量打开但 drift 有害 |
| A30 | trust-region decoder heads | 17.75180602939797 | 抑制 drift，但没产生有效 high route |
| A31 | transplanted route heads | 17.770243709681385 | high-route 改善，low-route 退化 |
| A32 | high A08 transplant + low lock | 17.786780968730323 | low safe，high 不够 |
| A33 | A31 high + A24/A32 low surgery | 17.823062311698262 | 首次超过 17.82 |
| A34 | formal A33 refinement | 17.820851508384713 | 不如 A33 |
| A35 | post-route adapter | 17.823132133493647 | 仅边际提升 |
| A36 | single checkpoint full dual backbone | 17.8470899392002 | 机制上界，容量足够 |
| A37 | A33/A36 native-teacher compression | 17.889658453438816 | compressed/shared aggregate milestone |
| A38 | standard-loss high-head-only tune | 17.889819259555214 | 微小提升，非实质突破 |
| A39 | A37 + trainable control canvas | 17.900369585886775 | aggregate 提升，high route 变差 |
| A40 | A37 + high-balanced sampling | 17.900562951956445 | 当前 aggregate numeric best，但 high route 未破 |
| A41 | A40 + crop-level oracle weighting | 17.900201657598004 | high-route 负例 |

### 4.2 A18 alignment：最关键的瓶颈表

| Run | Overall gap | High-route gap | Low-route gap | A08-oracle gap | A15-oracle gap |
|---|---:|---:|---:|---:|---:|
| A27 | -0.09554785697412059 | -0.3054244593410088 | +0.06278010621493545 | -0.45424864545861654 | +0.2100120739571167 |
| A28 | -0.1084385491240903 | -0.3390800359165398 | +0.06555415143863476 | -0.5188876236246397 | +0.24120325508008136 |
| A29 | -0.08565668045309131 | -0.236088142902285 | +0.027826703499809195 | -0.27720066420366934 | +0.07751041681591958 |
| A30 | -0.0951283464164498 | -0.23814779446593295 | +0.012763517901581372 | -0.27065612700046393 | +0.05439531852548818 |
| A31 | -0.07669066613303403 | -0.14121316850824578 | -0.02801579592015501 | -0.14761475713935954 | -0.016273847868386378 |
| A32 | -0.060153407084098394 | -0.211220889864822 | +0.05380978168030712 | -0.1930536941607739 | +0.053057948573810354 |
| A33 | -0.023872064116158373 | -0.12677254958688497 | +0.053754617905617835 | -0.08633674234915452 | +0.029338587711949447 |
| A34 | -0.026082867429706925 | -0.1309565862555424 | +0.053032394140660195 | -0.08089136622024616 | +0.02060585376223391 |
| A35 | -0.02380224232077156 | -0.12662823419234123 | +0.05376824277532487 | -0.08644686819587206 | +0.029561698239499248 |
| A37 | +0.042724077624394126 | -0.11159004428502844 | +0.1591364853806252 | -0.19984098726138502 | +0.2493535773419097 |
| A38 | +0.04288488374079478 | -0.1111094511726412 | +0.15905604867549208 | -0.19848443524653703 | +0.2484957851003737 |
| A39 | +0.053435210072354966 | -0.11858325626995542 | +0.18320352678672946 | -0.23903555195244477 | +0.3025769703157029 |
| A40 | +0.0536285761420247 | -0.1180115677202543 | +0.18311149168725271 | -0.23833660113789967 | +0.3023396530841825 |
| A41 | +0.05326728178358334 | -0.11813195131188482 | +0.182568457627533 | -0.23811165869770295 | +0.3014789718231976 |

读法：

- A37 是 compressed/shared aggregate milestone。
- A40 是当前 aggregate numeric best。
- A37/A38 在 high-route gap 上仍优于 A39/A40/A41。
- A39/A40/A41 的 aggregate 增益主要来自 low/A15 side；A08-oracle gap 反而明显差于 A37。

### 4.3 Route-regime 诊断

| Run | Selected delta | High selected delta | Low selected delta | High branch L1 vs A08 | Low branch L1 vs A15 |
|---|---:|---:|---:|---:|---:|
| A27 | 0.0009762731264345348 | 0.0007441960665029149 | 0.001151348803225055 | 0.01429054965156802 | 0.01471208993410855 |
| A28 | 0.0011144706892082467 | 0.0008386419774682889 | 0.0013225519980647061 | 0.015561815286271794 | 0.01525487631494016 |
| A29 | 0.00025362832675455136 | 0.0002827254742611373 | 0.0002316778470566006 | 0.262734996371491 | 0.34177368774748684 |
| A30 | 0.0000578112014773069 | 0.0000618680088908065 | 0.0000547508029022107 | 0.29880278470904326 | 0.3636353517833509 |
| A31 | 0.012584832096472383 | 0.019272921736850294 | 0.007539431139696063 | 0.12602153940256253 | 0.39231843226834345 |
| A32 | 0.006058236313983798 | 0.014088921660427437 | 0.0 | 0.23157927844413492 | 0.4520495779681624 |
| A33 | 0.008283325210213662 | 0.019263547000496886 | 0.0 | - | - |
| A34 | 0.008751856992021204 | 0.020353155795398148 | 0.0 | 0.32904680800992386 | 0.2585565087042357 |
| A35 | 0.008263196442276239 | 0.019216735912270324 | 0.0 | 0.3066226146941961 | 0.3575197092273779 |
| A37 | 0.0062688171397894624 | 0.014578644511138284 | 0.0 | 0.014976846487369648 | 0.012233833717018888 |
| A39 | 0.005716269994154572 | 0.01329365114919668 | 0.0 | 0.013812246537485789 | 0.01426901130757311 |
| A40 | 0.0056926918867975474 | 0.013238818341389645 | 0.0 | 0.013522875278668349 | 0.014295937117646661 |
| A41 | 0.005682019237428904 | 0.013213998226578845 | 0.0 | 0.01348984414754912 | 0.014252835606927411 |

关键解释：A39/A40/A41 的 high branch L1 vs A08 比 A37 更低，但 high-route PSNR gap 更差。这说明“更像 A08 teacher 的 L1”不等于“在关键 route/PSNR 上更接近 A18/A36/A08”。

## 5. 各阶段实验结论

### 5.1 A23：whole-output distillation 失败

A23 尝试把 A18 two-checkpoint router 蒸馏进一个 A15-initialized student。

结果：

- best-val test PSNR：`17.78064128762362`
- final checkpoint PSNR：`17.035365402315897`

结论：最终输出蒸馏会平均掉两个条件最优，或者偏回 A15-like 共享路径，不能保存 A08/A15 两套行为。

### 5.2 A24：第一次 single-checkpoint 结构正例

A24 加入 `InputStdRouteResidualHead`。

结果：

- PSNR：`17.790202882742463`
- SSIM：`0.7000000587105751`
- loss：`0.12000417917966842`
- total loss：`0.1232352689653635`

结论：A24 是第一个超过 A08/A15/A23 的 single-checkpoint structural conversion。但 route head 在 shared Gaussian/control/decoder 之后，只是末端小 delta，因此 high-route 仍远低于 A18。

### 5.3 A25/A26：balanced sampling 不是主瓶颈

A25 file-level balanced sampler 因 crop 后 route label 不稳定而失败。

A26 crop-aware balanced sampler 技术上实现 50/50 high/low batch，但验证明显低于 A24：

- A26 早期 val 约：`18.236443073291603`
- A24 val：`18.349088096641417`

结论：不是简单“高路由样本不够”。强行改变自然分布会损害训练。

### 5.4 A27/A28：terminal heads / feature expert 失败

A27：

- PSNR：`17.7513865188403`
- high-route gap：`-0.3054244593410088`
- low-route gap：`+0.06278010621493545`

A28：

- PSNR：`17.73849582669033`
- high-route gap：`-0.3390800359165398`
- low-route gap：`+0.06555415143863476`

结论：简单增强 route capacity 仍然没恢复 A08-like high route。

### 5.5 A29/A30：hard decoder heads 与 trust region 仍不足

A29：

- PSNR：`17.76127769536133`
- high-route gap：`-0.236088142902285`

A29 说明 route delta 增大可能有害，验证随 delta 增长下降。

A30：

- PSNR：`17.75180602939797`
- high-route gap：`-0.23814779446593295`
- selected delta：`0.0000578112014773069`

结论：trust region 能压住有害 drift，但不能创造有效 high-route representation。

### 5.6 A31/A32/A33：transplant + surgery 找到压缩候选

A31：

- PSNR：`17.770243709681385`
- high-route gap 改善到：`-0.14121316850824578`
- low-route gap 退化到：`-0.02801579592015501`

A32：

- PSNR：`17.786780968730323`
- low-route gap：`+0.05380978168030712`
- high-route gap：`-0.211220889864822`

A33：

- PSNR：`17.823062311698262`
- high-route gap：`-0.12677254958688497`
- low-route gap：`+0.053754617905617835`

结论：A33 证明 A31 high route + A24/A32 low protection 有效，首次突破 `17.82`，但仍没追上 A18。

### 5.7 A34/A35：微调和 post-adapter 不够

A34：

- PSNR：`17.820851508384713`
- high-route gap：`-0.1309565862555424`

A35：

- PSNR：`17.823132133493647`
- high-route gap：`-0.12662823419234123`

结论：输出空间微调、post-route adapter 都太晚、太小。

### 5.8 A36：full dual-backbone 上界证明

A36 把 A08/A15 两套完整 backbone 打包进一个 checkpoint，并在 checkpoint 内部用 fixed `input_std` router。

结果：

- PSNR：`17.8470899392002`
- gap vs A18 router：`+0.00015556338577912498`
- high-route gap：约 `-0.0001995`
- low-route gap：约 `+0.0004234`

结论：route-specific capacity 足够时，A08/A15 互补可以被保存。剩下的问题是压缩和共享。

### 5.9 A37：compressed/shared aggregate 突破

A37 从 A33 初始化，使用 A08/A15 native config teachers，锁定 low route，只训练 high/shared late modules。

结果：

- test PSNR：`17.889658453438816`
- standard-loss re-eval loss：`0.11911950781941413`
- total loss：`0.12235063299536705`
- SSIM：`0.7026848953962326`
- overall gap vs A18：`+0.042724077624394126`
- high-route gap：`-0.11159004428502844`
- low-route gap：`+0.1591364853806252`

结论：A37 是真实项目突破。它是第一个超过 A18/A36 aggregate router line 的 compressed/shared candidate。边界是 high-route 仍未达到 `>-0.10 dB`。

### 5.10 A38：standard-loss high-head-only 只有微小增益

A38 从 A37 出发，关闭 teacher branch loss，恢复 standard loss，只训练 high head/feature。

结果：

- PSNR：`17.889819259555214`
- high-route gap：`-0.1111094511726412`

结论：不是有意义的 high-route 突破；训练早期即峰值，后续衰减。

### 5.11 A39：trainable shared control canvas 提 aggregate，但伤 high route

A39 从 A37 出发，让 `gaussian_control_canvas` 可训练。

结果：

- PSNR：`17.900369585886775`
- high-route gap：`-0.11858325626995542`
- low-route gap：`+0.18320352678672946`
- A08-oracle gap：`-0.23903555195244477`

结论：shared control capacity 对 aggregate 有用，但不是 high-route 的正确杠杆。

### 5.12 A40：high-balanced sampling 提 aggregate，但 high route 没破

A40 冻结 `gaussian_control_canvas`，启用 high-`input_std` balanced sampling/crop targeting。

结果：

- PSNR：`17.900562951956445`
- 当前 aggregate numeric best
- high-route gap：`-0.1180115677202543`
- low-route gap：`+0.18311149168725271`
- A08-oracle gap：`-0.23833660113789967`

结论：多喂 high-route 样本不足以恢复 A08-tail。

### 5.13 A41：oracle-weighted high branch 是负例

A41 在 A40 基础上加入 crop-level A08-vs-A15 oracle weighting。

结果：

- PSNR：`17.900201657598004`
- high-route gap：`-0.11813195131188482`
- low-route gap：`+0.182568457627533`
- A08-oracle gap：`-0.23811165869770295`
- high branch L1 vs A08：`0.01348984414754912`

对比 A40：

- A40 high branch L1 vs A08：`0.013522875278668349`
- A41 只改善约 `0.000033031131119229395`
- high-route PSNR gap 没改善。

结论：scalar oracle reweighting 不是缺失机制。

## 6. 瓶颈诊断

### 6.1 路由信号已经被证明

A18/A19/A21 证明 fixed `input_std` router 稳定超过 A08/A15 单 checkpoint。A36 证明 full dual-backbone 单 checkpoint 可以几乎完整复现 A18。

所以不应继续怀疑 route signal 本身。

### 6.2 真实瓶颈是压缩与共享表示

A37 证明压缩可行。但 A39/A40/A41 暴露稳定 failure mode：

- aggregate PSNR 提升；
- low/A15 side 继续提升；
- high-route gap 相对 A37 变差；
- A08-oracle gap 相对 A37 明显变差。

这说明当前共享模型倾向于把新增容量和训练压力转化为 low-side/aggregate gain，而不是 A08-tail fidelity。

### 6.3 late correction 无法完全恢复 A08-like 内部行为

A24、A27、A35、A38 都显示：等 Gaussian attention、control canvas、decoder guidance、feature refinement、shared decoder feature 都已经计算完，再用末端 RGB delta 去修，能力有限。

high route 需要更早、更强的 route-conditioned internal capacity，或者需要显式 hard-case anchor 防止 shared path 擦掉 A08-tail。

### 6.4 Teacher L1 不是 promotion 指标

A39/A40/A41 都让 high branch L1 vs A08 降低，但 high-route PSNR 不升反降。这说明 L1 可能主要改善平均/低频误差，没有覆盖 PSNR-sensitive hard cases 和最终 selected route 行为。

下一步 promotion 必须看：

- high-route gap；
- A08-oracle gap；
- worst-gap list；
- per-sample margin vs A37/A18/A36。

### 6.5 high-input-std 不等于完整 A08-oracle set

worst-case 复盘发现，一些 nominal low-route 样本其实 A08 比 A15 更强，例如：

- `94_9400006.jpg`：route `low_b`，input_std `0.058188`，`a15_minus_a08 = -0.302716`
- `94_94000065.jpg`：route `low_b`，input_std `0.064002`，`a15_minus_a08 = -0.310023`
- `94_94000081.jpg`：route `low_b`，input_std `0.058379`，`a15_minus_a08 = -0.276238`

这解释了为什么 A40/A41 只按 high-`input_std` 加强采样没有解决 A08-oracle gap。

## 7. 当前瓶颈一句话

```text
我们已经做出了 compressed/shared aggregate breakthrough，
但当前共享模型仍不能在保住 A15/low-side 收益的同时，
保存 A08-oracle/high-fidelity hard cases。
```

更直白地说：整体分已经能赢，但模型仍在用一部分 A08-tail 作为代价换整体收益。下一步要打穿的就是这个 tradeoff。

## 8. 建议下一步实验

### A42：A08-oracle hard-case target cache

目的：不再只按 `input_std > threshold` 定义 high target，而是直接针对 A08-oracle hard cases。

设计：

- 从 A18 saved records 建离线 cache/list。
- 包含 `a15_minus_a08 <= 0` 的样本，包括 nominal low-route 但 A08 更强的样本。
- 包含当前 student vs router/A08 gap 最差样本。
- 记录每个样本：
  - filename；
  - input_std；
  - A08 PSNR；
  - A15 PSNR；
  - A18 selected expert；
  - A08/A15 oracle label；
  - A37/A40/A41 gap。
- 对 hard-case 集合施加 explicit A08 teacher anchor 或 GT+teacher mixed target。

验收：

- A08-oracle gap 优于 A37，且不能退到 A40/A41 水平；
- high-route gap 优于 A37 `-0.11159004428502844`；
- worst-gap list 有明确恢复。

### A43：A37 per-sample trust anchor

目的：A37 是当前 high-route 更安全的 compressed anchor。后续训练不能牺牲它的 hard cases。

设计：

- 保存 A37 hard-case 输出或 per-sample PSNR margin。
- 加 hinge/trust loss：

```text
new_model 在 A08-tail/hard-case set 上不能比 A37 差超过小 margin
```

- 允许 low/A15-side 改善，但不允许以 hard-case collapse 为代价。

验收：

- high-route gap 好于 A37；
- A08-oracle gap 不差于 A37；
- 初期 aggregate PSNR 不必超过 A40，只要核心瓶颈指标动了就是科学正例。

### A44：route-conditioned internal capacity

目的：把 route-specific capacity 前移，不再只靠 late RGB delta。

候选结构：

- route-conditioned decoder/refinement blocks；
- last decoder/refinement stages 的 partial dual-expert blocks；
- route-specific Gaussian/control branch，但必须加 anchor，避免 A39 式 shared-control redirection；
- 参考 A36 的 partial dual path，但参数量远小于完整双 backbone。

验收：

- high-route gap 必须优于 A37，最好越过 `>-0.10 dB`；
- A08-oracle gap 不能停留在 A40/A41 的 `~ -0.238 dB`；
- low-route selected delta 继续锁定或变化可解释；
- aggregate PSNR 是次级指标，先看 high-tail。

## 9. 后续 promotion policy

不要因为 aggregate PSNR 小涨就宣称 bottleneck breakthrough。

每个新实验必须报告：

- standard PSNR/SSIM/loss；
- A18 alignment：
  - overall gap；
  - high-route gap；
  - low-route gap；
  - A08-oracle gap；
  - A15-oracle gap；
- route-regime diagnostics：
  - selected delta；
  - high selected delta；
  - low selected delta；
  - high branch L1 vs A08；
  - low branch L1 vs A15；
- worst-gap list vs A18/A36/A37；
- 明确结论：aggregate breakthrough、high-route breakthrough、both、或 neither。

high-tail-first 晋级线：

- 必须优于 A37 high-route gap `-0.11159004428502844`；
- 目标线：high-route gap `>-0.10 dB`；
- A08-oracle gap 必须优于 A37 `-0.19984098726138502`，或者至少不能退到 A40/A41 的 `~ -0.238 dB`；
- low-route surplus 保持正向且可解释。

## 10. 最终判断

我们已经有真实突破：A37 是 compressed/shared aggregate breakthrough，A40 是 aggregate numeric best。现在没有突破的是更窄、更关键的 high-route/A08-oracle hard-case 瓶颈。

下一步最可能有效的突破口不是再调一个 scalar loss weight，也不是再做一个 high sampling variant，更不是继续加很晚的 output adapter。更有希望的是：

- 建 A08-oracle hard-case target cache；
- 用 A37 做 per-sample trust anchor；
- 增加更早、更强的 route-conditioned internal capacity，向 A36 的 route-specific capacity 靠近，但保持紧凑。

这是当前最清晰的瓶颈地图。
