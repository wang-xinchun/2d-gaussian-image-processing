# `M03/A_k03` 统一架构汇报与答辩文档

更新时间: `2026-05-16`

## 0. 使用说明

这是一份**统一版**文档，用来取代此前拆开的两份材料：

- `docs/m03_ak03_architecture_briefing_20260516.md`
- `docs/m03_ak03_architecture_deep_dive_and_defense_20260516.md`

如果你现在要准备汇报，建议只看这一份。

这份文档同时承担四个用途：

1. 给出最短可汇报口径。
2. 讲清楚当前最强冻结里程碑 `M03/A_k03` 的真实架构。
3. 解释每个关键设计为什么存在。
4. 提供面对导师追问时的答辩口径。

本文只锚定**当前最强且已冻结的正式里程碑**：

- 里程碑: `M03`
- 绑定实验: `V1/T01/A_k03`
- 冻结路径: [M03_A_k03_uhddeblur_official_baseline](/root/autodl-tmp/2DGauss/experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline)
- 配置路径: [a_k03_medium_uhddeblur_tlc_stepcap_extension.toml](/root/autodl-tmp/2DGauss/experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline/artifacts/configs/a_k03_medium_uhddeblur_tlc_stepcap_extension.toml)
- 当前冻结 official baseline: `26.199959289578075 dB`

后续 `A_t03/A_u03/A_v03/A_w03` 这些分支，只能作为“Gaussian 后续增强探索”来讲，不能替代 `M03/A_k03` 的正式汇报锚点。

## 1. 先给你最短汇报口径

### 1.1 一句话版本

我们的核心创新不是单独加一个 Gaussian 模块，而是把 2D Gaussian 变成一种面向 UHD 图像复原的**跨分辨率结构化先验**：先在低分辨率空间紧凑预测纹理/频率感知高斯参数，再通过高分辨率引导和低秩仿射调制，把这些先验高效回注到细节恢复过程中。

### 1.2 三句话版本

第一，我们不是在高分辨率上做更重的 dense attention，而是在低分辨率空间建立更紧凑的 Gaussian 参数场。
第二，我们的 Gaussian 不是纯几何高斯，而是带方向、频率、相位和谱衰减的纹理感知高斯原子。
第三，我们再把这些高斯先验渲染成高分辨率 attention，并通过 high-pass 增强和低秩 affine modulation 回注到恢复主干，所以它既轻量，又保留了 UHD 场景下需要的细节表达能力。

### 1.3 里程碑版本

当前最强冻结里程碑 `M03/A_k03` 证明，这条 Gaussian 主线已经不是概念，而是能够在 `UHD_deblur` 上稳定训练、完整评测、可冻结复现，并形成 official baseline 的真实架构版本。

## 2. 这套架构到底想解决什么问题

### 2.1 不是普通 restoration，而是 UHD restoration

普通图像复原里，很多方法可以默认接受：

- 更重的 backbone
- 更大的 dense attention
- 更贵的高分辨率全局建模

但到了 UHD 场景，问题变成：

1. 图像尺寸极大，full-resolution dense 建模代价很高。
2. 如果过度依赖低分辨率恢复，又容易丢失细节。
3. 真正难的不是有没有语义，而是有没有**足够便宜、足够稳定、又足够细**的细节表达方式。

所以我们真正回答的是：

> 能不能不在原图尺度上做特别重的 dense 建模，但又仍然保住 UHD 图像最关键的高频与细节恢复能力？

### 2.2 我们主动避开的两个极端

#### 极端一：全高分辨率重建模

问题：

1. 成本高。
2. 显存压力大。
3. 不符合“高斯服务于轻量化”的项目定位。

#### 极端二：纯低分辨率恢复后直接上采样

问题：

1. 高频细节容易被平滑掉。
2. 低分辨率预测很难直接承担最终细节恢复。
3. 恢复结果容易只有低频主体，没有高质量纹理。

因此我们的答案不是二选一，而是中间路线：

> 低分辨率负责紧凑参数化建模，高分辨率负责结构化细节回注。

## 3. 为什么我们的创新点是 Gaussian

### 3.1 Gaussian 在这里不是几何装饰，而是结构化表示语言

我们引入 2D Gaussian，不是为了做一个“看起来很新”的几何模块，而是把它升级成：

- 可在低分辨率网格上预测的紧凑参数单元；
- 可显式编码位置、尺度、方向的局部结构单元；
- 可继续承载纹理、频率和相位信息的细节先验单元；
- 可跨分辨率传递到高分辨率恢复过程中的引导单元。

因此，我们的 Gaussian 不是普通 feature map，也不是普通 attention map，而是一种：

> 可解释、可渲染、可调制的中间结构语言。

### 3.2 为什么 Gaussian 适合 UHD

Gaussian 对我们有三个特别关键的优点：

1. 它天然适合低分辨率紧凑表达。
2. 它比纯卷积特征图更有结构约束，便于跨分辨率传递。
3. 它能把“哪里重要、哪个方向重要、什么频率重要”压缩到少量参数中。

这非常适合 UHD 场景，因为 UHD 的真正难点不是缺一个更大的 dense 表示，而是缺一个：

> 更高效的高频和细节表达方式。

### 3.3 我们的 Gaussian 不是传统纯几何高斯

在当前实现中，每个 Gaussian 原子不是只含几何参数，而是包含 `10` 个维度：

1. `amplitude`
2. `mu_x`
3. `mu_y`
4. `sigma_x`
5. `sigma_y`
6. `theta`
7. `texture_gain`
8. `frequency`
9. `phase`
10. `beta`

这意味着：

- `mu / sigma / theta` 表示局部几何包络；
- `texture_gain / frequency / phase` 表示局部纹理载波；
- `beta` 表示频谱衰减。

所以我们的 Gaussian 已经不是“模糊斑点”，而是：

> 纹理/频率感知高斯原子。

这是你面对导师时最应该坚持的创新表述：

> 我们不是简单“用了 Gaussian”，而是把 Gaussian 从传统几何原子，升级成了服务于 UHD 图像复原的参数化结构先验。

## 4. 当前最强里程碑 `M03/A_k03` 的真实架构边界

这一节一定要讲准，因为很容易把后续探索分支和正式里程碑混在一起。

### 4.1 `M03/A_k03` 真正启用的模块

`M03/A_k03` 中实际启用的主体包括：

1. 高分辨率主干 stem
2. `pixel_unshuffle` 低分辨率分支
3. 纹理/频率描述子提取
4. Gaussian 参数头
5. Gaussian 渲染器
6. Directional frequency prior 投影
7. High-pass Gaussian guidance
8. 低秩仿射调制
9. Edge-guided high-resolution refiner
10. TLC-style train-inference consistency

### 4.2 `M03/A_k03` 中没有启用的模块

下面这些是后续探索项，不属于当前冻结最强里程碑的正式主体：

1. `GaussianGuidedDecoderModulator`
2. `GaussianGuidedTerminalResidualHead`
3. `GaussianGuidedFeatureRefinement`
4. `edge_weighted_reconstruction`

也就是说：

- `M03/A_k03` 已经证明 Gaussian 主线成立；
- 后续 `A_t03/A_u03/A_v03/A_w03` 证明的是 Gaussian 还能否作用到更靠后的层级。

### 4.3 `M03/A_k03` 的关键配置

`M03/A_k03` 的关键配置为：

- `width = 48`
- `lowres_scale = 4`
- `highres_blocks = 3`
- `lowres_blocks = 6`
- `decoder_blocks = 3`
- `affine_rank = 6`
- `gaussian_atoms = 6`
- `gaussian_param_dim = 10`
- `max_frequency = 6.0`
- `max_texture_gain = 1.2`
- `use_edge_guidance = true`
- `use_directional_frequency_guidance = true`
- `use_train_inference_consistency = true`
- `train_context_size = 784`
- `train_supervision_crop_size = 512`
- `batch_size = 7`
- `wall_clock_minutes = 300`

里程碑 formal preflight 的 `total_params = 246888`。

这个数字对汇报很有价值，因为它能支持一个重要论断：

> 我们的模型参数量并不大，UHD 训练的主要成本更多来自高分辨率激活和训练协议，而不是模型参数无节制膨胀。

## 5. 整体数据流与张量流

这一节是你真正吃透架构的关键。

主干实现位于：

- [restorer.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/restorer.py)

### 5.1 输入与 padding

输入：

```text
x: [B, 3, H, W]
```

首先执行：

```text
pad_to_multiple(x, lowres_scale)
```

目的：

1. 低分辨率分支依赖 `pixel_unshuffle`。
2. `pixel_unshuffle` 要求空间尺寸能被 `lowres_scale` 整除。
3. 保证低分辨率参数网格与高分辨率回注严格对齐。

### 5.2 高分辨率分支

高分辨率分支直接处理 padded 后的 RGB：

```text
highres_feature = highres_stem(padded)
```

输出近似为：

```text
[B, 48, H, W]
```

职责：

1. 保留原始像素网格上的细节载体；
2. 作为高分辨率恢复底座；
3. 接收低分辨率 Gaussian 先验的结构化回注。

一句话：

> 高分辨率分支负责保留“细节长在哪里、最后怎么落回像素网格”。

### 5.3 低分辨率分支

低分辨率分支不是简单插值下采样，而是：

```text
lowres_input = pixel_unshuffle(padded, 4)
```

若输入为 `[B, 3, H, W]`，则得到：

```text
[B, 3 * 4 * 4, H/4, W/4] = [B, 48, H/4, W/4]
```

这一步意义非常大：

1. 空间尺寸下降为原来的 `1/16`。
2. 原本局部 `4x4` 邻域信息被搬到通道维。
3. 比直接双线性下采样更保留局部原始细节。

你可以这样回答导师：

> 我们不是简单把大图模糊缩小，而是把局部高分辨率证据重排到低分辨率通道中，让低分辨率分支在压缩空间上仍然看到足够多的局部细节。

### 5.4 低分辨率纹理/频率描述子

低分辨率分支里显式计算：

- gray
- Sobel x
- Sobel y
- gradient magnitude
- Laplacian

实现见：

- [frequency.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/frequency.py)

逻辑是：

```text
lowres_rgb -> gray / grad_x / grad_y / grad_mag / lap
concat with lowres_feature
1x1 proj + residual refine
```

为什么要这样设计：

1. 纯语义特征容易偏向平滑和低频。
2. Gaussian 参数如果只看平滑语义，容易退化成“哪里有东西”的粗引导。
3. UHD 恢复真正重要的是边缘、方向、纹理和高频变化。

本质上，这一步是在做：

> 给低分辨率 Gaussian 参数预测提供显式高频证据，而不是完全依赖黑箱特征自己发现。

### 5.5 Gaussian 参数头

Gaussian 参数头见：

- [gaussian.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/gaussian.py)

它输出两类量：

1. `params`
2. `coeffs`

其中 `params` 本质形状是：

```text
[B, H_low, W_low, atoms, param_dim]
```

在 `M03/A_k03` 中：

- `atoms = 6`
- `param_dim = 10`

这意味着在低分辨率每个网格位置上，我们预测 `6` 个纹理/频率感知 Gaussian 原子。

### 5.6 每个 Gaussian 原子到底表达什么

当前 `10` 维参数分别对应：

1. 强度
2. 中心 x
3. 中心 y
4. x 方向尺度
5. y 方向尺度
6. 方向角
7. 纹理增益
8. 纹理频率
9. 相位
10. 高频衰减

这是我们“可解释性”的第一根支柱，因为它说明：

- 这不是一个不可解释的 dense 向量；
- 而是一组具有明确几何和频率意义的局部参数。

### 5.7 Gaussian 渲染器

Gaussian 渲染器把低分辨率参数场渲染为高分辨率 `attention_map`。

它在数学上做的是：

1. 用 `mu / sigma / theta` 构造定向高斯包络；
2. 用 `texture_gain / frequency / phase` 构造局部纹理载波；
3. 用 `beta` 控制高频衰减；
4. 多原子求和；
5. 把低分辨率 patch-level 结果拼回高分辨率 attention。

输出：

```text
attention_map: [B, 1, H, W]
```

很关键的一点是：

> 这里输出的是单通道结构化引导图，而不是多头 dense attention。

这反映了我们的立场：

- Gaussian 负责给出“哪里该增强、增强何种结构”；
- 不负责直接替代整套高分辨率表征。

## 6. 高频回注为什么这么设计

### 6.1 先做 high-pass，再做 Gaussian 调制

主干里的关键操作是：

```text
highpass = highres_feature - avg_pool2d(highres_feature)
modulated = highres_feature + attention_map * highpass
```

这一步体现了方法设计上的克制。

我们没有做的是：

- `attention_map * highres_feature`
- 或者用 attention 直接改写整张高分辨率特征图

我们做的是：

- 只让 Gaussian 重点增强高频残差部分

为什么：

1. restoration 最缺的是高频细节，而不是低频主体。
2. 低频主体本来已经在输入和主干里保留得不错。
3. 如果 Gaussian 对整张特征图平均施力，容易过度改写，甚至引入伪影。

所以这个设计的精髓是：

> 高斯不是来重建整张图的，它是来告诉网络“细节该怎么补”的。

### 6.2 这一步的可解释性

这一步特别好讲，因为：

- `attention_map` 决定空间上哪里更值得增强；
- `highpass` 明确代表高频残差信号；
- 两者相乘就是“在该补细节的地方补细节”。

相比纯黑箱 attention，这一步解释性明显更强。

## 7. 为什么还要有低秩 affine modulation

Gaussian 渲染出的 `attention_map` 是空间上的单通道引导，但细节恢复不只是空间问题，还有通道问题。

因此我们还预测 `affine_coeffs`，通过：

```text
features * (1 + scale) + bias
```

进行通道级调制。

实现见：

- [affine.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/affine.py)

### 7.1 为什么不用完整高分辨率通道仿射图

如果直接在高分辨率上预测完整 `C x H x W` 的 scale/bias：

1. 参数与显存成本太高；
2. 会破坏“低分辨率紧凑建模”的初衷；
3. 不符合轻量化目标。

所以我们改成：

1. 在低分辨率上预测 `rank = 6` 个系数图；
2. 通过学习到的 `scale_basis / bias_basis` 投影成高分辨率通道仿射参数。

这就是“低秩 affine modulation”的本质。

### 7.2 为什么这个设计合理

它有三个好处：

1. 用很小的 rank 表达大量通道调制行为。
2. 让低分辨率紧凑先验继续影响高分辨率通道语义。
3. 在表达能力与轻量化之间取得折中。

如果导师问：

> 为什么不直接做更自由、更强的 modulation？

推荐回答：

> 因为我们的目标不是最大自由度，而是受控的 UHD 友好调制。低秩设计是在限制高分辨率侧自由度，把复杂性尽量压回低分辨率参数场。

## 8. Directional prior 与 edge refiner 的角色

### 8.1 Directional prior 不是装饰，而是方向结构先验

`M03/A_k03` 开启了 `use_directional_frequency_guidance = true`。

Directional projector 会从 Gaussian 参数里聚合出一个 `6` 通道的方向/频率先验：

1. `dir_x`
2. `dir_y`
3. `freq_norm`
4. `texture_norm`
5. `coherence`
6. `beta_norm`

这意味着 Gaussian 不只告诉我们“哪里重要”，还告诉我们：

- 主方向是什么；
- 主要频率有多高；
- 纹理强度如何；
- 结构是否具有稳定方向性。

### 8.2 Edge refiner 为什么还需要

Gaussian 提供的是跨分辨率结构化先验，但最终图像恢复里最脆弱的仍然是边界和细线条。

所以 `M03/A_k03` 还启用了 edge refiner，它显式提取：

- gray
- `|grad_x|`
- `|grad_y|`
- `grad_mag`
- `|lap| + |local_highpass|`

然后把这些 edge cues 与 decoder feature、attention_map、directional_prior 融合，输出最后一段 refinement residual。

它的角色不是取代 Gaussian，而是做：

> Gaussian 先验指导下的最后一公里边缘修补。

### 8.3 为什么需要同时有 Gaussian 和 edge refiner

如果导师问：

> 既然你有 edge module，Gaussian 的必要性是什么？

最稳妥的回答是：

1. edge module 处理的是显式局部边界证据；
2. Gaussian 处理的是低分辨率紧凑结构先验和跨分辨率引导；
3. 二者层级不同，不是替代关系。

一句话：

> edge refiner 是局部细修，Gaussian 是跨分辨率结构主线。

## 9. 哪些信息走高分辨率分支，哪些信息走低分辨率高斯分支

这是理解双分支架构最关键的问题之一。

### 9.1 高分辨率分支更适合承载什么

高分辨率分支适合承载：

1. 精确空间位置
2. 局部边缘形状
3. 细小纹理的真实落点
4. 最终 RGB 恢复所需的像素级载体

换句话说，它保留的是：

> 细节“长在哪里、怎么落到像素网格上”。

### 9.2 低分辨率高斯分支更适合承载什么

低分辨率高斯分支适合承载：

1. 局部结构包络
2. 方向性
3. 高频/纹理强弱
4. 纹理频率与相位趋势
5. 跨更大空间范围的紧凑先验

换句话说，它保留的是：

> 细节“应该是什么结构类型、该往哪个方向增强、该强调什么频率”。

### 9.3 为什么这样分工合理

因为 UHD restoration 里最贵的部分，不应该去表达所有细节本身，而应该去表达：

- 哪些地方值得投入恢复容量；
- 恢复应该偏向何种结构；
- 如何把这些结构信息用小代价传给高分辨率主干。

这正是 Gaussian 分支最适合做的事情。

## 10. 可解释性到底体现在哪里

### 10.1 我们的可解释性是“相对可解释”，不是完全白盒

这套架构不是传统物理模型，不可能做到每个像素都有人类可读解释。

但相对纯 dense feature map / dense attention，它明显更可解释，因为中间变量拥有清晰角色：

1. `gaussian_params`
2. `attention_map`
3. `directional_prior`
4. `affine_coeffs`
5. `texture_features`

### 10.2 可以被可视化和解释的中间量

#### `gaussian_params`

可解释为：

- 强度
- 中心
- 方向
- 尺度
- 纹理频率
- 相位
- 高频衰减

#### `attention_map`

可解释为：

- 模型认为哪些空间区域更需要高频增强

#### `directional_prior`

可解释为：

- 区域的主方向和频率结构

#### `affine_coeffs`

可解释为：

- 低分辨率先验如何调节高分辨率通道

#### `texture_features`

可解释为：

- 低分辨率分支所感知到的显式高频/纹理证据

### 10.3 我们为什么敢说比普通 attention 更可解释

因为普通 attention 往往只能告诉你“模型看了哪里”，很难回答：

- 为什么看这里；
- 看的是方向、频率还是纹理；
- 后续如何作用到通道恢复里。

而我们的 Gaussian 链条至少把这些问题拆开了：

1. 参数化先验是什么；
2. 渲染出的空间引导是什么；
3. 高频增强如何发生；
4. 通道调制如何发生。

## 11. 训练设计为什么和架构绑得这么紧

`M03/A_k03` 不是单纯一个网络结构，它还有非常重要的训练协议配套：

- `use_train_inference_consistency = true`
- `train_context_size = 784`
- `train_supervision_crop_size = 512`

### 11.1 为什么不是直接 512 patch 训练到底

因为我们已经明确知道，大图 restoration 里存在：

> patch train / full-image inference mismatch

只给模型看小 patch，它学到的行为可能和 full-image 推理时不一致。

### 11.2 为什么 context 比 supervision 大

当前训练设计是：

- 给模型更大的上下文窗口 `784`
- 只对中心 `512` 区域做监督

目的：

1. 让模型在训练中看到更大空间上下文；
2. 同时控制显存与训练稳定性；
3. 减少纯 patch 训练带来的边界伪差和协议偏移。

所以更准确地说：

> 架构负责跨分辨率信息流，训练协议负责减少 patch/full-image 行为错配，两者是配套设计。

## 12. `M03/A_k03` 为什么是可信里程碑

因为它不是一次“刚好跑出最高分”的偶然结果，而是满足完整 acceptance 条件：

1. frozen weights 存在
2. code snapshot 存在
3. logs 存在
4. train/val/test metrics 存在
5. result_analysis 存在
6. previous baseline 对比明确

并且它给出两个很强的实验事实：

1. `A_k03 final.pt` 是当前冻结 official baseline，`26.199959289578075 dB`
2. best internal val checkpoint 不是 best official test checkpoint

这让它在汇报上更可信，因为这说明我们不是只靠 internal val 自我挑选最好点，而是承认 official post-hoc 才是最终接受标准。

## 13. 这套架构最容易被导师问的质疑，以及建议回答

下面这部分建议你直接熟悉到可以脱口而出。

### 质疑 1：为什么非得用 Gaussian，普通 CNN/Transformer modulation 不行吗？

建议回答：

普通 modulation 当然也能做，但我们的目标不是再做一个更强黑箱调制器，而是要在 UHD 场景下找到一种更紧凑、更结构化、跨分辨率更自然的先验表达。Gaussian 的优势在于它既有几何结构，又能扩展进方向、频率和相位信息，因此更适合作为低分辨率紧凑参数场，再渲染回高分辨率引导。

### 质疑 2：为什么不直接在高分辨率上做 attention？

建议回答：

因为 UHD 场景下 full-resolution dense attention 的代价太高，而且不符合我们轻量化的核心目标。我们不是否认 attention 有效，而是把“高分辨率直接学习巨大 attention”改成“低分辨率学习高斯参数，再渲染高分辨率结构化引导”，本质上是在重写 attention 的生成方式。

### 质疑 3：为什么 `attention_map` 只有单通道，会不会表达力不够？

建议回答：

我们刻意让空间引导保持单通道，是为了让它承担“结构位置引导”而不是“全功能表示”。真正的通道级自由度由后面的低秩 affine modulation 补足。这样空间与通道两种作用被解耦，更容易控制，也更符合轻量化约束。

### 质疑 4：为什么还要显式加 Sobel、Laplacian，这不是手工特征吗？

建议回答：

是的，它们是显式先验，但这里不是为了回到传统 handcrafted pipeline，而是为了给低分辨率 Gaussian 参数预测提供稳定高频证据。因为 UHD 复原里真正紧缺的是高频结构表达，完全依赖黑箱语义特征容易让 Gaussian 参数退化成平滑区域注意力，而不是细节结构先验。

### 质疑 5：为什么先做 high-pass 再调制？

建议回答：

因为 restoration 最需要增强的是细节残差，而不是把整张高分辨率特征都重新写一遍。先提取 high-pass，再用 Gaussian attention 去增强，本质上是把模型能力集中投入到真正困难的高频恢复上。

### 质疑 6：为什么仿射调制要做成低秩？

建议回答：

直接生成完整高分辨率通道仿射图代价太高，也会破坏低分辨率紧凑建模的初衷。低秩设计是一种受控自由度：它保留通道调制能力，但把复杂性约束在一个较小 rank 上，更符合 UHD 轻量化目标。

### 质疑 7：你的高斯真的在里程碑里起核心作用吗？还是附属模块？

建议回答：

在 `M03/A_k03` 里，高斯不是附属模块，而是在主信息流上。低分辨率参数头、Gaussian renderer、directional prior、high-pass modulation、affine modulation 都处于主路径中。后续新增的 decoder guidance 或 feature refinement 才是高斯作用点的进一步延伸，不是高斯主线是否存在的前提。

### 质疑 8：为什么你们后来又去试 feature refinement、terminal residual，这是不是说明原架构不成立？

建议回答：

恰恰相反。正是因为 `M03/A_k03` 已经证明 Gaussian 主线成立，我们才有资格继续问“Gaussian 放在哪个层级最有效”。后续分支是在探索更优作用点，不是在补救主线不存在的问题。

### 质疑 9：这套架构最强在哪里，最弱又在哪里？

建议回答：

最强的是它给出了一个符合 UHD 条件的紧凑结构化建模路线，高斯既可解释又适合跨分辨率回注，并且已经形成可冻结 baseline。最弱的是绝对性能离更强 public reference 仍有差距，说明我们当前高斯作用方式虽然成立，但还没有完全打穿高分辨率细节表达瓶颈。

### 质疑 10：你们为什么说这个架构有解释性，但又不敢说是完全可解释？

建议回答：

因为我们确实有一组比普通黑箱 attention 更有语义的中间变量，比如高斯参数、方向先验、attention map 和 affine 系数，但它本质上仍是神经网络，不是显式物理反演模型。我们说的是“结构上更可解释”，不是“逐像素完全白盒”。

## 14. 这套架构当前最诚实的总结

如果面对认真追问的导师，最稳妥的总结不是吹得很满，而是这样说：

1. 我们已经证明 Gaussian 可以作为 UHD restoration 的核心结构化中间表示，而不是边缘模块。
2. 我们已经证明“低分辨率紧凑参数化 + 高分辨率结构化回注”这条主线能形成稳定、可复现、可冻结的 strongest baseline。
3. 我们还没有证明当前版本已经达到这条主线的最终上限，因此后续才继续探索 Gaussian 的更强作用点。

换句话说：

> `M03/A_k03` 的价值不是“问题已经解决完了”，而是“我们已经找到了一个可信、可解释、可扩展的正确主线”。

## 15. 如果你要画图，建议怎么画

建议画一张 6 步图：

1. 输入 `x`
2. 高分辨率 stem
3. `pixel_unshuffle` 低分辨率分支
4. Gaussian 参数头
5. Gaussian renderer + directional prior
6. high-pass modulation + low-rank affine + edge refiner + decoder

图上最值得标出来的四个点：

1. `lowres_scale = 4`
2. `atoms = 6, param_dim = 10`
3. `attention_map` 是单通道高分辨率结构化引导
4. `affine_rank = 6`

如果要再加一张解释图，最推荐：

1. 输入图
2. `attention_map`
3. `directional_prior` 中的方向分量
4. 输出图

这样最容易把“可解释性”讲出来。

## 16. 最后一段你可以直接背

> 我们的核心不是简单把高斯加进网络，而是把 2D Gaussian 变成了一种跨分辨率结构化先验：在低分辨率空间里用纹理/频率感知高斯原子做紧凑参数化建模，再把这些先验渲染成高分辨率 attention，并通过 high-pass 增强和低秩仿射调制回注到恢复主干里。这样既避免了 UHD 条件下昂贵的 full-resolution dense 建模，又尽量保留了细节恢复所需的高频表达能力。
