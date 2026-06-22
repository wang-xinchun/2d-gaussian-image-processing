# `M03/A_k03` 架构深度说明与答辩手册

更新时间: `2026-05-16`

## 1. 这份文档是干什么的

这不是一份“简短汇报稿”，而是一份给你**真正吃透架构**用的深度文档。

目标有四个：

1. 把当前最强冻结里程碑 `M03/A_k03` 的真实架构讲清楚。
2. 把每个模块为什么这么设计讲清楚。
3. 把这套架构到底“可解释”到什么程度讲清楚。
4. 给出面对导师追问时可直接使用的答辩口径。

本文默认只锚定**当前已冻结、已接受、可复现**的最强里程碑：

- 里程碑: `M03`
- 绑定实验: `V1/T01/A_k03`
- 冻结包: [M03_A_k03_uhddeblur_official_baseline](/root/autodl-tmp/2DGauss/experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline)
- 对应配置: [a_k03_medium_uhddeblur_tlc_stepcap_extension.toml](/root/autodl-tmp/2DGauss/experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline/artifacts/configs/a_k03_medium_uhddeblur_tlc_stepcap_extension.toml)
- 当前冻结 official baseline: `26.199959289578075 dB`

如果你只需要一份较短的汇报版，请看：

- [m03_ak03_architecture_briefing_20260516.md](/root/autodl-tmp/2DGauss/docs/m03_ak03_architecture_briefing_20260516.md)

## 2. 先把一句话讲准

这套架构最准确的一句话不是“我们用了高斯”，而是：

> 我们提出的是一种面向 UHD 图像复原的跨分辨率 Gaussian-guided 恢复框架：在低分辨率空间预测纹理/频率感知 2D Gaussian 参数场，再把这些结构化先验渲染并调制回高分辨率特征恢复过程。

这里面有四个关键词，一个都不能少：

1. `跨分辨率`
2. `Gaussian-guided`
3. `纹理/频率感知`
4. `结构化回注`

## 3. 这套架构到底想解决什么问题

### 3.1 不是普通 restoration 的问题，而是 UHD restoration 的问题

普通图像复原里，很多方法默认可以接受：

- 更重的 backbone
- 更大的 dense attention
- 更高分辨率上的更密集建模

但在 UHD 场景下，问题立刻变了：

1. 图像尺寸太大，原图尺度 dense 建模的显存和计算成本非常高。
2. 如果过度依赖低分辨率建模，又会损失最关键的高频细节。
3. 真正困难的不是“有没有语义”，而是“有没有足够细、足够稳、又足够便宜的细节表达方式”。

所以我们真正回答的是：

> 能不能不在原图尺度上做特别重的 dense 建模，但又仍然保住 UHD 场景最重要的高频与细节恢复能力？

### 3.2 我们为什么不走两种常见极端

我们显式避开了两个极端：

#### 极端一：全高分辨率重建模

问题：

1. 成本高。
2. 不轻量。
3. 不符合我们“高斯主要服务于轻量化”的项目定位。

#### 极端二：纯低分辨率恢复后简单上采样

问题：

1. 高频细节容易被平滑掉。
2. 恢复结果容易只保留低频主体。
3. 很难在 UHD 细节上给出可信的恢复。

因此我们的答案不是二选一，而是：

> 低分辨率负责紧凑建模，高分辨率负责结构化细节回注。

## 4. 当前冻结里程碑 `M03/A_k03` 的准确架构边界

这一节非常重要，因为导师很可能会问：

> 你现在讲的这些 Gaussian decoder guidance、feature refinement，到底是不是你最强里程碑里已经用了的？

答案必须非常严谨。

### 4.1 `M03/A_k03` 中实际启用的模块

`M03/A_k03` 这版配置中，以下模块是**真实启用**的：

1. 高分辨率主干 stem
2. 低分辨率分支 `pixel_unshuffle + lowres_proj`
3. 纹理/频率描述子提取
4. Gaussian 参数头
5. Gaussian 渲染器
6. Directional frequency prior 投影
7. High-pass Gaussian guidance
8. 低秩仿射调制
9. Edge-guided high-resolution refiner
10. TLC-style train-inference consistency

### 4.2 `M03/A_k03` 中没有启用的模块

下面这些是**后续探索项**，不属于 `M03/A_k03` 的正式已验证主体：

1. `GaussianGuidedDecoderModulator`
2. `GaussianGuidedTerminalResidualHead`
3. `GaussianGuidedFeatureRefinement`
4. `edge_weighted_reconstruction`

也就是说，汇报时必须区分：

- `M03/A_k03` 证明的是 Gaussian 主线本身已经成立；
- 后续 short probes 证明的是“Gaussian 还能不能作用到更靠后的层级”。

### 4.3 `M03/A_k03` 的关键配置

从 [a_k03_medium_uhddeblur_tlc_stepcap_extension.toml](/root/autodl-tmp/2DGauss/experiments/V1/milestones/M03_A_k03_uhddeblur_official_baseline/artifacts/configs/a_k03_medium_uhddeblur_tlc_stepcap_extension.toml) 看，`M03/A_k03` 的关键配置是：

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

此外，里程碑记录里 formal preflight 的 `total_params = 246888`。

这个数字在汇报里很有用，因为它能支持你讲“主干参数量并不大，UHD 成本主要来自激活和高分辨率训练，而不是模型参数膨胀”。

## 5. 整体数据流和张量流

这一节是“真正吃透架构”的核心。

主干实现在：

- [restorer.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/restorer.py)

### 5.1 输入与 padding

输入是：

```text
x: [B, 3, H, W]
```

首先会执行：

```text
pad_to_multiple(x, lowres_scale)
```

原因：

1. 低分辨率分支依赖 `pixel_unshuffle`。
2. `pixel_unshuffle` 要求空间尺寸能被 `lowres_scale` 整除。
3. 这样能保证低分辨率参数网格和后续高分辨率回注之间严格对齐。

### 5.2 高分辨率分支

高分辨率分支直接处理 padded 后的 RGB：

```text
highres_feature = highres_stem(padded)
```

输出形状：

```text
[B, 48, H, W]
```

它的职责不是做全局建模，而是：

1. 保留原始像素网格上的细节载体；
2. 作为高分辨率恢复的空间底座；
3. 接收低分辨率 Gaussian 先验的回注。

一句话：

> 高分辨率分支保留“细节发生的位置和画布”。

### 5.3 低分辨率分支

低分辨率分支不是普通插值下采样，而是：

```text
lowres_input = pixel_unshuffle(padded, 4)
```

若输入是 `[B, 3, H, W]`，那么：

```text
lowres_input: [B, 3 * 4 * 4, H/4, W/4] = [B, 48, H/4, W/4]
```

这一步的意义非常大：

1. 空间尺寸下降为原来的 `1/16`。
2. 原本局部 `4x4` 邻域的信息被搬到通道维。
3. 比直接双线性下采样更保留局部原始信息。

这也是一个很适合回答导师的点：

> 我们不是简单把大图模糊缩小，而是把局部高分辨率信息重排到低分辨率通道中，让低分辨率分支在压缩空间上仍然看到足够多的局部细节证据。

### 5.4 低分辨率纹理/频率描述子

在低分辨率分支里，我们还显式计算纹理/频率描述子：

- gray
- Sobel x
- Sobel y
- gradient magnitude
- Laplacian

对应实现见：

- [frequency.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/frequency.py)

这里的逻辑是：

```text
lowres_rgb -> gray / grad_x / grad_y / grad_mag / lap
concat with lowres_feature
1x1 proj + residual refine
```

为什么这么做：

1. 纯语义特征容易偏向平滑和低频。
2. Gaussian 参数如果只看平滑语义，容易退化成“哪里有物体”的粗引导。
3. UHD 复原真正重要的是边缘、方向、纹理和高频变化。

所以这一步的本质是：

> 我们强迫低分辨率 Gaussian 参数预测看到显式的高频证据，而不是只让它从黑箱特征里自己猜。

### 5.5 Gaussian 参数头

Gaussian 参数头见：

- [gaussian.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/gaussian.py)

它输出两类结果：

1. `params`
2. `coeffs`

其中 `params` 的本质形状是：

```text
[B, H_low, W_low, atoms, param_dim]
```

在 `M03/A_k03` 中：

- `atoms = 6`
- `param_dim = 10`

也就是说，在低分辨率每个网格位置上，我们预测 `6` 个纹理/频率感知高斯原子。

### 5.6 每个 Gaussian 原子到底表达什么

当前 `10` 维参数分别表示：

1. `amplitude`: 强度
2. `mu_x`: 局部中心 x
3. `mu_y`: 局部中心 y
4. `sigma_x`: x 方向尺度
5. `sigma_y`: y 方向尺度
6. `theta`: 方向角
7. `texture_gain`: 纹理载波增益
8. `frequency`: 纹理频率
9. `phase`: 相位
10. `beta`: 频谱衰减

这件事是我们“可解释性”的第一根支柱。

因为这说明：

- 并不是一个不可解释的 dense 向量；
- 而是一组具有清晰物理/几何/频率意义的局部参数。

### 5.7 Gaussian 渲染器

Gaussian 渲染器会把低分辨率参数场渲染为高分辨率 `attention_map`。

它在数学上做的是：

1. 用 `mu / sigma / theta` 形成定向高斯包络；
2. 用 `texture_gain / frequency / phase` 形成局部纹理载波；
3. 用 `beta` 控制高频衰减；
4. 多个原子求和；
5. 把低分辨率 patch-level 结果拼接回高分辨率 attention。

输出形状：

```text
attention_map: [B, 1, H, W]
```

非常值得强调的一点是：

> 这里输出的是单通道结构化引导图，而不是多头 dense attention。

这体现了我们的架构立场：

- Gaussian 负责给出“哪里该增强、增强什么类型的结构”；
- 不负责直接替代整套高分辨率表征。

## 6. 高频回注为什么这么设计

### 6.1 先做 high-pass，再做 Gaussian 调制

当前主干里的关键操作是：

```text
highpass = highres_feature - avg_pool2d(highres_feature)
modulated = highres_feature + attention_map * highpass
```

这个设计非常值得汇报，因为它体现了方法论上的克制。

我们没有做的是：

- `attention_map * highres_feature`
- 或者用 attention 直接重写整个高分辨率特征

我们做的是：

- 只让 Gaussian 重点增强高频残差部分

为什么：

1. 复原任务最缺的是高频细节，而不是低频主体。
2. 低频主体本来已经在输入和主干里保留得不错。
3. 如果 Gaussian 对整张特征图平均施力，容易造成过度改写甚至伪影。

所以这个设计的精髓是：

> 高斯不是来重建整张图的，它是来指导“细节该怎么补”的。

### 6.2 这一步的可解释性

这一步特别好解释，因为你可以直接说：

- `attention_map` 决定空间上哪里该更关注；
- `highpass` 明确代表高频残差信号；
- 两者相乘的结果就是“在该补细节的地方补细节”。

这比纯黑箱 attention 更容易讲清楚。

## 7. 为什么还要有低秩 affine modulation

高斯渲染出的 `attention_map` 是空间上的单通道引导，但细节恢复不只是空间问题，还有通道问题。

所以我们还会预测 `affine_coeffs`，并通过：

```text
features * (1 + scale) + bias
```

去做通道级调制。

对应实现见：

- [affine.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/affine.py)

### 7.1 为什么不用完整通道仿射图

如果直接在高分辨率上预测完整 `C x H x W` 的 scale/bias：

1. 参数和显存成本太大；
2. 很容易违背我们轻量化目标；
3. 也会让低分辨率紧凑先验的意义被削弱。

所以我们改成：

1. 在低分辨率上预测 `rank = 6` 个系数图；
2. 通过固定学习的 `scale_basis / bias_basis` 投影成通道仿射参数。

这就是“低秩 affine modulation”的本质。

### 7.2 为什么这个设计合理

这个设计的好处有三个：

1. 用很小的 rank 表达大量通道调制行为。
2. 让低分辨率紧凑参数场继续影响高分辨率通道语义。
3. 在“表达能力”和“轻量化”之间取得折中。

导师如果问你：

> 为什么不直接做更强、更自由的 modulation？

你可以答：

> 因为我们的目标不是最大自由度，而是可控的 UHD 友好调制。低秩设计是为了限制高分辨率侧的自由度，把复杂性尽量压回低分辨率参数场。

## 8. Directional prior 和 edge refiner 的角色

### 8.1 Directional prior 不是装饰，而是方向结构先验

`M03/A_k03` 开启了 `use_directional_frequency_guidance = true`。

Directional projector 会从 Gaussian 参数里聚合出一个 `6` 通道的方向/频率先验：

1. `dir_x`
2. `dir_y`
3. `freq_norm`
4. `texture_norm`
5. `coherence`
6. `beta_norm`

也就是说，Gaussian 不只是告诉我们“哪里重要”，还会告诉我们：

- 主要方向是什么；
- 频率大概多高；
- 纹理强度如何；
- 结构是否具有一致方向性。

### 8.2 Edge refiner 为什么还需要

Gaussian 提供的是结构化全局/局部先验，但最终图像恢复里最脆弱的往往还是边界和细线条。

所以 `M03/A_k03` 里还启用了 edge refiner，它会显式提取：

- gray
- `|grad_x|`
- `|grad_y|`
- `grad_mag`
- `|lap| + |local_highpass|`

然后把这些 edge cues 与 decoder feature、attention_map、directional_prior 融合，再输出最后一段小的 refinement residual。

它的角色不是取代 Gaussian，而是做：

> 高斯先验指导下的最后一公里边缘修补。

### 8.3 为什么要同时有 Gaussian 和 edge refiner

如果导师问：

> 既然你有 edge module，Gaussian 的必要性是什么？

最好的回答是：

1. edge module 处理的是显式局部边界证据；
2. Gaussian 处理的是低分辨率紧凑结构先验和跨分辨率引导；
3. 两者层级不同，不是互相替代关系。

一句话：

> edge refiner 是局部细修，Gaussian 是跨分辨率结构主线。

## 9. 哪些信息走高分辨率分支，哪些信息走低分辨率高斯分支

这是理解整个双分支结构的关键问题。

### 9.1 高分辨率分支负责什么

高分辨率分支更适合承载：

1. 精确空间位置
2. 局部边缘形状
3. 细小纹理的真实落点
4. 最终 RGB 恢复所需的像素级载体

也就是说，它保留的是：

> 细节“长在哪里、怎么落到像素网格上”。

### 9.2 低分辨率高斯分支负责什么

低分辨率高斯分支更适合承载：

1. 局部结构包络
2. 方向性
3. 高频/纹理强弱
4. 纹理频率与相位趋势
5. 跨更大空间范围的紧凑先验

也就是说，它保留的是：

> 细节“应该是什么结构类型、该往哪个方向增强、该强调什么频率”。

### 9.3 为什么这样分工合理

因为 UHD restoration 里最贵的部分，不应该去表达所有细节本身，而应该去表达：

- 哪些地方值得投入恢复容量；
- 恢复应该偏向何种结构；
- 如何用小代价把这种结构先验传给高分辨率主干。

这正是 Gaussian 分支适合做的事。

## 10. 可解释性到底体现在哪里

这里必须讲得既积极又诚实。

### 10.1 我们的可解释性是“相对可解释”，不是“完全白盒”

这套架构不是传统物理模型，不可能做到每个像素都有人类语义解释。

但相对纯 dense feature map / dense attention，它显著更可解释，因为中间变量有明确语义：

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

- 该区域主要方向和频率结构

#### `affine_coeffs`

可解释为：

- 低分辨率先验如何调节高分辨率通道

#### `texture_features`

可解释为：

- 低分辨率分支所感知到的显式高频/纹理证据

### 10.3 我们为什么敢说它比普通 attention 更可解释

因为普通 attention 常常只能告诉你“模型看了哪里”，却很难告诉你：

- 为什么看这里；
- 看的是方向、频率还是纹理；
- 后续如何作用到通道恢复里。

而我们的 Gaussian 链条至少把这些问题拆开了：

1. 参数化先验是什么；
2. 渲染出的空间引导是什么；
3. 高分辨率高频增强如何发生；
4. 通道调制如何发生。

## 11. 训练设计为什么和架构绑得这么紧

`M03/A_k03` 不只是一个网络结构，它还有一个非常重要的训练协议配套：

- `use_train_inference_consistency = true`
- `train_context_size = 784`
- `train_supervision_crop_size = 512`

### 11.1 为什么不是直接用 512 patch 训练到底

因为我们已经明确知道，大图 restoration 里存在：

> patch train / full-image inference mismatch

只给模型看小 patch，它学到的行为可能和 full-image 推理时不一致。

### 11.2 为什么 context 比 supervision 大

当前训练设计是：

- 给模型更大的上下文窗口 `784`
- 但只对中心 `512` 区域做监督

这样做的目的：

1. 让模型在训练中看到更大空间上下文；
2. 同时控制显存和训练稳定性；
3. 减少纯 patch 训练带来的边界伪差和协议偏移。

所以你可以把这理解为：

> 架构负责跨分辨率信息流，训练协议负责减少 patch/full-image 行为错配，二者是配套设计。

## 12. 为什么 `M03/A_k03` 是可信里程碑

因为它不是一次“刚好跑出最高分”的偶然结果，而是满足了完整 acceptance 条件：

1. frozen weights 存在
2. code snapshot 存在
3. logs 存在
4. train/val/test metrics 存在
5. result_analysis 存在
6. previous baseline 对比明确

并且它有两个很重要的实验结论：

1. `A_k03 final.pt` 是当前冻结 official baseline，`26.199959289578075 dB`
2. best internal val checkpoint 不是 best official test checkpoint

这让它在汇报上更可信，因为它说明我们不是只靠 internal val 自我选择最佳点，而是承认 official post-hoc 才是最终接受标准。

## 13. 这套架构最容易被导师问的质疑，以及建议回答

下面这部分建议你直接背熟。

### 质疑 1：为什么非得用 Gaussian，普通 CNN/Transformer modulation 不行吗？

建议回答：

普通 modulation 当然也能做，但我们的目标不是“再做一个更强黑箱调制器”，而是要在 UHD 场景下找到一种更紧凑、更结构化、跨分辨率更自然的先验表达。Gaussian 的优势在于它既有几何结构，又能扩展进方向、频率和相位信息，因此更适合作为低分辨率紧凑参数场，再渲染回高分辨率引导。

### 质疑 2：为什么不直接在高分辨率上做 attention？

建议回答：

因为 UHD 场景下 full-resolution dense attention 的代价太高，而且不符合我们轻量化的核心目标。我们不是否认 attention 有效，而是把“高分辨率大规模学习 attention”改成“低分辨率学习高斯参数，再渲染高分辨率结构化引导”，本质上是在重写 attention 的生成方式。

### 质疑 3：为什么 attention_map 只有单通道，会不会表达力不够？

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

### 质疑 7：你的高斯真的在里程碑里起核心作用吗？还是只是附属模块？

建议回答：

在 `M03/A_k03` 里，高斯不是附属模块，而是在主信息流上：低分辨率参数头、Gaussian renderer、directional prior、high-pass modulation、affine modulation 都在主路径里。后续新增的 decoder guidance 或 feature refinement 才是高斯作用点的进一步延伸，不是高斯主线存在与否的前提。

### 质疑 8：为什么你们后来又去试 feature refinement、terminal residual，这是不是说明原架构不成立？

建议回答：

恰恰相反。正是因为 `M03/A_k03` 已经证明 Gaussian 主线成立，我们才有资格继续问“Gaussian 放在哪个层级最有效”。后续分支是在探索更优作用点，不是在补救主线不存在的问题。

### 质疑 9：这套架构最强在哪里，最弱又在哪里？

建议回答：

最强的是它给出了一个符合 UHD 条件的紧凑结构化建模路线，高斯既可解释又适合跨分辨率回注，并且已经形成可冻结 baseline。最弱的是绝对性能离更强 public reference 仍有差距，说明我们当前高斯作用方式虽然成立，但还没有完全打穿高分辨率细节表达瓶颈。

## 14. 这套架构当前最诚实的总结

如果你面对的是认真追问的导师，最稳妥的总结方式不是吹得很满，而是这样说：

1. 我们已经证明了 Gaussian 可以作为 UHD restoration 的核心结构化中间表示，而不是边缘模块。
2. 我们已经证明了“低分辨率紧凑参数化 + 高分辨率结构化回注”这条主线能形成稳定、可复现、可冻结的 strongest baseline。
3. 我们还没有证明当前版本已经达到这条主线的最终上限，因此后续才继续探索 Gaussian 的更强作用点。

换句话说：

> `M03/A_k03` 的价值不是“问题已经解决完了”，而是“我们已经找到了一个可信、可解释、可扩展的正确主线”。

## 15. 如果你要在汇报里画图，建议怎么画

建议画一张 6 步图，不要画太复杂：

1. 输入 `x`
2. 高分辨率 stem
3. `pixel_unshuffle` 低分辨率分支
4. Gaussian 参数头
5. Gaussian renderer + directional prior
6. high-pass modulation + low-rank affine + decoder + edge refiner

图上最值得标出来的四个点：

1. `lowres_scale = 4`
2. `atoms = 6, param_dim = 10`
3. `attention_map` 是高分辨率单通道结构化引导
4. `affine_rank = 6`

如果要再加一张解释图，最推荐的是：

1. 输入图
2. `attention_map`
3. `directional_prior` 中的方向分量
4. 输出图

这样最容易把“可解释性”讲出来。

## 16. 最后一句你可以直接背

如果导师只给你几十秒总结，你可以直接说：

> 我们的核心不是简单把高斯加进网络，而是把 2D Gaussian 变成了一种跨分辨率结构化先验：在低分辨率空间里用纹理/频率感知高斯原子做紧凑参数化建模，再把这些先验渲染成高分辨率 attention，并通过 high-pass 增强和低秩仿射调制回注到恢复主干里。这样既避免了 UHD 条件下昂贵的 full-resolution dense 建模，又尽量保留了细节恢复所需的高频表达能力。
