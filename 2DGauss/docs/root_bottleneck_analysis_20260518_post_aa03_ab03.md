# Root Bottleneck Analysis After `A_aa03` And `A_ab03`

## Record Time

- Time: `2026-05-18 00:43:43 CST`

## One-Sentence Conclusion

当前真正的根本瓶颈，不是“缺一个新的 hint”，也不是“边缘不够强”或“decoder 自身统计不够多”，而是：

> **Gaussian 先验虽然已经被放到了正确的大位置上，但它还没有形成一个足够有效、足够选择性的高分辨率解码协调机制。**

换句话说，当前系统已经知道“Gaussian 应该作用在 feature space”，但还没有真正解决：

> **Gaussian 先验到底应该在什么区域、以什么置信度、通过什么共享表示，去协调 decoder 的高分辨率细节恢复。**

## 为什么这才是根本瓶颈

现有证据链已经把几个容易混淆的假设基本排除了。

### 1. 不是训练跑不动

- `A_y03` 能稳定完成 `5` 小时 medium training，并刷新冻结里程碑。
- `A_aa03` clean finish at `299.9259` minutes。
- `A_ab03` clean finish at `36.1174` minutes。

所以现在的障碍不是 runtime、显存、DataLoader 或训练闭环。

### 2. 不是 Gaussian 放错了大位置

`M04/A_y03` 的成功已经证明最核心的大方向是对的：

- Gaussian 不该直接去控 terminal RGB residual；
- Gaussian 放在 decoder feature space 确实更有效；
- 轻量 edge-weighted supervision 作为辅助 bias 也是合理的。

这意味着：

> 我们已经解决了“Gaussian 应不应该进入高分辨率恢复主线”的问题。

但我们还没有解决：

> 它应该如何**精确协调**高分辨率恢复。

### 3. 也不是“多一个 cue 就会更强”

后续两条关键实验正好把这个问题钉死了：

- `A_aa03`：给 feature refinement 加 decoder self-detail hint，short tier 有局部正信号，但 medium scale 不能超过 `M04/A_y03`。
- `A_ab03`：把显式 edge-aware prior 直接塞进 feature refinement，official test 反而掉到 `25.1723 dB`，明显低于 `A_z03=25.3857 dB`。

这两条 together 说明：

- decoder 自身统计不是那个决定性增量；
- 显式边缘 cue 直接注入也不是那个决定性增量；
- 问题不在于“还少哪一种局部信息”，而在于这些信息**如何被组织和选择**。

## 代码层面的根因

当前架构的高分辨率恢复链路是：

1. lowres branch 生成 Gaussian params 与 attention
2. highres stem 得到高分辨率 feature
3. `attention_map * highpass` 和 affine modulation 先做一次主调制
4. decoder blocks 做高分辨率解码
5. decoder guidance 再调一次
6. feature refinement 再调一次
7. 最后 residual head 输出 RGB 残差
8. edge refiner 再补一次 terminal detail

关键位置在 [restorer.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/restorer.py:140) 之后：

- decoder guidance 先作用在 `decoder_feature`
- feature refinement 再作用在 `decoder_feature`
- edge refiner 最后再作用于 `residual`

这条链路的根本问题是：

> **这些模块虽然都在“帮 decoder”，但它们彼此之间并没有形成一个共享的、可置信的、可分工的控制机制。**

### Decoder guidance 的局限

`GaussianGuidedDecoderModulator` 只看 prior，不看 decoder 当前状态
[decoder_guidance.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/decoder_guidance.py:35)。

这意味着它擅长做“先验驱动的整体调制”，但不擅长回答：

- decoder 现在到底哪里已经恢复好了；
- 哪里还缺细节；
- 当前 modulation 是不是该减弱。

### Feature refinement 的局限

`GaussianGuidedFeatureRefinement` 虽然会把 prior 和 `decoder_feature` 融合
[feature_refinement.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/feature_refinement.py:139)，
但它目前仍然是“每次追加一种 cue，再重新学一套 gate/scale/bias”。

这带来两个问题：

1. 新 cue 很容易只是增加局部表达，却没有改变真正的决策机制。
2. cue 之间没有明确的共享控制面，结果就是“信息更多”，但不一定“控制更准”。

### Edge refiner 的局限

`HighResolutionEdgeGuidedRefiner` 已经拥有很强的显式边缘描述子
[edge.py](/root/autodl-tmp/2DGauss/src/uhd_gaussian/models/edge.py:61)，
但它的位置在 terminal residual 之后，它更像“最后补一点边缘”。

它不是当前主恢复过程的统一协调器。

这解释了为什么：

- edge information 作为轻辅助 supervision 有用；
- 但直接把 edge descriptor 再塞进 feature refinement，并不会自动变成更好的主机制。

## 因此，真正卡住我们的是什么

真正卡住我们的，是下面这个更高一级的问题：

> **Gaussian 先验到 decoder 细节恢复之间，缺少一个“共享且选择性”的控制接口。**

这个接口至少要回答三件事：

1. 哪些区域值得继续用 Gaussian 去加压。
2. 哪些区域已经恢复得差不多，应该抑制额外干预。
3. decoder guidance、feature refinement、edge detail correction 之间谁负责哪一段，而不是各自重复做弱耦合调制。

所以从根上说，当前瓶颈不是“信息不足”，而是：

> **控制不够 selective，模块之间也不够 coordinated。**

## 这也解释了为什么现有实验会呈现现在的图景

### 为什么 `A_y03` 会成功

因为它只要求一个相对温和的目标：

- 把 Gaussian 放到 feature space；
- 配上轻一点的 edge bias；
- 给足够训练预算。

这足以形成当前最优 baseline。

### 为什么 `A_aa03` 不够

因为 decoder self-hint 只是多告诉模型一点“我现在的特征能量是什么样”，
但它没有改变模块之间的协调关系。

所以 short tier 会有局部增益，medium tier 不够。

### 为什么 `A_ab03` 反而变差

因为它把显式 edge cue 直接灌进 feature refinement，但没有同时解决：

- 这些 edge cue 何时可信；
- 与 decoder 当前恢复状态如何对齐；
- 与已有 edge refiner / decoder guidance 怎么分工。

结果就是 cue 更强了，但控制更乱了。

## 结论层判断

因此，当前最根本的瓶颈不是：

- 训练时长不够；
- 训练数据不够；
- 再加一个边缘模块；
- 再加一个 decoder hint；
- 再把高分辨率 cue 塞进 Gaussian refinement。

而是：

> **我们还没有把 Gaussian 变成一个“能和 decoder 当前状态协同决策”的高分辨率控制器。**

## 下一步应该遵守的原则

基于这个根因，后续实验不该再优先做：

- 新 cue 直塞 feature refinement；
- edge-aware 近邻变体；
- detail-hint 近邻变体；
- 单纯加大某个局部信号强度。

后续真正值得做的，应该是：

1. 让 Gaussian refinement 拥有更明确的 confidence / uncertainty 选择机制。
2. 让 decoder guidance 和 feature refinement 共享一套更统一的 prior control surface。
3. 让 edge 信息不再作为“又一个并列 cue”，而是只在必要时影响控制强度或区域选择。

## 最终判断

当前根本瓶颈可以压缩成一句最重要的话：

> **我们缺的不是更多先验，而是一个能让 Gaussian 先验和 decoder 当前恢复状态发生有效协同的轻量控制机制。**

这也是为什么下一步最优先的正交方向，不再是“再换一种 hint”，而应该是：

- `confidence-normalized refinement`
- `shared-prior coupled decoder-guidance / feature-refinement`

因为它们都直接对准“选择性控制”和“模块协调”这两个根因。
