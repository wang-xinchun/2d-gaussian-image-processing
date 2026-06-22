# 绝对性能提升路线图（2026-05-14）

## 文档目的

这份文档用于在当前 `A_k03/M03` 基线和轻量化约束下，明确后续提升
绝对性能的研究优先级。

这里的“绝对性能”优先指：

- 当前统一 official benchmark 口径下的 `PSNR`
- 后续可逐步补齐的 `SSIM`
- 真实视觉质量相关的 `LPIPS`

但当前路线排序仍以 `official PSNR` 为主，因为这是现阶段项目里最完整、
最统一、最连续的比较轴。

## 当前主线位置

截至当前：

- `UHD_deblur` accepted baseline: `A_k03 final.pt = 26.2000 dB`
- `UHD-Blur` lightweight `90%` gate: `26.5743 dB`
- 剩余差距：约 `0.3743 dB`

现有证据还表明：

1. `A_i03` 的大收益来自 `TLC` 风格的 train-inference consistency 修复；
2. `A_j03` 说明简单把 context 从 `784` 增到 `864` 不是当前优先方向；
3. `A_k03` 说明更长训练仍有效，但“同模板继续拉长”的边际收益已明显
   小于那次结构性跃迁；
4. 因此下一步更值得押注的是轻量结构升级，而不是继续做近同构 replay。

## 一句话路线判断

> 接下来要提高绝对性能，最优先的不是“看更大”或“训更久”，而是让
> `Gaussian` 先验更直接、更有效地控制高分辨率重建。

## 高优先级

### P1. 轻量级 Gaussian-first 模板升级

优先级：最高

推荐方向：

- 让 Gaussian 派生出的 attention / directional / texture prior 更直接介入
  high-resolution reconstruction
- 优先改 decoder 或 final reconstruction path，而不是把主体 backbone 做重
- 新机制必须默认可关闭、旧 checkpoint 兼容、参数增量受控

当前第一候选：

- `A_l03`: Gaussian-guided decoder modulation

为什么排第一：

1. 这是和 `A_i03 -> A_k03` 主线最同向的升级；
2. 它延续的是已验证有效的“结构性修正”路线，而不是重复预算变量；
3. 它不依赖更重主干，符合轻量化约束；
4. 当前剩余差距只有 `0.3743 dB`，更像需要一次干净的结构补点，而不是
   粗暴增重。

执行建议：

1. 先做 static / smoke / VRAM gate
2. 再做一条中训或中长训验证
3. 仍以 official full-image test 作最终裁决

### P2. 强化 high-res 末端而不是扩大 low-res 主体

优先级：高

推荐方向：

- 更有效的 edge-guided refinement
- 更直接的 Gaussian prior 到 decoder / residual branch 的调制
- 更细粒度但受控的高频重建路径

不推荐的替代方向：

- 宽很多、更深很多的通用 backbone
- 让 Gaussian 退化成边缘提示模块

为什么排高：

- 当前瓶颈更像“已有先验没有被充分用进最终重建”，而不是 low-res 表达
  完全不够
- 这和我们目前所有正向增益的来源是一致的

### P3. 结构升级后配中长训，而不是只看短训

优先级：高

原因：

- `A_k03` 再次证明 final endpoint 往往优于中途 checkpoint
- 小 slice internal validation 不能稳定替代 official test
- 结构变量很容易被短训低估

执行建议：

- 短训只用于“能不能跑、方向是否明显错误”
- 真正判断是否替换 baseline，至少要给中训预算
- official test 始终保留为 post-hoc

## 中优先级

### P4. 在结构升级稳定后，再继续做预算延长

优先级：中

适用前提：

- 轻量结构升级已经证明有效
- 或者新结构信号不明显，需要拿长训确认

为什么不是第一优先级：

- `A_k03` 已经证明“继续拉长训练”是有效但不够的变量
- 再做一次近似的更长训练，大概率有收益，但更像补充项而不是最优主变量

正确用法：

- 作为“放大已验证结构增益”的第二阶段工具
- 不是当前第一轮筛选工具

### P5. 继续补高频 / 感知损失，但只能受控地加

优先级：中

可以探索：

- Laplacian / gradient loss 权重再校准
- 轻量的频域一致性或高频一致性约束
- 后续逐步引入 `LPIPS` 作为补充观测

为什么只排中：

- 这类变量通常能帮助细节，但不一定稳定转化为 official PSNR
- 在当前阶段，结构性主变量的优先级更高

## 低优先级

### P6. 再次把 context 直接做大

优先级：低

当前依据：

- `A_j03` 是明确负结果
- 至少在当前预算和模板下，`864` context 不是应当优先回去的方向

结论：

- 未来不是永远不能再碰更大 context
- 但在当前阶段，不应把它当成主线第一变量

### P7. 继续做更多 near-isomorphic replay

优先级：低

例如：

- 只改很小的训练时长
- 只改很小的 val 设置
- 只做近似重复的 context / batch 重排

为什么排低：

- 这些变量的信息密度开始下降
- 很容易消耗算力而不带来结构性推进

### P8. 更重 backbone 或明显增重的替代方案

优先级：最低

原因：

- 违背项目当前最核心的轻量化约束
- 会削弱 `2D Gaussian` 的主线辨识度
- 即使涨点，也不应作为当前优先主线

## 建议的推进顺序

### 第一阶段

目标：

- 验证 `A_l03` 这种 Gaussian-first 轻量结构升级是否真的有正信号

动作：

1. 完成 design-only -> gate-ready
2. 做显存窗口校准
3. 部署一条中训验证

### 第二阶段

目标：

- 若 `A_l03` 正向，则确认能否替换 `A_k03/M03`

动作：

1. 做 official full-image test
2. 若超过 `26.2000 dB`，再看是否逼近或跨过 `26.5743 dB`
3. 若正向但未完全过线，再决定是否给更长预算放大收益

### 第三阶段

目标：

- 若第一候选不足，再扩展到同类型轻量模板升级家族

候选方向：

- 更强但仍轻量的 decoder guidance
- 更细粒度的 Gaussian-to-residual modulation
- 更强但受控的 edge/high-frequency refinement

## 当前不建议的动作

当前不建议优先做：

1. 再开一条简单 `864 context` 主线
2. 再做一轮几乎同构的纯长训 replay 作为第一选择
3. 直接转向明显更重的主干
4. 把资源先分散到 general benchmark 同协议复现线

## 当前结论

当前提升绝对性能的正确主线应该是：

1. 继续坚持 `Gaussian-first + lightweight`；
2. 优先做 high-res reconstruction 末端的轻量结构升级；
3. 用中长训去验证结构增益，而不是只做短训筛选；
4. 预算延长是放大器，不再是当前第一主变量；
5. 更大 context 和更重 backbone 目前都不是优先路线。
