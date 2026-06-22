# 下一阶段实验设计（未部署版）

更新时间: 2026-05-06

## 文档定位

这份文档用于把当前项目的下一阶段实验推进到“可检查、可部署、但尚未部署”的状态。

本次设计严格停在以下边界：

- 可以写实验配置、gate、manifest、阶段目标和 run 顺序；
- 不做 smoke calibration；
- 不做 formal preflight；
- 不启动训练；
- 不生成 launch 进程。

## 一、当前为什么先设计而不部署

当前项目已经有两个关键信号：

1. `A01` 已经证明 `V1/T01` 在 `UHD-LL_down4` 上能够端到端训练；
2. `A_a02` 已经证明同一主线可以迁移到 `UHD_haze`，但同时暴露出 4K JPEG / DataLoader 吞吐是实打实的部署风险。

因此，下一阶段实验不能只停留在“有个想法”，也不能直接跳到“马上启动”，而是应该先把 run 级设计补齐，再按门槛依次过：

1. run 身份和数据口径固定；
2. 评估目标固定；
3. runtime 目标固定；
4. smoke / preflight / launch 逐步通过。

## 二、当前推荐的实验顺序

### 2.1 立即准备的 run：`V1/T01/A03`

- 数据集：`UHD_deblur`
- 任务：UHD 去模糊
- 角色：建立 `UHD_deblur` 第一条可复现 baseline
- 当前状态：**设计已落地，未部署**

为什么优先是它：

1. `UHD_deblur` 是当前已验收的核心论文口径数据集之一；
2. 我们已经有低光和去雾两条线的正向信号，再补去模糊就能明显增强“跨 UHD restoration 任务可行”的论据；
3. 按当前阶段指标目标，`A03` 的第一任务不是追 SOTA，而是先建立可信 baseline。

### 2.2 A03 之后的分支

如果未来 `A03` 启动并给出正向结果，后续建议不是立刻乱开多条线，而是按下面顺序推进：

1. `A_b03` 或同类分支：
   在 `UHD_deblur` 上做同协议改进，目标是相对 `A03` baseline 稳定提升约 `0.5 dB`。
2. `A04` 控制实验：
   回到 `UHD-LL_down4` 做一轮更干净的 protocol-control replay，用于补强可比性和汇报稳定性。
3. `UHD_haze` 优化分支：
   如果继续推进去雾，则应优先处理数据管线吞吐，再做高显存版本重跑，而不是直接堆 batch。

## 三、A03 的设计原则

`A03` 当前采用的是一个**保守、可比较、便于解释**的设计原则：

1. 先不改模型结构，继续使用当前 `V1` 默认参数。
2. 先不改损失混合，继续使用 `Charbonnier + small gradient loss`。
3. 先不追求最大 GPU 占用，而是优先保证 run 能进入稳定 validation/checkpoint 闭环。
4. 先建立 baseline，再谈更强调参。

这意味着 `A03` 更像是：

> 一个面向 `UHD_deblur` 的任务迁移基线实验，
> 而不是一上来就做复杂 loss、复杂 schedule、复杂结构的大改版实验。

## 四、A03 当前已固定的设计内容

### 4.1 数据与 split

- train: `UHD_deblur/train/input_new -> gt_new`
- internal val: 从 train 固定切出前 `100` 对
- effective train: 剩余 `1864` 对
- official test: `UHD_deblur/test/input300 -> gt300`
- official test image count: `300`

注意事项：

- `gt300` 目录里有少量非图片元数据文件；
- 当前 dataloader 会按图片扩展名过滤；
- 因此评估口径仍然是 `300` 对图像。

### 4.2 运行目标

当前 `A03` 设计值为：

- tier: `short`
- wall clock cap: `58 min`
- max steps: `2000`
- batch size: `16`
- workers: `8`
- train crop: `512`
- val crop: `512`
- val/checkpoint interval: `250`

这组值当前只是**待校准的初始设计值**，不是最终 launch 值。它们来源于 `A_a02` 的保守可行包络，用来减少跨任务初次设计的变量数量。

### 4.3 目标解释

按当前阶段指标目标，`A03` 的成功标准分三层：

1. 能稳定进入 train / validation / checkpoint 闭环；
2. 能建立第一条可复现 `UHD_deblur` official baseline；
3. 这个 baseline 足够稳定，值得后续分支去追 `+0.5 dB`。

## 五、当前已经落地的文件

本轮已经把下面这些文件准备好：

- `experiments/V1/templates/T01/configs/a03_short_uhddeblur_patch.toml`
- `experiments/V1/runs/A03/configs/a03_short_uhddeblur_patch.toml`
- `experiments/V1/runs/A03/gate_check.md`
- `experiments/V1/runs/A03/manifest.md`

这些文件的含义是：

- 模板配置已经固定；
- run 身份已经固定；
- gate 检查项已经展开；
- manifest 已经明确写出目的、split、runtime 目标、未完成事项。

## 六、当前刻意没有做的事情

为了符合“先设计、不部署”的边界，本轮**刻意没有做**下面这些事情：

1. 没有运行 `smoke_calibrate_v1_t01.py`
2. 没有运行 `train_v1_t01.py --preflight`
3. 没有创建 `launch_tmux.sh`
4. 没有启动任何训练 session
5. 没有生成 checkpoint / metric / log

## 七、下一步如何从设计切到部署

等你允许进入部署阶段时，最自然的顺序就是：

1. 先对 `A03` 运行 config 语法检查；
2. 对 `UHD_deblur` 做 data-inclusive smoke calibration；
3. 根据 smoke 结果修正 batch / workers / max_steps；
4. 对最终配置做 formal preflight；
5. 如果 gate 全部通过，再生成 launch 脚本并启动。

这样做的好处是，我们不会再重演原始 `A02` 那种“GPU 看上去能装下，但真实训练吞吐无效”的问题。
