# 模型代码框架

## 当前目标

这份代码框架不是论文最终实现，而是一个**可扩展、可前向跑通、和当前研究方向一致**的初版 PyTorch 骨架。

当前这版模型架构正式命名为 **`V1` 架构**，代表项目第一版可执行架构。训练版本命名规则见 `docs/versioning_policy.md`。

## 目录结构

- `src/uhd_gaussian/config.py`
  - 模型配置与高斯渲染配置。
- `src/uhd_gaussian/utils.py`
  - 输入 padding / 输出裁剪等基础工具。
- `src/uhd_gaussian/models/blocks.py`
  - 残差卷积基础块。
- `src/uhd_gaussian/models/frequency.py`
  - 纹理/频率描述子提取模块。
- `src/uhd_gaussian/models/gaussian.py`
  - 2D Gaussian 参数头与局部注意力渲染器。
- `src/uhd_gaussian/models/affine.py`
  - 低秩仿射调制模块。
- `src/uhd_gaussian/models/restorer.py`
  - 顶层模型 `CrossResolutionGaussianRestorer`。
- `scripts/smoke_test_model_framework.py`
  - 前向冒烟测试脚本。

## 当前已经落实的主线

这版代码已经落实了与研究方向基线一致的 4 个核心环节：

1. **低分辨率紧凑建模**
   - 输入先通过 `pixel_unshuffle` 进入低分辨率分支。

2. **纹理/频率感知特征**
   - 使用灰度、Sobel 梯度、梯度幅值、Laplacian 形成基础频率描述子。

3. **2D Gaussian 参数预测**
   - 在低分辨率网格上预测 `K` 个高斯原子参数。
   - 当前每个原子用 `10` 个标量描述，和我们前面讨论的方向保持一致。

4. **高分辨率仿射回注**
   - 先渲染高分辨率标量注意力图，再做高频残差加权与低秩通道仿射调制。

## 当前模型输入输出

输入：

- `x`: `[B, 3, H, W]`

默认输出：

- `restored`: `[B, 3, H, W]`

如果 `return_aux=True`，还会返回：

- `attention_map`
- `gaussian_params`
- `affine_coeffs`
- `lowres_features`
- `texture_features`

## 当前代码的定位

这版代码适合作为：

- 后续真实方法实现的起点
- 模块替换与消融的代码母体
- 训练/推理流水线继续搭建的基础

这版代码**还不是**：

- 论文最终模型
- 最优结构
- 高性能实现
- 完整训练框架

## 仍然保留为可替换组件的部分

- backbone 结构
- 频率感知模块形式
- 2D Gaussian 参数定义
- attention 渲染方式
- 仿射作用位置与形式
- 损失函数与训练流程

## 冒烟测试

在项目根目录运行：

```bash
python3 scripts/smoke_test_model_framework.py
```

如果输出 `smoke_test: ok`，说明当前代码框架至少已经能够完成一次前向传播，并且输入输出尺寸对齐。
