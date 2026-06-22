# 解包数据集验收报告

## 验收时间

- 北京时间：`2026-04-24 08:48:13 CST`
- 数据目录：`/root/autodl-tmp/2D_Gaussian_Dataset`
- 本次对象：
  - `UHD-LL_down4.zip`
  - `4K-Rain13k.zip`
  - `8K/8K.zip` with `8K.z01` ... `8K.z06`

## 解包前检查

本次先检查了磁盘空间和工具：

- `/root/autodl-tmp` 可用空间：约 `108G`
- 可用解包/查看工具：
  - `/usr/bin/unzip`
  - `/root/miniconda3/bin/bsdtar`
  - `/usr/bin/zipinfo`
  - `/usr/bin/zip`

归档未压缩体积：

| 归档 | 压缩包大小 | 未压缩体积 | 条目数 | 处理决定 |
|---|---:|---:|---:|---|
| `UHD-LL_down4.zip` | 571,350,122 bytes | 570,560,277 bytes | 8,622 | 已解包 |
| `4K-Rain13k.zip` | 14,153,963,710 bytes | 14,604,886,378 bytes | 26,007 | 已解包 |
| `8K/8K.zip` split archive | 132,041,868,189 bytes total by manifest | 132,119,221,684 bytes | 23,747 | 未完整解包，空间不足 |

## 解包结果

### UHD-LL_down4

解包路径：

- `/root/autodl-tmp/2D_Gaussian_Dataset/UHD-LL_down4`

解包命令口径：

- 使用 `unzip -n`
- 排除了 `__MACOSX/*` 和 `*/.DS_Store`
- 不覆盖已有文件

验收结果：

| 子集 | input | gt | 成对数 | data_list | 缺失配对 | 图像坏文件 | 分辨率 |
|---|---:|---:|---:|---:|---:|---:|---|
| `training_set` | 2000 | 2000 | 2000 | 2000 | 0 | 0 | 960x540 |
| `testing_set` | 150 | 150 | 150 | 150 | 0 | 0 | 960x540 |

结论：

> `UHD-LL_down4` 已解包并通过验收，可作为 `UHD-LL` 的下采样成对训练/测试数据使用。

### 4K-Rain13k

解包路径：

- `/root/autodl-tmp/2D_Gaussian_Dataset/4K-Rain13k`

解包命令口径：

- 使用 `unzip -n`
- 不覆盖已有文件

验收结果：

| 子集 | input | target | 成对数 | 缺失 target | 缺失 input | 图像坏文件 | 分辨率 |
|---|---:|---:|---:|---:|---:|---:|---|
| `train` | 12500 | 12500 | 12500 | 0 | 0 | 0 | 3840x2160 |
| `test` | 500 | 500 | 500 | 0 | 0 | 0 | 3840x2160 |

结论：

> `4K-Rain13k` 已解包并通过验收，可作为 4K 去雨成对训练/测试数据使用。

## 8K 状态

8K 归档文件：

- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z01`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z02`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z03`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z04`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z05`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.z06`
- `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K.zip`

目录级验收结果：

| 子集 | HR | LR_X2 | LR_X3 | LR_X3_BD | LR_X3_DN | LR_X4 | LR_X8 | LR_X16 | 配对情况 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `train` | 2029 | 2029 | 2029 | 2029 | 2029 | 2029 | 2029 | 2029 | 文件名集合完全一致 |
| `test` | 937 | 937 | 937 | 937 | 937 | 937 | 937 | 937 | 文件名集合完全一致 |

初次验收时不能完整解包的原因：

1. `8K` 未压缩体积约 `132.1G`。
2. 解包前 `/root/autodl-tmp` 可用空间约 `108G`，解包 `UHD-LL_down4` 和 `4K-Rain13k` 后可用空间约 `94G`。
3. 在保留原始分卷包的前提下，磁盘空间不足以完整落地 8K。
4. 当前系统中 `unzip` 对 split ZIP 的直接流式读取支持不好，会出现 multi-part offset 警告；`bsdtar` 无法识别该分卷 ZIP。

初次结论：

> `8K` 目前只能给出“归档完整 + 目录级配对完整”的判断，不能给出“已完整解包可直接训练”的判断。

后续若要完整解包 `8K`，需要先满足至少一项条件：

- 释放超过 `132G` 的额外空间，建议预留 `150G+`；
- 或将 8K 转移到更大的磁盘；
- 或明确允许删除/迁移原始分卷归档后再解包；
- 或安装/使用可直接处理 split ZIP 的工具链，并确认不需要生成额外完整单包。

### 8K 解包更新

- 更新时间：`2026-04-24 09:18:23 CST`
- 新状态：硬盘扩容后，`8K` 已完成解包并通过验收。

处理流程：

1. 重新确认 `/root/autodl-tmp` 可用空间约 `608G`。
2. 使用 `zip -s 0` 将 split ZIP 转为临时单体包：
   `/root/autodl-tmp/2D_Gaussian_Dataset/8K/8K_unsplit_tmp.zip`
3. 从临时单体包解包到：
   `/root/autodl-tmp/2D_Gaussian_Dataset/8K/extracted/8K`
4. 完成验收后删除临时单体包。
5. 原始 split ZIP 分卷仍保留。

解包后文件统计：

| 子集 | HR | LR_X2 | LR_X3 | LR_X3_BD | LR_X3_DN | LR_X4 | LR_X8 | LR_X16 | 配对情况 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `train` | 2029 | 2029 | 2029 | 2029 | 2029 | 2029 | 2029 | 2029 | 文件名集合完全一致 |
| `test` | 937 | 937 | 937 | 937 | 937 | 937 | 937 | 937 | 文件名集合完全一致 |

图像抽样检查：

- `HR`: `7680x4320`, RGB
- `LR_X2`: `3840x2160`, RGB
- `LR_X3`, `LR_X3_BD`, `LR_X3_DN`: `2560x1440`, RGB
- `LR_X4`: `1920x1080`, RGB
- `LR_X8`: `960x540`, RGB
- `LR_X16`: `480x270`, RGB

结论：

> `8K` 已解包并通过验收，可作为 8K 超分辨率训练/测试数据使用。

## 当前训练可用性总表

| 数据集 | 当前状态 | 是否可直接训练 | 备注 |
|---|---|---|---|
| `UHD-LL` | 已解包、已验收 | 是 | 2000 train + 150 test，3840x2160 |
| `UHD-LL_down4` | 已解包、已验收 | 是 | 2000 train + 150 test，960x540 |
| `LOL-v1` | 已解包、已验收 | 是 | 485 train + 15 eval，600x400 |
| `LOL-v2` | 已解包、已验收 | 是 | 689 train + 100 test，600x400 |
| `4K-Rain13k` | 已解包、已验收 | 是 | 12500 train + 500 test，3840x2160 |
| `4KDehaze` | usable clean view 已建立 | 部分可用 | 2040 张 clean groundtruth；不是成对去雾数据 |
| `8K` | 已解包、已验收 | 是 | 2029 train + 937 test，HR 为 7680x4320，多倍率 LR |

## 最终结论

本次已经把此前未解包的数据推进到如下状态：

1. `UHD-LL_down4.zip`：**已解包，通过验收，可用。**
2. `4K-Rain13k.zip`：**已解包，通过验收，可用。**
3. `8K`：**已解包，通过验收，可用。**

对当前项目最有价值的新结果是：

> 除 `UHD-LL` 外，现在又多了两个可直接用于高分辨率监督训练的数据集：`4K-Rain13k` 和 `8K`。
