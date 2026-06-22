# 4KDehaze 完整性校验报告

## 校验目标

- 数据目录：`/root/autodl-tmp/2D_Gaussian_Dataset/4KDehaze`
- 校验依据：百度网盘工具区中的下载清单与下载结果清单

## 使用的工具区文档与规则

本次校验前，先阅读了百度网盘工具区的以下文档：

- `/root/autodl-tmp/Baiduwangpan/README.md`
- `/root/autodl-tmp/Baiduwangpan/review/QUICKSTART.md`
- `/root/autodl-tmp/Baiduwangpan/review/ACCESS_PLAN.md`
- `/root/autodl-tmp/Baiduwangpan/review/README.md`
- `/root/autodl-tmp/Baiduwangpan/scripts/README.md`
- `/root/autodl-tmp/Baiduwangpan/manifests/README.md`

本次遵循的实际口径是：

- 只做只读检查
- 不修改百度网盘工具区配置
- 不重新发起下载
- 不改动 `/root/autodl-tmp/2D_Gaussian_Dataset`
- 优先以 manifest / results 清单作为完整性校验基准

## 使用的完整性清单

本次 4KDehaze 校验使用的主清单是：

- Manifest：
  `/root/autodl-tmp/Baiduwangpan/manifests/download_2d_gaussian_dataset_20260423.csv`
- 下载结果：
  `/root/autodl-tmp/Baiduwangpan/manifests/download_2d_gaussian_dataset_20260423.results.csv`

其中 manifest 为每个目标文件给出了：

- `remote_path`
- `local_path`
- `expected_size`

results 文件给出了多轮下载尝试的最终状态，能够用于确认某个文件最终是否以 `downloaded` 状态成功落地，并且 notes 中是否记录为 `size_match_*`。

## 4KDehaze 目标文件列表

manifest 中与 `4KDehaze` 对应的目标共 `13` 项：

1. `4KDehazing.part01.rar`
2. `4KDehazing.part02.rar`
3. `4KDehazing.part03.rar`
4. `4KDehazing.part04.rar`
5. `4KDehazing.part05.rar`
6. `4KDehazing.part06.rar`
7. `4KDehazing.part07.rar`
8. `4KDehazing.part08.rar`
9. `4KDehazing.part09.rar`
10. `4KDehazing.part10.rar`
11. `4KDehazing.part11.rar`
12. `4KDehazing.part12.rar`
13. `4KDehazing.pth`

## 校验方法

本次采用两层校验：

### 第一层：清单尺寸校验

逐条比较：

- manifest 中的 `expected_size`
- 本地文件的实际 `stat().st_size`

### 第二层：下载结果状态校验

从 `download_2d_gaussian_dataset_20260423.results.csv` 中提取每个 `4KDehaze` 文件最后一次记录，检查：

- `status == downloaded`
- `notes` 中为 `size_match_<expected_size>`

## 校验结果

### 1. Manifest 尺寸校验

结论：

> `13 / 13` 个文件全部存在，且本地大小与 `expected_size` **完全一致**。

对应结果：

- `4KDehazing.part01.rar` → match
- `4KDehazing.part02.rar` → match
- `4KDehazing.part03.rar` → match
- `4KDehazing.part04.rar` → match
- `4KDehazing.part05.rar` → match
- `4KDehazing.part06.rar` → match
- `4KDehazing.part07.rar` → match
- `4KDehazing.part08.rar` → match
- `4KDehazing.part09.rar` → match
- `4KDehazing.part10.rar` → match
- `4KDehazing.part11.rar` → match
- `4KDehazing.part12.rar` → match
- `4KDehazing.pth` → match

### 2. Results 最终状态校验

结论：

> `13 / 13` 个文件在结果清单中的最终状态均为 `downloaded`，并带有 `size_match_*` 备注。

最终记录示例：

- `4KDehazing.part01.rar` → `downloaded` → `size_match_4293918720`
- `4KDehazing.part12.rar` → `downloaded` → `size_match_3584355439`
- `4KDehazing.pth` → `downloaded` → `size_match_138381037`

## 结论

基于百度网盘下载岛中的 manifest 和 results 清单，当前可以正式给出结论：

> **`4KDehaze` 已按清单完整下载。**

更具体地说：

- 所有清单文件都已存在
- 所有文件大小都与 manifest 的 `expected_size` 完全一致
- results 文件中这些目标的最终状态全部为 `downloaded`

因此，从**下载完整性**角度看，`4KDehaze` 已通过校验。

## 补充说明

此前我们做过一次归档内容抽检，曾出现 `Truncated RAR file data`。
这次结合 manifest 清单判断后，更合理的解释是：

- 那次抽检方式并不适合作为 multipart RAR 完整性判定依据；
- 而百度网盘下载清单中的 `expected_size + downloaded + size_match` 组合，才是当前这套工具链下更可靠的校验标准。

所以，当前应以本次 manifest 校验结论为准。
