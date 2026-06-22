#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path


DATASET_ROOT = Path(os.environ.get("TWODGAUSS_DATASET_ROOT", "/home/ldj/wxc/baiduwangpan/2D高斯数据集"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = PROJECT_ROOT / "datasets"
VIEW_ROOT = REGISTRY_ROOT / "view"

DATASET_META = {
    "4KDehaze": {
        "task": "dehazing",
        "note": "4KDehaze 当前含 inputs/groundtrues 同名配对，可作为本机配对去雾训练集；无官方 split，训练前使用项目生成的 deterministic local split。",
    },
    "8K": {
        "task": "super_resolution",
        "note": "8K 已解包验收，可用于超分辨率训练/验证；原始分卷归档仍保留。",
    },
    "LOL-v1": {
        "task": "low_light",
        "note": "LOL-v1 已验收为普通分辨率配对低光照数据集。",
    },
    "LOL-v2": {
        "task": "low_light",
        "note": "LOL-v2 已验收为普通分辨率配对低光照数据集。",
    },
    "UHD-LL": {
        "task": "low_light",
        "note": "UHD-LL 已验收为直接可训练的配对 UHD 低光照数据集。",
    },
    "UHD-LL_down4": {
        "task": "low_light",
        "note": "UHD-LL_down4 已验收为直接可训练的降采样配对低光照数据集，适合短训和训练管线验证。",
    },
    "4K-Rain13k": {
        "task": "deraining",
        "note": "4K-Rain13k 已验收为直接可训练的配对 4K 去雨数据集。",
    },
    "UHD4K (SR)": {
        "task": "super_resolution",
        "note": "UHD4K (SR)/4K 已解包验收，HR/LR 多倍率目录可用；当前 T01 restoration 模板不能直接训练 SR，需要 SR 专用模板。",
    },
    "UHD_haze": {
        "task": "uhd_dehazing",
        "note": "UHD_haze 已解包验收为配对 UHD 去雾数据集；本地 test split 为 231 对，较论文清单 230 多 1 对，训练/评估 manifest 需显式记录口径。",
    },
    "UHD_deblur": {
        "task": "uhd_deblurring",
        "note": "UHD_deblur 已解包验收为配对 UHD 去模糊数据集；图片计数按扩展名过滤，忽略少量 .DS_Store/.baiduyun.uploading.cfg 元数据文件。",
    },
}


def classify_dataset(path: Path) -> tuple[str, str]:
    files = [p for p in path.rglob("*") if p.is_file()]
    dirs = [p for p in path.rglob("*") if p.is_dir()]

    if path.name in {"UHD-LL", "UHD-LL_down4"}:
        train_input = path / "training_set" / "input"
        train_gt = path / "training_set" / "gt"
        test_input = path / "testing_set" / "input"
        test_gt = path / "testing_set" / "gt"
        if all(p.is_dir() for p in (train_input, train_gt, test_input, test_gt)):
            return "accepted_paired_trainable", "训练/测试 input-gt 配对目录齐全。"

    if path.name == "4K-Rain13k":
        train_input = path / "train" / "input"
        train_target = path / "train" / "target"
        test_input = path / "test" / "input"
        test_target = path / "test" / "target"
        if all(p.is_dir() for p in (train_input, train_target, test_input, test_target)):
            return "accepted_paired_trainable", "训练/测试 input-target 配对目录齐全。"

    if path.name == "UHD_haze":
        train_input = path / "train" / "input"
        train_gt = path / "train" / "gt"
        test_input = path / "test" / "input"
        test_gt = path / "test" / "gt"
        if all(p.is_dir() for p in (train_input, train_gt, test_input, test_gt)):
            return "accepted_paired_trainable", "训练/测试 input-gt 配对目录齐全。"

    if path.name == "UHD_deblur":
        train_input = path / "train" / "input_new"
        train_gt = path / "train" / "gt_new"
        test_input = path / "test" / "input300"
        test_gt = path / "test" / "gt300"
        if all(p.is_dir() for p in (train_input, train_gt, test_input, test_gt)):
            return "accepted_paired_trainable", "训练/测试 input-gt 配对目录齐全。"

    if path.name == "LOL-v1":
        train_input = path / "our485" / "input"
        train_gt = path / "our485" / "gt"
        test_input = path / "eval15" / "input"
        test_gt = path / "eval15" / "gt"
        if all(p.is_dir() for p in (train_input, train_gt, test_input, test_gt)) and files:
            return "accepted_paired_trainable", "our485/eval15 input-gt 配对目录齐全。"

    if path.name == "LOL-v2":
        train_input = path / "Train" / "input"
        train_gt = path / "Train" / "gt"
        test_input = path / "Test" / "input"
        test_gt = path / "Test" / "gt"
        if all(p.is_dir() for p in (train_input, train_gt, test_input, test_gt)) and files:
            return "accepted_paired_trainable", "Train/Test input-gt 配对目录齐全。"

    if path.name == "8K":
        train_hr = path / "train" / "HR"
        test_hr = path / "test" / "HR"
        if train_hr.is_dir() and test_hr.is_dir():
            return "accepted_sr_template_required", "8K 已解包，HR/LR 多倍率目录可用；需 SR 专用训练模板。"

    if path.name == "UHD4K (SR)":
        sr_root = path / "4K"
        lr_dirs = ["LR_X2", "LR_X3", "LR_X3_BD", "LR_X3_DN", "LR_X4", "LR_X8", "LR_X16"]
        train_hr = sr_root / "train" / "HR"
        test_hr = sr_root / "test" / "HR"
        if train_hr.is_dir() and test_hr.is_dir() and all(
            (sr_root / split / lr_dir).is_dir()
            for split in ("train", "test")
            for lr_dir in lr_dirs
        ):
            return "accepted_sr_template_required", "UHD4K (SR)/4K 已解包，HR/LR 多倍率目录可用；需 SR 专用训练模板。"

    if path.name == "4KDehaze":
        paired_input = path / "4KDehazing" / "inputs"
        paired_gt = path / "4KDehazing" / "groundtrues"
        if paired_input.is_dir() and paired_gt.is_dir():
            return "accepted_paired_local_split_required", "4KDehazing inputs/groundtrues 同名配对目录齐全；需要项目本地 split。"

    if len(files) > 0:
        return "available_files", "已存在可见文件内容。"

    if len(dirs) > 0:
        return "directory_only", "仅有目录，暂无文件。"

    return "unknown", "当前状态未知。"


def relative_link_target(path: Path, link_parent: Path) -> str:
    return os.path.relpath(path, start=link_parent)


def rebuild_view(entries: list[dict[str, str]]) -> None:
    if VIEW_ROOT.exists():
        shutil.rmtree(VIEW_ROOT)
    VIEW_ROOT.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        category_dir = VIEW_ROOT / entry["status"]
        category_dir.mkdir(parents=True, exist_ok=True)
        link_path = category_dir / entry["dataset_name"]
        target = Path(entry["absolute_path"])
        rel_target = relative_link_target(target, category_dir)
        link_path.symlink_to(rel_target)


def build_status_tsv(entries: list[dict[str, str]]) -> None:
    lines = [
        "\t".join(
            [
                "dataset_name",
                "task",
                "status",
                "file_count",
                "directory_count",
                "absolute_path",
                "summary",
            ]
        )
    ]
    for entry in entries:
        lines.append(
            "\t".join(
                [
                    entry["dataset_name"],
                    entry["task"],
                    entry["status"],
                    str(entry["file_count"]),
                    str(entry["directory_count"]),
                    entry["absolute_path"],
                    entry["summary"],
                ]
            )
        )
    (REGISTRY_ROOT / "status.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_readme(entries: list[dict[str, str]]) -> None:
    status_counts = Counter(entry["status"] for entry in entries)
    lines = [
        "# 数据集整理视图",
        "",
        "这份目录是对 `/root/autodl-tmp/2D_Gaussian_Dataset` 的**外部整理视图**。",
        "",
        "整理原则：",
        "",
        "- 不移动原始下载文件。",
        "- 不重命名原始下载文件。",
        "- 不解压正在下载的归档。",
        "- 仅在项目目录下建立状态清单和符号链接视图。",
        "",
        "## 生成结果",
        "",
        "- `status.tsv`: 当前数据集状态总表。",
        "- `view/`: 按状态分类的符号链接视图。",
        "",
        "## 当前状态计数",
        "",
    ]

    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: `{count}`")

    lines.extend(
        [
            "",
            "## 当前数据集概览",
            "",
        ]
    )

    for entry in entries:
        lines.append(f"### {entry['dataset_name']}")
        lines.append("")
        lines.append(f"- 任务：`{entry['task']}`")
        lines.append(f"- 状态：`{entry['status']}`")
        lines.append(f"- 文件数：`{entry['file_count']}`")
        lines.append(f"- 目录数：`{entry['directory_count']}`")
        lines.append(f"- 原始路径：`{entry['absolute_path']}`")
        lines.append(f"- 说明：{entry['summary']}")
        lines.append("")

    lines.extend(
        [
            "## 重新生成",
            "",
            "- 运行 `python3 scripts/build_dataset_registry.py`。",
            "",
        ]
    )

    (REGISTRY_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []
    for dataset_dir in sorted(
        p for p in DATASET_ROOT.iterdir() if p.is_dir() and p.name != "_archives"
    ):
        files = [p for p in dataset_dir.rglob("*") if p.is_file()]
        dirs = [p for p in dataset_dir.rglob("*") if p.is_dir()]
        status, auto_summary = classify_dataset(dataset_dir)
        meta = DATASET_META.get(dataset_dir.name, {})
        note = meta.get("note", "")
        summary = auto_summary if not note else f"{auto_summary} {note}"

        entries.append(
            {
                "dataset_name": dataset_dir.name,
                "task": meta.get("task", "unknown"),
                "status": status,
                "file_count": len(files),
                "directory_count": len(dirs),
                "absolute_path": str(dataset_dir),
                "summary": summary,
            }
        )

    build_status_tsv(entries)
    build_readme(entries)
    rebuild_view(entries)


if __name__ == "__main__":
    main()
