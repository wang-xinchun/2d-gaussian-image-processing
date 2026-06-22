#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

from PIL import Image


DATASET_ROOT = Path("/root/autodl-tmp/2D_Gaussian_Dataset/4KDehaze")
EXTRACTED_GT = DATASET_ROOT / "extracted" / "4KDehazing" / "groundtrues"
USABLE_ROOT = DATASET_ROOT / "usable"
USABLE_GT = USABLE_ROOT / "groundtrues"
VALID_CSV = USABLE_ROOT / "valid_images.csv"
CORRUPT_CSV = USABLE_ROOT / "corrupt_images.csv"
README = USABLE_ROOT / "README.md"


def image_info(path: Path) -> tuple[bool, tuple[int, int] | None, str | None]:
    try:
        with Image.open(path) as img:
            img.load()
            return True, img.size, None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"


def reset_usable_root() -> None:
    if USABLE_ROOT.exists():
        shutil.rmtree(USABLE_ROOT)
    USABLE_GT.mkdir(parents=True, exist_ok=True)


def relative_symlink(target: Path, link_path: Path) -> None:
    rel = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(valid_rows: list[dict[str, str]], corrupt_rows: list[dict[str, str]]) -> None:
    lines = [
        "# 4KDehaze 可用数据视图",
        "",
        "这份目录是从 `4KDehaze/extracted/4KDehazing/groundtrues` 扫描后生成的**可用视图**。",
        "",
        "处理原则：",
        "",
        "- 不修改原始分卷压缩包。",
        "- 不删除原始解压结果。",
        "- 只把可正常读取的图像链接到 `usable/groundtrues/`。",
        "- 把损坏文件单独记录到 `corrupt_images.csv`。",
        "",
        "## 当前结果",
        "",
        f"- 有效图像数：`{len(valid_rows)}`",
        f"- 损坏图像数：`{len(corrupt_rows)}`",
        f"- 原始扫描目录：`{EXTRACTED_GT}`",
        f"- 可用视图目录：`{USABLE_GT}`",
        "",
        "## 说明",
        "",
        "- 当前解压得到的内容是 `groundtrues`，即一组 UHD 清晰图像。",
        "- 这不等同于一个已经配对好的完整去雾 benchmark。",
        "- 但它已经可以作为一套可直接读取的 UHD 清晰图像集合使用。",
        "",
    ]

    if corrupt_rows:
        lines.extend(
            [
                "## 损坏文件",
                "",
            ]
        )
        for row in corrupt_rows:
            lines.append(f"- `{row['filename']}`: {row['error']}")
        lines.append("")

    README.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not EXTRACTED_GT.exists():
        print(f"missing_extracted_groundtruth_dir: {EXTRACTED_GT}", file=sys.stderr)
        return 1

    reset_usable_root()

    valid_rows: list[dict[str, str]] = []
    corrupt_rows: list[dict[str, str]] = []

    for image_path in sorted(EXTRACTED_GT.glob("*.jpg")):
        ok, size, error = image_info(image_path)
        if ok:
            link_path = USABLE_GT / image_path.name
            relative_symlink(image_path, link_path)
            valid_rows.append(
                {
                    "filename": image_path.name,
                    "path": str(image_path),
                    "width": str(size[0]),
                    "height": str(size[1]),
                }
            )
        else:
            corrupt_rows.append(
                {
                    "filename": image_path.name,
                    "path": str(image_path),
                    "error": error or "unknown_error",
                }
            )

    write_csv(
        VALID_CSV,
        valid_rows,
        ["filename", "path", "width", "height"],
    )
    write_csv(
        CORRUPT_CSV,
        corrupt_rows,
        ["filename", "path", "error"],
    )
    write_readme(valid_rows, corrupt_rows)

    print(f"valid_images={len(valid_rows)}")
    print(f"corrupt_images={len(corrupt_rows)}")
    print(f"usable_dir={USABLE_GT}")
    if corrupt_rows:
        print("corrupt_list:")
        for row in corrupt_rows:
            print(f"  {row['filename']}: {row['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
