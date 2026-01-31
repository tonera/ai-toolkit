#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量重命名数据集图片（并同步重命名同名 caption .txt）。

目标：把乱七八糟的文件名统一成类似：
  sakimi_0001.jpg / sakimi_0002.png ...

特性：
- 两阶段重命名（先改到临时名再改到最终名），避免重名覆盖
- 支持 dry-run 预览
- 支持递归子目录（默认不开）

用法示例：
  # 先预览
  python3 scripts/rename_dataset_files.py --dataset /Users/zhangtao/training/sakimi --prefix sakimi --dry-run

  # 真正执行（从 1 开始，4 位补零）
  python3 scripts/rename_dataset_files.py --dataset /Users/zhangtao/training/sakimi --prefix sakimi --start 1 --pad 4
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(dataset_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in dataset_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in dataset_dir.iterdir() if p.is_file()]

    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    # 排序策略：按相对路径排序，保证可重复
    images.sort(key=lambda p: str(p.relative_to(dataset_dir)).lower())
    return images


def build_pairs(images: Iterable[Path]) -> List[Tuple[Path, Path | None]]:
    pairs: List[Tuple[Path, Path | None]] = []
    for img in images:
        txt = img.with_suffix(".txt")
        pairs.append((img, txt if txt.exists() else None))
    return pairs


def format_name(prefix: str, idx: int, pad: int, ext: str) -> str:
    return f"{prefix}_{idx:0{pad}d}{ext}"


def rename_path(src: Path, dst: Path, dry_run: bool) -> None:
    if src == dst:
        return
    if dst.exists():
        raise FileExistsError(f"目标文件已存在，停止以避免覆盖: {dst}")
    if dry_run:
        return
    src.rename(dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="数据集目录（图片与 txt 同目录）")
    parser.add_argument("--prefix", type=str, default="image", help="新文件名前缀")
    parser.add_argument("--start", type=int, default=1, help="起始编号")
    parser.add_argument("--pad", type=int, default=4, help="补零位数")
    parser.add_argument("--recursive", action="store_true", help="递归子目录（默认只处理当前目录）")
    parser.add_argument("--dry-run", action="store_true", help="只预览不改名")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset 目录不存在: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"dataset 不是目录: {dataset_dir}")

    images = iter_images(dataset_dir, recursive=args.recursive)
    if not images:
        print(f"未找到图片文件（支持 {sorted(IMAGE_EXTS)}）: {dataset_dir}")
        return 1

    # 生成最终目标路径（保持原扩展名）
    pairs = build_pairs(images)
    planned = []
    idx = args.start
    for img, txt in pairs:
        new_img = img.with_name(format_name(args.prefix, idx, args.pad, img.suffix.lower()))
        new_txt = new_img.with_suffix(".txt") if txt is not None else None
        planned.append((img, new_img, txt, new_txt))
        idx += 1

    # 冲突检查：最终目标路径不能重复、且不能覆盖现有其他文件
    final_targets = [p[1] for p in planned] + [p[3] for p in planned if p[3] is not None]
    final_targets_nonnull = [p for p in final_targets if p is not None]
    if len(set(final_targets_nonnull)) != len(final_targets_nonnull):
        raise RuntimeError("生成的目标文件名发生重复，请检查 prefix/pad/start 或目录结构")

    for src_img, dst_img, src_txt, dst_txt in planned:
        # 若目标就是自己，允许（不做事）；否则禁止覆盖
        if dst_img != src_img and dst_img.exists():
            raise FileExistsError(f"目标图片已存在，停止以避免覆盖: {dst_img}")
        if src_txt is not None and dst_txt is not None and dst_txt != src_txt and dst_txt.exists():
            raise FileExistsError(f"目标 txt 已存在，停止以避免覆盖: {dst_txt}")

    # 打印预览（前 20 条）
    print(f"数据集: {dataset_dir}")
    print(f"图片数量: {len(images)}")
    print(f"模式: {'dry-run' if args.dry_run else '执行改名'}")
    preview_n = min(20, len(planned))
    for i in range(preview_n):
        src_img, dst_img, src_txt, dst_txt = planned[i]
        rel_src = src_img.relative_to(dataset_dir)
        rel_dst = dst_img.relative_to(dataset_dir)
        print(f"- {rel_src}  ->  {rel_dst}")
        if src_txt is not None and dst_txt is not None:
            print(f"  {src_txt.relative_to(dataset_dir)}  ->  {dst_txt.relative_to(dataset_dir)}")
    if len(planned) > preview_n:
        print(f"... 还有 {len(planned) - preview_n} 条未显示")

    # 两阶段改名：先到临时名，避免 A->B、B->A 之类互相覆盖
    salt = uuid.uuid4().hex[:8]
    tmp_planned = []
    for n, (src_img, dst_img, src_txt, dst_txt) in enumerate(planned, start=0):
        tmp_img = src_img.with_name(f".__tmp__{salt}__{n}{src_img.suffix.lower()}")
        tmp_txt = tmp_img.with_suffix(".txt") if src_txt is not None else None
        tmp_planned.append((src_img, tmp_img, src_txt, tmp_txt, dst_img, dst_txt))

    # stage 1: src -> tmp
    for src_img, tmp_img, src_txt, tmp_txt, _, _ in tmp_planned:
        rename_path(src_img, tmp_img, args.dry_run)
        if src_txt is not None and tmp_txt is not None:
            rename_path(src_txt, tmp_txt, args.dry_run)

    # stage 2: tmp -> final
    for _, tmp_img, _, tmp_txt, dst_img, dst_txt in tmp_planned:
        rename_path(tmp_img, dst_img, args.dry_run)
        if tmp_txt is not None and dst_txt is not None:
            rename_path(tmp_txt, dst_txt, args.dry_run)

    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

