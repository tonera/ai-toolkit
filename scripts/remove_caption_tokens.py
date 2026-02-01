#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量从指定目录下的 .txt caption 中删除指定 token（按“逗号分隔 token”的 AI-Toolkit 风格）。

特点：
- 精确按 token 删除（例如删除 "wide hips" 不会误伤 "wide hips and ..." 这种更长 token）
- 自动清理多余的逗号、空格
- 支持递归子目录
- 支持 dry-run 预览改动数量

用法示例：
  python scripts/remove_caption_tokens.py --dir /path/to/dataset --tokens "wide hips" "narrow waist" --dry-run
  python scripts/remove_caption_tokens.py --dir /path/to/dataset --tokens "wide hips" "narrow waist"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_caption_tokens(text: str) -> List[str]:
    # captions 通常用逗号分隔；保留带空格的短语 token
    tokens = [t.strip() for t in (text or "").strip().split(",")]
    return [t for t in tokens if t]


def format_caption_tokens(tokens: Iterable[str]) -> str:
    return ", ".join([t.strip() for t in tokens if t and t.strip()])


def list_txt_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in root.rglob("*.txt") if p.is_file()])
    return sorted([p for p in root.glob("*.txt") if p.is_file()])


def remove_tokens_from_caption(
    caption: str,
    tokens_to_remove: List[str],
    ignore_case: bool,
) -> Tuple[str, bool]:
    original_tokens = parse_caption_tokens(caption)
    if not original_tokens:
        return caption, False

    remove_set = set()
    if ignore_case:
        remove_set = {t.casefold() for t in tokens_to_remove}
        kept = [t for t in original_tokens if t.casefold() not in remove_set]
    else:
        remove_set = set(tokens_to_remove)
        kept = [t for t in original_tokens if t not in remove_set]

    new_caption = format_caption_tokens(kept)
    changed = new_caption.strip() != (caption or "").strip()
    return new_caption, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="包含 .txt captions 的目录",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        nargs="+",
        required=True,
        help='要删除的 token（按逗号分隔的完整 token；含空格请加引号），如 "wide hips"',
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理子目录（默认不递归；建议开）",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="忽略大小写匹配 token",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计/预览，不写入文件",
    )
    args = parser.parse_args()

    root = Path(args.dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"不是目录: {root}")

    tokens_to_remove = [t.strip() for t in args.tokens if t and t.strip()]
    if not tokens_to_remove:
        raise ValueError("--tokens 不能为空")

    txt_files = list_txt_files(root, recursive=args.recursive)
    if not txt_files:
        print(f"未找到 .txt: {root}")
        return 1

    changed_files = 0
    scanned_files = 0

    for p in txt_files:
        scanned_files += 1
        old = p.read_text(encoding="utf-8", errors="ignore")
        new, changed = remove_tokens_from_caption(
            old,
            tokens_to_remove=tokens_to_remove,
            ignore_case=bool(args.ignore_case),
        )
        if changed:
            changed_files += 1
            if args.dry_run:
                # 仅打印简要预览，避免刷屏
                print(f"[DRY] {p}")
            else:
                p.write_text(new, encoding="utf-8")

    mode = "DRY-RUN" if args.dry_run else "APPLIED"
    print(f"{mode}: scanned={scanned_files}, changed={changed_files}, root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

