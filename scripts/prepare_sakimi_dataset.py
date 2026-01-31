#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为风格 LoRA 训练准备 captions：
- 如果图片没有同名 .txt：创建一个
- 如果已有 .txt：确保包含指定触发词（并把触发词放在最前面）

默认数据集路径：/Users/zhangtao/training/sakimi
默认触发词（按你的要求）：
  - sakimi
  - Sakimichan
  - Sakimichan style
  - Sakimi style
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


DEFAULT_TRIGGERS = [
    "sakimi",
    "Sakimichan",
    "Sakimichan style",
    "Sakimi style",
]

# 适用于“单人/人像/全身偏多”的风格 LoRA：稳定、不容易写错
DEFAULT_HIGH_LEVEL_TAGS = [
    "solo",
    "detailed illustration",
    "sharp lineart",
    "smooth shading",
    "dramatic lighting",
    "high contrast",
    # 如果你确认几乎都是全身，可以额外加：
    # "full body",
]


def list_images(folder: Path) -> List[Path]:
    images: List[Path] = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix in IMAGE_EXTS:
            images.append(p)
    return sorted(images)


def parse_caption_tokens(text: str) -> List[str]:
    # AI-Toolkit captions 通常用逗号分隔；保留带空格的短语 token（例如 "Sakimichan style"）
    tokens = [t.strip() for t in text.strip().split(",")]
    return [t for t in tokens if t]


def format_caption_tokens(tokens: Iterable[str]) -> str:
    # 统一用 ", " 连接
    return ", ".join([t.strip() for t in tokens if t and t.strip()])


def ensure_triggers_first(existing: List[str], triggers: List[str]) -> List[str]:
    # 把 triggers 放最前，去重并保持其顺序
    seen = set()
    out: List[str] = []
    for t in triggers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    for t in existing:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def dedupe_preserve_order(tokens: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tokens:
        t = (t or "").strip()
        if not t:
            continue
        if t in seen:
            continue
        out.append(t)
        seen.add(t)
    return out


def build_new_tokens(
    existing_tokens: List[str],
    triggers: List[str],
    high_level_tags: List[str],
    overwrite: bool,
) -> List[str]:
    """
    目标顺序（稳定）：
      触发词... , 高层词... , 其余原 caption token...
    """
    triggers = dedupe_preserve_order(triggers)
    high_level_tags = dedupe_preserve_order(high_level_tags)

    if overwrite:
        return dedupe_preserve_order([*triggers, *high_level_tags])

    existing_tokens = dedupe_preserve_order(existing_tokens)
    trigger_set = set(triggers)
    hl_set = set(high_level_tags)
    other = [t for t in existing_tokens if t not in trigger_set and t not in hl_set]
    return dedupe_preserve_order([*triggers, *high_level_tags, *other])


def process_one(
    image_path: Path,
    triggers: List[str],
    high_level_tags: List[str],
    overwrite: bool,
    dry_run: bool,
) -> bool:
    txt_path = image_path.with_suffix(".txt")
    changed = False

    if txt_path.exists():
        old = txt_path.read_text(encoding="utf-8", errors="ignore")
        tokens = parse_caption_tokens(old)
        new_tokens = build_new_tokens(
            existing_tokens=tokens,
            triggers=triggers,
            high_level_tags=high_level_tags,
            overwrite=overwrite,
        )
        new = format_caption_tokens(new_tokens)
        if new.strip() != old.strip():
            changed = True
            if not dry_run:
                txt_path.write_text(new, encoding="utf-8")
    else:
        changed = True
        new_tokens = build_new_tokens(
            existing_tokens=[],
            triggers=triggers,
            high_level_tags=high_level_tags,
            overwrite=True,  # 新建时等价于覆盖：只写规范化内容
        )
        new = format_caption_tokens(new_tokens)
        if not dry_run:
            txt_path.write_text(new, encoding="utf-8")

    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Users/zhangtao/training/sakimi",
        help="数据集目录（含图片）",
    )
    parser.add_argument(
        "--triggers",
        type=str,
        nargs="*",
        default=DEFAULT_TRIGGERS,
        help="触发词（按逗号分隔 caption token，含空格的触发词请用引号包起来）",
    )
    parser.add_argument(
        "--high-level-tags",
        type=str,
        nargs="*",
        default=DEFAULT_HIGH_LEVEL_TAGS,
        help="自动补齐的高层词（含空格的词请用引号包起来）。传空字符串可清空。",
    )
    parser.add_argument(
        "--no-high-level-tags",
        action="store_true",
        help="关闭自动补齐高层词（只补齐触发词）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="直接覆盖重写 captions：每张图只写“触发词 + 高层词”，不保留原 caption 的其他内容",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印统计不写文件")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset 目录不存在: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"dataset 不是目录: {dataset_dir}")

    triggers = [t.strip() for t in args.triggers if t and t.strip()]
    if not triggers:
        raise ValueError("triggers 不能为空")

    high_level_tags = []
    if not args.no_high_level_tags:
        high_level_tags = [t.strip() for t in args.high_level_tags if t and t.strip()]

    images = list_images(dataset_dir)
    if not images:
        print(f"未找到图片（支持 .jpg/.jpeg/.png）: {dataset_dir}")
        return 1

    changed_count = 0
    for img in images:
        if process_one(
            img,
            triggers=triggers,
            high_level_tags=high_level_tags,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        ):
            changed_count += 1

    print(f"数据集: {dataset_dir}")
    print(f"图片数量: {len(images)}")
    print(f"写入/更新 caption 数量: {changed_count}" + ("（dry-run）" if args.dry_run else ""))
    print("触发词: " + " / ".join(triggers))
    if high_level_tags:
        print("高层词: " + " / ".join(high_level_tags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

