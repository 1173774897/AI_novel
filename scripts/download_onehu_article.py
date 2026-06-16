#!/usr/bin/env python3
"""从 onehu.xyz 文章页提取正文，保存为 UTF-8 文本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.onehu_fetch import fetch_article_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 onehu 文章正文")
    parser.add_argument("url", help="文章 URL")
    parser.add_argument("-o", "--output", required=True, help="输出 .txt 路径")
    args = parser.parse_args(argv)

    text = fetch_article_text(args.url)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"已保存 {len(text)} 字 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
