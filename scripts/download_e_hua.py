#!/usr/bin/env python3
"""下载《恶之花：暗黑困境中的觉醒和救赎》全 12 篇，各存为一个 txt。

来源: https://onehu.xyz/categories/恶之花.../
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote
from urllib.request import Request, urlopen

from src.onehu_fetch import fetch_article_text

BASE = "https://onehu.xyz"
CATEGORY_URL = (
    BASE
    + "/categories/%E6%81%B6%E4%B9%8B%E8%8A%B1%EF%BC%9A"
    "%E6%9A%97%E9%BB%91%E5%9B%B0%E5%A2%83%E4%B8%AD%E7%9A%84%E8%A7%89%E9%86%92%E5%92%8C%E6%95%91%E8%B5%8E/"
)
_SERIES_TITLE = "恶之花：暗黑困境中的觉醒和救赎"
_LINK_RE = re.compile(
    r'href="(/2024/[^"]+)"[^>]*>\s*<time>[^<]+</time>\s*'
    r'<div class="list-group-item-title">(\d+\.\s*[^<]+)</div>',
)
_CHAPTER_NUM_RE = re.compile(r"^(\d+)\.\s*")


def _list_chapters() -> list[tuple[int, str, str]]:
    req = Request(CATEGORY_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; AI_novel/1.0)"})
    with urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    items: list[tuple[int, str, str]] = []
    for path, title in _LINK_RE.findall(html):
        m = _CHAPTER_NUM_RE.match(title.strip())
        if not m:
            raise ValueError(f"无法解析章节号: {title!r}")
        num = int(m.group(1))
        name = title.split(".", 1)[1].strip()
        items.append((num, name, path))
    items.sort(key=lambda x: x[0])
    return items


def _heading_core(heading: str) -> str:
    return re.sub(r"^\d+\.\s*", "", heading).strip()


def _strip_article_header(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines:
        return "", ""
    heading = lines[0].strip()
    core = _heading_core(heading)
    body_lines = lines[1:]
    skip = {heading, core, _SERIES_TITLE}
    while body_lines and body_lines[0].strip() in skip:
        body_lines.pop(0)
    return heading, "\n".join(body_lines).strip()


def _safe_filename(num: int, name: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|，。！？\s]+', "", name)
    return f"{num:02d}-{safe}.txt"


def main(out_dir: Path | None = None) -> int:
    chapters = _list_chapters()
    if len(chapters) != 12:
        print(f"警告: 预期 12 篇，实际 {len(chapters)} 篇", file=sys.stderr)
    nums = [n for n, _, _ in chapters]
    if nums != list(range(1, len(nums) + 1)):
        print(f"警告: 章节序号不连续: {nums}", file=sys.stderr)

    root = out_dir or Path("input/恶之花")
    root.mkdir(parents=True, exist_ok=True)

    for num, name, path in chapters:
        url = BASE + path
        print(f"下载 {num}. {name} …", flush=True)
        raw = fetch_article_text(url)
        heading, body = _strip_article_header(raw)
        text = "\n".join(
            [
                _SERIES_TITLE,
                f"【{name}】",
                "",
                heading or f"{num}. {name}",
                "",
                body,
                "",
            ]
        ).strip() + "\n"
        out = root / _safe_filename(num, name)
        out.write_text(text, encoding="utf-8")
        complete = "（全文完）" in body
        print(f"  → {out} ({len(text)} 字, {'完整' if complete else '可能未完'})")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(main(target))
