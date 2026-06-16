#!/usr/bin/env python3
"""下载《完美谋杀》7 个故事，各合并为一个 txt。

来源: https://onehu.xyz/categories/完美谋杀.../
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from src.onehu_fetch import fetch_article_text

BASE = "https://onehu.xyz"

STORIES: list[dict] = [
    {
        "file": "01-皮箱里的诡异面具.txt",
        "title": "皮箱里的诡异面具",
        "chapters": [
            "/2025/10/06/1%E7%9A%AE%E7%AE%B1%E9%87%8C%E7%9A%84%E8%AF%A1%E5%BC%82%E9%9D%A2%E5%85%B7%EF%BC%8C%E5%88%B0%E5%BA%95%E6%98%AF%E5%81%9A%E4%BB%80%E4%B9%88%E7%9A%84%EF%BC%9F",
            "/2025/10/07/2%E9%9A%BE%E4%BB%A5%E5%90%AF%E9%BD%BF%E7%9A%84%E8%81%8C%E4%B8%9A%EF%BC%9A%E5%90%83%E6%92%AD%E5%AE%A2",
            "/2025/10/07/3%E4%B8%8B%E6%B0%B4%E9%81%93%E9%87%8C%E6%83%8A%E7%8E%B0%E4%BA%BA%E4%BD%93%E7%BB%84%E7%BB%87",
            "/2025/10/07/4%E6%9C%89%E6%97%B6%E5%80%99%EF%BC%8C%E7%9C%9F%E7%9B%B8%E5%B0%B1%E5%9C%A8%E4%B8%80%E6%AD%A5%E4%B9%8B%E9%81%A5",
        ],
    },
    {
        "file": "02-案发现场有只狗.txt",
        "title": "案发现场有只狗",
        "chapters": [
            "/2025/10/08/7%E6%A1%88%E5%8F%91%E7%8E%B0%E5%9C%BA%E6%9C%89%E5%8F%AA%E7%8B%97%EF%BC%9F%E5%8F%AF%E8%B0%81%E9%83%BD%E6%B2%A1%E8%A7%81%E8%BF%87%E5%AE%83",
            "/2025/10/08/9%E4%BB%96%E7%BB%88%E4%BA%8E%E4%B8%8D%E5%B1%91%E5%86%8D%E4%BC%AA%E8%A3%85%E8%87%AA%E5%B7%B1",
            "/2025/10/08/10%E6%B0%B4%E8%90%BD%E7%9F%B3%E5%87%BA%EF%BC%8C%E4%BD%86%E7%9C%9F%E7%9B%B8%E6%B0%B8%E8%BF%9C%E9%82%A3%E6%A0%B7%E6%99%A6%E6%B6%A9%E3%80%81%E7%97%9B%E6%A5%9A",
            "/2025/10/09/11%E7%8A%AF%E7%BD%AA%E5%AB%8C%E7%96%91%E4%BA%BA%E4%B8%89%E5%B9%B4%E5%89%8D%E5%B0%B1%E6%AD%BB%E4%BA%86",
        ],
    },
    {
        "file": "03-不杀他会后悔一辈子.txt",
        "title": "不杀他，我会后悔一辈子",
        "chapters": [
            "/2025/10/09/13%E4%BA%8B%E5%AE%9E%E8%AF%81%E6%8D%AE%E4%BF%B1%E5%9C%A8%EF%BC%8C%E4%BD%86%E6%88%91%E7%9A%84%E5%B7%A5%E4%BD%9C%E8%BF%98%E6%B2%A1%E6%9C%89%E5%AE%8C",
            "/2025/10/09/14%E4%B8%8D%E6%9D%80%E4%BB%96%EF%BC%8C%E6%88%91%E4%BC%9A%E5%90%8E%E6%82%94%E4%B8%80%E8%BE%88%E5%AD%90",
        ],
    },
    {
        "file": "04-神秘的U盘.txt",
        "title": "那个神秘的 U 盘",
        "chapters": [
            "/2025/10/10/16%E9%82%A3%E4%B8%AA%E7%A5%9E%E7%A7%98%E7%9A%84%20U%20%E7%9B%98%E9%87%8C%E5%88%B0%E5%BA%95%E6%9C%89%E4%BB%80%E4%B9%88%EF%BC%9F",
            "/2025/10/10/17%E4%BF%A1%E6%81%AF%E7%9A%84%E6%BA%90%E5%A4%B4%E5%87%BA%E7%8E%B0%E4%BA%86%E4%B8%80%E4%B8%AA%E7%86%9F%E6%82%89%E7%9A%84%E5%90%8D%E5%AD%97",
        ],
    },
    {
        "file": "05-案子起源于意外死亡.txt",
        "title": "案子起源于一起意外死亡事故",
        "chapters": [
            "/2025/10/10/19%E6%A1%88%E5%AD%90%E8%B5%B7%E6%BA%90%E4%BA%8E%E4%B8%80%E8%B5%B7%E6%84%8F%E5%A4%96%E6%AD%BB%E4%BA%A1%E4%BA%8B%E6%95%85",
            "/2025/10/11/22%E6%98%AF%E6%97%B6%E5%80%99%E5%92%8C%E4%BB%96%E8%A7%81%E4%B8%AA%E9%9D%A2%E4%BA%86",
        ],
    },
    {
        "file": "06-神秘跟踪者.txt",
        "title": "故意暴露身份的神秘跟踪者",
        "chapters": [
            "/2025/10/11/25%E6%95%85%E6%84%8F%E6%9A%B4%E9%9C%B2%E8%BA%AB%E4%BB%BD%E7%9A%84%E7%A5%9E%E7%A7%98%E8%B7%9F%E8%B8%AA%E8%80%85",
            "/2025/10/11/26%E5%B0%B1%E8%BF%99%E4%B8%80%E6%AC%A1%EF%BC%8C%E8%BF%98%E6%98%AF%E5%87%BA%E4%BA%8B%E4%BA%86",
        ],
    },
    {
        "file": "07-她为什么坚持学美术.txt",
        "title": "她为什么非要坚持学美术",
        "chapters": [
            "/2025/10/12/27%E5%A5%B9%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9D%9E%E8%A6%81%E5%9D%9A%E6%8C%81%E5%AD%A6%E7%BE%8E%E6%9C%AF%EF%BC%9F",
            "/2025/10/12/28%E5%B9%B8%E5%A5%BD%E5%BD%93%E6%97%B6%E4%BB%96%E8%BF%BD%E6%B1%82%E7%9A%84%E4%B8%8D%E6%98%AF%E6%88%91",
        ],
    },
]

_CHAPTER_HEADING_RE = re.compile(
    r"^(\d+\.\s*.+?)\n\1\n完美谋杀",
    re.MULTILINE,
)


_SERIES_TITLE = "完美谋杀：一位老刑警笔下的 7 个真实重案故事"


def _heading_core(heading: str) -> str:
    return re.sub(r"^\d+\.\s*", "", heading).strip()


def _strip_chapter_header(text: str) -> tuple[str, str]:
    """返回 (章节标题, 正文)。"""
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


def merge_story(story: dict) -> str:
    parts = [
        _SERIES_TITLE,
        f"【{story['title']}】",
        "",
    ]
    for path in story["chapters"]:
        url = BASE + path
        raw = fetch_article_text(url)
        heading, body = _strip_chapter_header(raw)
        parts.extend([heading or path, "", body, ""])
    return "\n".join(parts).strip() + "\n"


def main(out_dir: Path | None = None) -> int:
    root = out_dir or Path("input/完美谋杀")
    root.mkdir(parents=True, exist_ok=True)
    for story in STORIES:
        out = root / story["file"]
        print(f"下载 {story['title']} …", flush=True)
        text = merge_story(story)
        out.write_text(text, encoding="utf-8")
        print(f"  → {out} ({len(text)} 字, {len(story['chapters'])} 节)")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(main(target))
