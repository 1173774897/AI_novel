"""onehu.xyz 文章正文抓取。"""

from __future__ import annotations

import re
from html import unescape
from urllib.request import Request, urlopen

_ARTICLE_RE = re.compile(r"<article[^>]*>(.*?)</article>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_FOOTER_MARKERS = (
    "备案号:",
    "上一篇",
    "下一篇",
    "没错就是我",
    "×\n完美谋杀",
    "×\n恶之花",
)


def clean_article_text(text: str) -> str:
    """去掉 onehu 页脚、推荐链接等杂质。"""
    for marker in _FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def fetch_article_text(url: str, *, timeout: float = 60.0) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AI_novel/1.0)"})
    with urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    match = _ARTICLE_RE.search(html)
    if not match:
        raise ValueError(f"未找到 <article> 正文: {url}")
    body = unescape(_TAG_RE.sub("", match.group(1)))
    lines = [ln.strip() for ln in body.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    if not text.strip():
        raise ValueError(f"正文为空: {url}")
    return clean_article_text(text)
