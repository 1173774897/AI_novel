"""配音与字幕共用断句规则。"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 中文/字母/数字 — 无任何可朗读字符则视为标点残片
_SPEAKABLE_RE = re.compile(r"[\w\u4e00-\u9fff]")

# 省略号「……」是句内停顿，不是句末；若当作断句点会产生 edge-tts 无法合成的单字符片段
_SENTENCE_END_OUTSIDE = frozenset("。！？!?")
# 单独成句时 edge-tts 无法合成的纯标点/引号片段
_ORPHAN_CHARS = frozenset("「」""''【】…、，；：.!?！。？.—–-")


def _is_orphan_fragment(text: str) -> bool:
    """是否为无法独立朗读的标点/引号残片（如连续问句后的「」、单独「……」）。"""
    stripped = text.strip()
    if not stripped:
        return True
    if not _SPEAKABLE_RE.search(stripped):
        return True
    if len(stripped) > 3:
        return False
    return all(ch in _ORPHAN_CHARS or ch.isspace() for ch in stripped)


def is_unspeakable_fragment(text: str) -> bool:
    """TTS 是否应跳过该片段（纯标点/引号，edge-tts 无法合成）。"""
    return _is_orphan_fragment(text)


def _merge_orphan_fragments(segments: list[str]) -> list[str]:
    """将孤立标点/引号并入相邻句子，避免 TTS 合成失败。"""
    merged: list[str] = []
    for seg in segments:
        if not seg.strip():
            continue
        if merged and _is_orphan_fragment(seg):
            merged[-1] += seg
        else:
            merged.append(seg)

    while len(merged) >= 2 and _is_orphan_fragment(merged[0]):
        merged[1] = merged[0] + merged[1]
        merged.pop(0)

    # 整段仅余标点残片（如分段边界上的单独「】」）直接丢弃
    if len(merged) == 1 and _is_orphan_fragment(merged[0]):
        return []
    return merged


@dataclass
class _QuoteState:
    corner: bool = False
    ascii_dq: bool = False
    curly: bool = False
    wechat: bool = False  # 【】群聊/微信消息框

    @property
    def inside(self) -> bool:
        return self.corner or self.ascii_dq or self.curly or self.wechat


def _split_text_with_quotes(text: str, *, break_on_newline: bool) -> list[str]:
    """引号优先断句：「」/双引号闭合处先断句；引号内不因 。！？ 断开。"""
    segments: list[str] = []
    buf: list[str] = []
    qs = _QuoteState()

    def flush() -> None:
        seg = "".join(buf).strip()
        if seg:
            segments.append(seg)
        buf.clear()

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if break_on_newline and ch == "\n":
            flush()
            i += 1
            continue

        if ch == "「":
            buf.append(ch)
            qs.corner = True
            i += 1
            continue
        if ch == "」":
            buf.append(ch)
            qs.corner = False
            flush()
            i += 1
            continue

        if ch == "【":
            buf.append(ch)
            qs.wechat = True
            i += 1
            continue
        if ch == "】":
            buf.append(ch)
            qs.wechat = False
            flush()
            i += 1
            continue

        if ch == '"':
            buf.append(ch)
            qs.ascii_dq = not qs.ascii_dq
            if not qs.ascii_dq:
                flush()
            i += 1
            continue

        if ch == "\u201c":
            buf.append(ch)
            qs.curly = True
            i += 1
            continue
        if ch == "\u201d":
            buf.append(ch)
            qs.curly = False
            flush()
            i += 1
            continue

        if ch == "…":
            while i < n and text[i] == "…":
                buf.append(text[i])
                i += 1
            continue

        buf.append(ch)
        if not qs.inside and ch in _SENTENCE_END_OUTSIDE:
            flush()
            i += 1
            while i < n and text[i] in " \t":
                i += 1
            continue
        i += 1

    flush()
    return segments if segments else [text.strip()]


def split_sentences(text: str) -> list[str]:
    """段落内断句：引号闭合优先，其次句末标点（引号内不断句）。"""
    text = (text or "").strip()
    if not text:
        return []
    return _merge_orphan_fragments(_split_text_with_quotes(text, break_on_newline=False))


def split_utterances(text: str) -> list[str]:
    """按换行、引号闭合与句末标点断句（TTS/字幕）。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []
    return _merge_orphan_fragments(_split_text_with_quotes(text, break_on_newline=True))
