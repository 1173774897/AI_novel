"""TTS rate/volume 百分比解析与叠加（config 基准 + 情感偏移）。"""

from __future__ import annotations

import re

_PERCENT_RE = re.compile(r"^\s*([+-]?)(\d+(?:\.\d+)?)\s*%\s*$")

# edge-tts 常见可用区间（保守钳制）
_MIN_PERCENT = -50.0
_MAX_PERCENT = 100.0


def parse_percent(value: str | None, *, default: float = 0.0) -> float:
    """将 ``+15%`` / ``-10%`` 解析为浮点百分比。"""
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    match = _PERCENT_RE.match(text)
    if not match:
        return default
    sign = -1.0 if match.group(1) == "-" else 1.0
    return sign * float(match.group(2))


def format_percent(value: float) -> str:
    """将浮点百分比格式化为 edge-tts 字符串。"""
    rounded = int(round(value))
    if rounded > 0:
        return f"+{rounded}%"
    if rounded < 0:
        return f"{rounded}%"
    return "+0%"


def combine_percent(base: str | None, delta: str | None) -> str:
    """基准百分比与偏移量相加，例如 ``+15%`` + ``+5%`` → ``+20%``。"""
    total = parse_percent(base) + parse_percent(delta)
    total = max(_MIN_PERCENT, min(_MAX_PERCENT, total))
    return format_percent(total)
