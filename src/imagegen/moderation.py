"""云端生图内容审核回退 — 敏感 prompt 软化。"""

from __future__ import annotations

import re

# 英文 prompt 中易触发平台审核的词（替换为中性表述）
_RISKY_EN = re.compile(
    r"\b("
    r"nude|naked|nsfw|porn|erotic|sexual|sex|intercourse|"
    r"gore|bloody|blood|corpse|dead body|mutilat|"
    r"abuse|assault|rape|violat|"
    r"explicit|obscene|hentai"
    r")\b",
    re.IGNORECASE,
)

_SAFE_SUFFIX = (
    ", cinematic illustration, PG-13, safe for work, "
    "no nudity, no violence, no gore, atmospheric mood, implied storytelling"
)


def soften_image_prompt(prompt: str, attempt: int = 0) -> str:
    """将 prompt 软化为更易通过云端审核的版本。

    attempt 0: 剔除敏感词 + 安全后缀
    attempt 1+: 退化为通用电影感空镜
    """
    if attempt >= 1:
        return (
            "A subtle cinematic establishing shot, film noir atmosphere, "
            "dramatic window light, urban interior, moody shadows, "
            "professional movie still, no people in focus"
            + _SAFE_SUFFIX
        )

    cleaned = _RISKY_EN.sub("", prompt)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")
    if not cleaned:
        cleaned = "A subtle cinematic scene with dramatic lighting"
    return cleaned + _SAFE_SUFFIX


def is_content_moderation_error(detail: str) -> bool:
    """判断 DashScope/同类 API 错误是否为内容审核拦截。"""
    markers = (
        "DataInspectionFailed",
        "inappropriate content",
        "Inappropriate",
        "内容审核",
        "违规",
    )
    lowered = detail.lower()
    return any(m.lower() in lowered for m in markers)
