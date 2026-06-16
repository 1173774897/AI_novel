"""云端生图内容审核回退 — 敏感 prompt 软化。"""

from __future__ import annotations

import re

# 英文 prompt 中易触发平台审核的词（替换为中性表述）
_RISKY_EN = re.compile(
    r"\b("
    r"nude|naked|nsfw|porn|erotic|sexual|sex|intercourse|"
    r"gore|bloody|blood|bloodstain|corpse|dead body|mutilat|murder|killer|"
    r"stab|stabbing|axe|hatchet|decapitat|severed|gruesome|horrifying|"
    r"abuse|assault|rape|violat|terrifying|grotesque|menacing|"
    r"explicit|obscene|hentai|sinister smile|hollow eyes|lifeless eyes"
    r")\b",
    re.IGNORECASE,
)

# 中文角色卡/场景描述中易触发即梦拒稿的词
_CN_VIOLENT = re.compile(
    r"血[渍迹泊]?|杀[人死手]?|尸[体骨首]?|无头|滴血|断[肢头颈]|"
    r"鬼|恐怖|凶[恶狠煞]|刀|斧|刺|捅|眼白|凹陷|怪诞|残忍|流血|鲜血|"
    r"消防斧|菜刀|螺丝刀|追杀|嗜血|狞笑|劈|砍|尸横|残杀"
)

_SAFE_SUFFIX = (
    ", cinematic illustration, PG-13, safe for work, "
    "no nudity, no violence, no gore, atmospheric mood, implied storytelling"
)

_SUBTLE_HORROR_SUFFIX = (
    ", subtle psychological tension, implied unease, off-screen threat, "
    "no gore, no blood, no corpse"
)

# 即梦 dreamina API 对 prompt 长度敏感，超长易返回 ret=1046 InvalidNode
JIMENG_PROMPT_MAX_CHARS = 1200

_CAST_BLOCK_RE = re.compile(
    r"【(?:角色设定表|本段相关角色|本段出场角色)[^】]*】.*?(?=【|$)",
    re.DOTALL,
)

_MODERATION_SOFTEN_ATTEMPTS = 3
_MODERATION_REGEN_ATTEMPTS = 3

# 换角度重生 prompt 时注入 LLM / 本地模式的构图提示
_ALTERNATE_ANGLE_HINTS_CN = (
    "改用远景或环境空镜：强调场景与氛围，人物尽量小或不出镜，避免敏感动作特写。",
    "改用过肩、侧面或背影视角：避免正面人脸特写与直接暴力画面。",
    "改用隐喻构图：道具、光影、门框、剪影等含蓄表达，PG-13，无血腥暴力。",
)
_ALTERNATE_ANGLE_HINTS_EN = (
    "wide establishing shot, environment and mood, characters small or off-screen",
    "over-the-shoulder or side/back view, no frontal face close-up",
    "symbolic still life, silhouette, doorway light, implied storytelling, PG-13",
)

_SOFTEN_WIDER_SUFFIX = (
    ", wide establishing shot, environmental focus, "
    "characters small or off-screen, no explicit violence, PG-13"
)

_SOFTEN_MINIMAL_SUFFIX = (
    ", atmospheric wide shot, mood and setting only, "
    "no gore, no blood, no weapons in focus, implied storytelling, PG-13"
)

_MINIMAL_SAFE_PROMPTS = (
    "A dim apartment hallway at night, anime illustration, 2D cel shading, "
    "empty corridor, warm light under a door crack, moody atmosphere, no people visible",
    "A quiet urban interior at night, anime style illustration, soft window light, "
    "hand-drawn animation aesthetic, peaceful empty room, NOT photorealistic",
)


def _extract_scene_core(cleaned: str, max_chars: int = 350) -> str:
    """保留净化后场景描述的核心片段，避免过早退化为通用空镜。"""
    text = re.sub(r"\s+", " ", (cleaned or "").strip())
    if not text:
        return "A subtle cinematic scene with dramatic lighting"
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip(" ,") + "..."


def minimal_safe_fallback_prompt(variant: int = 0) -> str:
    """6 次重试均失败后的通用空镜保底 prompt（不含原场景语义）。"""
    idx = min(max(0, variant), len(_MINIMAL_SAFE_PROMPTS) - 1)
    return _MINIMAL_SAFE_PROMPTS[idx] + _SAFE_SUFFIX


def soften_image_prompt_for_attempt(base_prompt: str, soften_index: int) -> str:
    """第 soften_index 次软化（0=首次软化，1=二次，2=三次）。"""
    if soften_index <= 0:
        return truncate_image_prompt_for_jimeng(
            soften_image_prompt(base_prompt, attempt=0)
        )
    if soften_index == 1:
        return truncate_image_prompt_for_jimeng(
            soften_image_prompt(base_prompt, attempt=1)
        )
    return truncate_image_prompt_for_jimeng(
        soften_image_prompt(base_prompt, attempt=2)
    )


def alternate_angle_hint(variant: int) -> tuple[str, str]:
    """返回 (中文, 英文) 换角度构图提示。variant 为 0..2。"""
    i = max(0, min(variant, len(_ALTERNATE_ANGLE_HINTS_CN) - 1))
    return _ALTERNATE_ANGLE_HINTS_CN[i], _ALTERNATE_ANGLE_HINTS_EN[i]


def sanitize_image_prompt_text(prompt: str) -> str:
    """剔除卡司块与中英文敏感词，保留可生图的场景描述。"""
    text = _CAST_BLOCK_RE.sub(" ", prompt or "")
    text = _CN_VIOLENT.sub(" ", text)
    text = _RISKY_EN.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,")
    return text


def truncate_image_prompt_for_jimeng(
    prompt: str,
    max_chars: int = JIMENG_PROMPT_MAX_CHARS,
) -> str:
    """截断生图 prompt，优先保留英文场景与尾部风格约束。"""
    text = re.sub(r"\s+", " ", (prompt or "").strip())
    if len(text) <= max_chars:
        return text

    trimmed = text
    while len(trimmed) > max_chars:
        block = _CAST_BLOCK_RE.search(trimmed)
        if not block:
            break
        trimmed = (
            trimmed[: block.start()].rstrip(" ,")
            + " "
            + trimmed[block.end() :].lstrip(" ,")
        )
        trimmed = re.sub(r"\s+", " ", trimmed).strip(" ,")
        if len(trimmed) <= max_chars:
            return trimmed

    if len(trimmed) <= max_chars:
        return trimmed

    head_len = max(200, int(max_chars * 0.62))
    tail_len = max(120, max_chars - head_len - 5)
    if head_len + tail_len + 5 > max_chars:
        tail_len = max(80, max_chars - head_len - 5)
    return (
        trimmed[:head_len].rstrip(" ,")
        + " ... "
        + trimmed[-tail_len:].lstrip(" ,")
    )


def _is_length_error(error_detail: str) -> bool:
    lowered = (error_detail or "").lower()
    return "invalidnode" in lowered or "1046" in lowered


def _is_generation_failure(error_detail: str) -> bool:
    lowered = (error_detail or "").lower()
    return (
        is_content_moderation_error(error_detail)
        or "final generation failed" in lowered
        or "generation failed" in lowered
    )


def retry_image_prompt_after_failure(
    prompt: str,
    attempt: int,
    error_detail: str = "",
) -> str:
    """按失败类型缩短或软化 prompt 后重试。"""
    if _is_length_error(error_detail):
        limits = (JIMENG_PROMPT_MAX_CHARS, 900, 600, 400)
        base = soften_image_prompt(prompt, min(attempt, 1))
        return truncate_image_prompt_for_jimeng(
            base,
            max_chars=limits[min(attempt, len(limits) - 1)],
        )

    if _is_generation_failure(error_detail):
        if attempt == 0:
            cleaned = sanitize_image_prompt_text(prompt)
            if not cleaned:
                cleaned = (
                    "A tense first-person view through a peephole, "
                    "dim apartment hallway, cinematic anime style"
                )
            return truncate_image_prompt_for_jimeng(
                cleaned + _SUBTLE_HORROR_SUFFIX + _SAFE_SUFFIX,
                max_chars=1000,
            )
        if attempt == 1:
            return soften_image_prompt(prompt, 1)
        if attempt == 2:
            return soften_image_prompt(prompt, 2)
        return minimal_safe_fallback_prompt(min(attempt - 3, len(_MINIMAL_SAFE_PROMPTS) - 1))

    return soften_image_prompt(prompt, attempt)


def soften_image_prompt(prompt: str, attempt: int = 0) -> str:
    """将 prompt 软化为更易通过云端审核的版本。

    attempt 0: 剔除敏感词 + 安全后缀（保留场景语义）
    attempt 1: 保留场景核心 + 远景/环境化表述
    attempt 2: 进一步压缩场景核心 + 更强 PG-13 约束
    attempt 3+: 仅由 minimal_safe_fallback_prompt 用于最终保底
    """
    cleaned = sanitize_image_prompt_text(prompt)
    if not cleaned:
        cleaned = "A subtle cinematic scene with dramatic lighting"

    if attempt >= 2:
        core = _extract_scene_core(cleaned, max_chars=280)
        return core + _SOFTEN_MINIMAL_SUFFIX + _SAFE_SUFFIX

    if attempt >= 1:
        core = _extract_scene_core(cleaned, max_chars=450)
        return core + _SOFTEN_WIDER_SUFFIX + _SUBTLE_HORROR_SUFFIX + _SAFE_SUFFIX

    return cleaned + _SUBTLE_HORROR_SUFFIX + _SAFE_SUFFIX


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


def is_jimeng_retryable_error(detail: str) -> bool:
    """判断即梦 CLI 生图失败是否可通过软化 prompt 重试。"""
    if is_content_moderation_error(detail):
        return True
    markers = (
        "generation failed",
        "final generation failed",
        "aigccomplianceconfirmationrequired",
        "compliance",
        "content policy",
        "sensitive",
        "审核",
        "违规",
        "risk",
        "invalidnode",
        "1046",
    )
    lowered = detail.lower()
    return any(m in lowered for m in markers)
