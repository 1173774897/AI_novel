"""TTS 音色解析 — gender / voices 映射为 edge-tts voice ID。"""

from __future__ import annotations

from typing import Any

# 未在 config.yaml 提供 tts.voices 时的默认值
DEFAULT_TTS_VOICES: dict[str, str] = {
    "male": "zh-CN-YunxiNeural",
    "female": "zh-CN-XiaoxiaoNeural",
}

_VALID_GENDERS = frozenset(DEFAULT_TTS_VOICES)


def resolve_tts_voice(tts_cfg: dict[str, Any] | None) -> str:
    """根据 tts 配置解析 edge-tts voice ID。

    优先级:
      1. tts.voice — 显式指定（覆盖 gender）
      2. tts.gender + tts.voices — 按性别选音色
    """
    cfg = tts_cfg or {}

    explicit = cfg.get("voice")
    if explicit:
        return str(explicit)

    gender = str(cfg.get("gender", "male")).strip().lower()
    if gender not in _VALID_GENDERS:
        raise ValueError(
            f"tts.gender 必须是 {sorted(_VALID_GENDERS)} 之一，收到: {gender!r}"
        )

    voices = cfg.get("voices") or {}
    voice = voices.get(gender) or DEFAULT_TTS_VOICES[gender]
    return str(voice)


def apply_tts_voice(cfg: dict[str, Any]) -> None:
    """就地解析 tts.voice，供 load_config 与各 TTS 入口共用。"""
    tts = cfg.setdefault("tts", {})
    if not isinstance(tts, dict):
        raise ValueError("tts 配置必须是字典")
    tts["voice"] = resolve_tts_voice(tts)
