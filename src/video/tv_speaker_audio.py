"""CRT 电视扬声器音效 - FFmpeg 音频滤镜链。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TvSpeakerAudioConfig:
    enabled: bool = True
    highpass_hz: float = 200.0
    lowpass_hz: float = 4800.0
    mono: bool = True
    volume: float = 0.9
    compress: bool = True


def resolve_tv_speaker_audio_config(intro_cfg: dict) -> TvSpeakerAudioConfig:
    """从 intro 配置解析电视扬声器音效参数。"""
    raw = intro_cfg.get("tv_speaker_audio")
    if raw is False or raw == 0:
        return TvSpeakerAudioConfig(enabled=False)
    if raw is None:
        return TvSpeakerAudioConfig()
    if raw is True:
        return TvSpeakerAudioConfig(enabled=True)
    if not isinstance(raw, dict):
        raise ValueError("intro.tv_speaker_audio 须为 true | false | 对象")

    enabled = raw.get("enabled", True)
    if enabled is False or enabled == 0:
        return TvSpeakerAudioConfig(enabled=False)

    highpass = float(raw.get("highpass_hz", 200))
    lowpass = float(raw.get("lowpass_hz", 4800))
    if highpass <= 0 or lowpass <= highpass:
        raise ValueError(
            f"tv_speaker_audio 频带无效: highpass={highpass}, lowpass={lowpass}"
        )

    volume = float(raw.get("volume", 0.9))
    if not 0.0 < volume <= 2.0:
        raise ValueError(f"tv_speaker_audio.volume 须在 (0, 2]: {volume}")

    return TvSpeakerAudioConfig(
        enabled=True,
        highpass_hz=highpass,
        lowpass_hz=lowpass,
        mono=bool(raw.get("mono", True)),
        volume=volume,
        compress=bool(raw.get("compress", True)),
    )


def build_tv_speaker_audio_filter(cfg: TvSpeakerAudioConfig) -> str:
    """生成 FFmpeg 音频滤镜链（不含输入/输出标签）。"""
    parts = [
        f"highpass=f={cfg.highpass_hz:.0f}",
        f"lowpass=f={cfg.lowpass_hz:.0f}",
    ]
    if cfg.mono:
        parts.append("pan=mono|c0=0.5*c0+0.5*c1")
        parts.append("pan=stereo|c0=c0|c1=c0")
    if cfg.compress:
        parts.append(
            "acompressor=threshold=-20dB:ratio=3:attack=5:release=80:makeup=1"
        )
    parts.append(f"volume={cfg.volume:.3f}")
    parts.append("alimiter=limit=0.95")
    parts.append("aformat=sample_rates=48000:channel_layouts=stereo")
    return ",".join(parts)
