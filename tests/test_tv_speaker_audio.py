"""CRT 电视扬声器音效测试。"""
from __future__ import annotations

import pytest

from src.video.tv_speaker_audio import (
    TvSpeakerAudioConfig,
    build_tv_speaker_audio_filter,
    resolve_tv_speaker_audio_config,
)

pytestmark = pytest.mark.signature


class TestResolveTvSpeakerAudioConfig:
    def test_default_enabled(self):
        cfg = resolve_tv_speaker_audio_config({})
        assert cfg.enabled is True
        assert cfg.highpass_hz == 200

    def test_disabled_false(self):
        cfg = resolve_tv_speaker_audio_config({"tv_speaker_audio": False})
        assert cfg.enabled is False

    def test_custom_dict(self):
        cfg = resolve_tv_speaker_audio_config(
            {
                "tv_speaker_audio": {
                    "highpass_hz": 150,
                    "lowpass_hz": 4000,
                    "volume": 0.85,
                    "mono": False,
                }
            }
        )
        assert cfg.highpass_hz == 150
        assert cfg.mono is False

    def test_invalid_band_raises(self):
        with pytest.raises(ValueError, match="频带无效"):
            resolve_tv_speaker_audio_config(
                {"tv_speaker_audio": {"highpass_hz": 5000, "lowpass_hz": 4000}}
            )


class TestBuildTvSpeakerAudioFilter:
    def test_contains_bandpass_and_limiter(self):
        chain = build_tv_speaker_audio_filter(TvSpeakerAudioConfig())
        assert "highpass=f=200" in chain
        assert "lowpass=f=4800" in chain
        assert "acompressor" in chain
        assert "alimiter" in chain

    def test_mono_disabled_skips_mono_pan(self):
        chain = build_tv_speaker_audio_filter(
            TvSpeakerAudioConfig(mono=False, compress=False)
        )
        assert "pan=mono" not in chain
        assert "acompressor" not in chain
