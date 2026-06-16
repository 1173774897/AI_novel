"""TTS 音色 gender / voices 解析测试。"""

import pytest

from src.tts.voices import DEFAULT_TTS_VOICES, apply_tts_voice, resolve_tts_voice


@pytest.mark.signature
class TestTtsVoiceResolution:
    def test_male_gender_uses_male_voice(self):
        voice = resolve_tts_voice(
            {
                "gender": "male",
                "voices": {
                    "male": "zh-CN-YunxiNeural",
                    "female": "zh-CN-XiaoxiaoNeural",
                },
            }
        )
        assert voice == "zh-CN-YunxiNeural"

    def test_female_gender_uses_female_voice(self):
        voice = resolve_tts_voice(
            {
                "gender": "female",
                "voices": {
                    "male": "zh-CN-YunxiNeural",
                    "female": "zh-CN-XiaoxiaoNeural",
                },
            }
        )
        assert voice == "zh-CN-XiaoxiaoNeural"

    def test_explicit_voice_overrides_gender(self):
        voice = resolve_tts_voice(
            {
                "gender": "female",
                "voice": "zh-CN-YunjianNeural",
                "voices": {"male": "x", "female": "y"},
            }
        )
        assert voice == "zh-CN-YunjianNeural"

    def test_missing_voices_falls_back_to_defaults(self):
        assert resolve_tts_voice({"gender": "male"}) == DEFAULT_TTS_VOICES["male"]
        assert resolve_tts_voice({"gender": "female"}) == DEFAULT_TTS_VOICES["female"]

    def test_invalid_gender_raises(self):
        with pytest.raises(ValueError, match="tts.gender"):
            resolve_tts_voice({"gender": "other"})

    def test_apply_tts_voice_mutates_config(self):
        cfg = {
            "tts": {
                "gender": "female",
                "voices": {
                    "male": "zh-CN-YunxiNeural",
                    "female": "zh-CN-XiaoxiaoNeural",
                },
            }
        }
        apply_tts_voice(cfg)
        assert cfg["tts"]["voice"] == "zh-CN-XiaoxiaoNeural"

    def test_load_config_resolves_voice_from_gender(self):
        from src.config_manager import load_config

        cfg = load_config()
        assert cfg["tts"]["gender"] in ("male", "female")
        assert cfg["tts"]["voice"] == cfg["tts"]["voices"][cfg["tts"]["gender"]]

    def test_tts_engine_uses_female_when_configured(self):
        from src.tts.tts_engine import TTSEngine

        engine = TTSEngine(
            {
                "gender": "female",
                "voices": {
                    "male": "zh-CN-YunxiNeural",
                    "female": "zh-CN-XiaoxiaoNeural",
                },
            }
        )
        assert engine.voice == "zh-CN-XiaoxiaoNeural"
