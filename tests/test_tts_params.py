"""TTS 基准 + 情感偏移叠加测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tts.tts_params import combine_percent, format_percent, parse_percent
from src.tools.tts_tool import TTSTool

pytestmark = pytest.mark.signature


class TestTtsParams:
    def test_parse_percent_signed(self):
        assert parse_percent("+15%") == 15.0
        assert parse_percent("-10%") == -10.0
        assert parse_percent("+0%") == 0.0

    def test_combine_base_and_delta(self):
        assert combine_percent("+15%", "+5%") == "+20%"
        assert combine_percent("+15%", "+0%") == "+15%"

    def test_combine_negative_clamped(self):
        assert combine_percent("+10%", "-80%") == "-50%"

    def test_format_percent_zero(self):
        assert format_percent(0.0) == "+0%"


class TestTTSToolRateStacking:
    def test_run_combines_base_rate_with_emotion_delta(self, tmp_path):
        with patch("src.tts.tts_engine.TTSEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.synthesize.return_value = (tmp_path / "a.mp3", [])
            mock_engine_cls.return_value = mock_engine

            config = {
                "tts": {"voice": "zh-CN-YunxiNeural", "rate": "+15%", "volume": "+0%"},
                "subtitle": {"enabled": False},
            }
            tool = TTSTool(config)
            audio = tmp_path / "0000.mp3"
            srt = tmp_path / "0000.srt"

            tool.run("测试", audio, srt, rate="+5%", volume="+5%")

            tts_cfg = mock_engine_cls.call_args[0][0]
            assert tts_cfg["rate"] == "+20%"
            assert tts_cfg["volume"] == "+5%"

    def test_run_without_delta_uses_base_only(self, tmp_path):
        with patch("src.tts.tts_engine.TTSEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.synthesize.return_value = (tmp_path / "a.mp3", [])
            mock_engine_cls.return_value = mock_engine

            config = {
                "tts": {"voice": "zh-CN-YunxiNeural", "rate": "+15%", "volume": "+0%"},
                "subtitle": {"enabled": False},
            }
            tool = TTSTool(config)
            audio = tmp_path / "0000.mp3"
            srt = tmp_path / "0000.srt"

            tool.run("测试", audio, srt)

            tts_cfg = mock_engine_cls.call_args[0][0]
            assert tts_cfg["rate"] == "+15%"
