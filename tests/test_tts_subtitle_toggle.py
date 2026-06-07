"""字幕开关测试。"""

from pathlib import Path

import pytest


@pytest.mark.signature
def test_tts_tool_skips_srt_when_disabled(tmp_path):
    from unittest.mock import MagicMock, patch

    from src.tools.tts_tool import TTSTool

    config = {
        "tts": {"voice": "zh-CN-YunxiNeural"},
        "subtitle": {"enabled": False},
    }
    tool = TTSTool(config)
    audio_path = tmp_path / "0000.mp3"
    srt_path = tmp_path / "0000.srt"

    mock_engine = MagicMock()
    mock_engine.synthesize.return_value = (audio_path, [])

    with patch.object(tool, "_get_engine", return_value=mock_engine):
        with patch.object(tool, "_get_sub_gen") as mock_sub:
            tool.run("测试文本", audio_path, srt_path)
            mock_sub.return_value.generate_srt.assert_not_called()

    assert srt_path.exists()
    assert srt_path.read_text(encoding="utf-8") == ""
