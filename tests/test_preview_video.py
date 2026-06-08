"""预览合成测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.preview_video import collect_segment_assets, preview_workspace

pytestmark = pytest.mark.signature


def _touch(path: Path, size: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


class TestCollectSegmentAssets:
    def test_collect_first_two_segments(self, tmp_path):
        ws = tmp_path / "proj"
        for i in range(2):
            _touch(ws / "images" / f"{i:04d}.png")
            _touch(ws / "audio" / f"{i:04d}.mp3")
            _touch(ws / "subtitles" / f"{i:04d}.srt", size=10)

        images, audio_srt = collect_segment_assets(ws, 2)
        assert len(images) == 2
        assert len(audio_srt) == 2
        assert images[0].name == "0000.png"
        assert audio_srt[1]["audio"].name == "0001.mp3"
        assert audio_srt[0]["srt"].name == "0000.srt"

    def test_collect_raises_when_image_missing(self, tmp_path):
        ws = tmp_path / "proj"
        _touch(ws / "audio" / "0000.mp3")
        with pytest.raises(FileNotFoundError, match="分镜 0"):
            collect_segment_assets(ws, 1)

    def test_collect_raises_on_invalid_count(self, tmp_path):
        with pytest.raises(ValueError, match="count"):
            collect_segment_assets(tmp_path, 0)


class TestPreviewWorkspace:
    def test_preview_workspace_calls_assembler(self, tmp_path, monkeypatch):
        ws = tmp_path / "demo"
        _touch(ws / "images" / "0000.png")
        _touch(ws / "audio" / "0000.mp3")
        _touch(ws / "subtitles" / "0000.srt", size=10)

        called: dict = {}

        class FakeTool:
            def __init__(self, config):
                called["config"] = config

            def run(self, **kwargs):
                called.update(kwargs)
                return kwargs["output_path"]

        monkeypatch.setattr("src.tools.video_assemble_tool.VideoAssembleTool", FakeTool)

        out = tmp_path / "out" / "preview.mp4"
        cfg = {"project": {"default_output": "output"}, "video": {"resolution": [1080, 1920]}}
        result = preview_workspace(ws, cfg, count=1, output_path=out)

        assert result == out
        assert len(called["images"]) == 1
        assert called["workspace"] == ws
