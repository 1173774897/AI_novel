"""片头内容拼接测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.video.intro_content import (
    fit_audio_to_duration,
    resolve_ending_config,
    resolve_focus_clip_config,
)

pytestmark = pytest.mark.signature


class TestResolveFocusClipConfig:
    def test_default_paths(self, tmp_path, monkeypatch):
        clip = tmp_path / "media" / "focus.mp4"
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b"v")
        monkeypatch.setattr(
            "src.video.intro_content.resolve_project_root",
            lambda: tmp_path,
        )
        path, text = resolve_focus_clip_config({})
        assert path is None
        assert text == "关掉杂念，故事开始咯"

        path, text = resolve_focus_clip_config(
            {"focus_clip": "media/focus.mp4", "focus_tagline": "开始"}
        )
        assert path == clip
        assert text == "开始"

    def test_disabled(self):
        path, text = resolve_focus_clip_config({"focus_clip": False})
        assert path is None


class TestResolveEndingConfig:
    def test_missing_shutdown_raises(self, tmp_path, monkeypatch):
        clip = tmp_path / "media" / "thanks.mp4"
        _touch(clip)
        monkeypatch.setattr(
            "src.video.intro_content.resolve_project_root",
            lambda: tmp_path,
        )
        with pytest.raises(FileNotFoundError, match="关电视片段不存在"):
            resolve_ending_config({})


def _touch(path: Path, size: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


class TestFitAudioToDuration:
    def test_stretches_audio(self, tmp_path, monkeypatch):
        inp = tmp_path / "in.mp3"
        out = tmp_path / "out.m4a"
        inp.write_bytes(b"x")

        def fake_probe(path, stream="a:0"):
            return 4.0

        def fake_run(cmd, **kwargs):
            out.write_bytes(b"fitted")

        monkeypatch.setattr(
            "src.video.intro_content.probe_media_duration", fake_probe
        )
        monkeypatch.setattr("src.video.intro_content.subprocess.run", fake_run)

        result = fit_audio_to_duration(inp, out, 2.0)
        assert result == out
        assert out.exists()

    def test_copy_when_close(self, tmp_path, monkeypatch):
        inp = tmp_path / "in.mp3"
        out = tmp_path / "out.m4a"
        inp.write_bytes(b"same")

        monkeypatch.setattr(
            "src.video.intro_content.probe_media_duration",
            lambda path, stream="a:0": 2.0,
        )

        result = fit_audio_to_duration(inp, out, 2.005)
        assert result.read_bytes() == b"same"


class TestAppendVideoClip:
    def test_second_audio_volume_in_filter(self, tmp_path, monkeypatch):
        first = tmp_path / "a.mp4"
        second = tmp_path / "b.mp4"
        out = tmp_path / "out.mp4"
        first.write_bytes(b"a")
        second.write_bytes(b"b")

        captured: dict = {}

        def fake_probe(path):
            return 1280, 720

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

        monkeypatch.setattr(
            "src.video.intro_content.probe_video_stream_size", fake_probe
        )
        monkeypatch.setattr("src.video.intro_content.subprocess.run", fake_run)

        from src.video.intro_content import append_video_clip

        append_video_clip(
            first, second, out, codec="libx264", crf=18, fps=30, second_audio_volume=0.5
        )
        filter_idx = captured["cmd"].index("-filter_complex") + 1
        assert "volume=0.5" in captured["cmd"][filter_idx]


class TestPrependBlackLead:
    def test_generates_black_and_silence(self, tmp_path, monkeypatch):
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        inp.write_bytes(b"v")
        captured: dict = {}

        monkeypatch.setattr(
            "src.video.intro_content.probe_video_stream_size",
            lambda path: (1920, 1080),
        )
        monkeypatch.setattr(
            "src.video.intro_content.subprocess.run",
            lambda cmd, **kwargs: captured.update({"cmd": cmd}),
        )

        from src.video.intro_content import prepend_black_lead

        prepend_black_lead(inp, out, duration=1.0, codec="libx264", crf=18, fps=30)
        assert "color=c=black:s=1920x1080" in " ".join(captured["cmd"])
        assert "anullsrc" in " ".join(captured["cmd"])

    def test_zero_duration_copies(self, tmp_path, monkeypatch):
        inp = tmp_path / "in.mp4"
        out = tmp_path / "out.mp4"
        inp.write_bytes(b"orig")

        from src.video.intro_content import prepend_black_lead

        result = prepend_black_lead(inp, out, duration=0, codec="libx264", crf=18, fps=30)
        assert result.read_bytes() == b"orig"
