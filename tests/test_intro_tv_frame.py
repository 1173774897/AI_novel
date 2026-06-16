"""电视机框片头合成测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.video.intro_tv_frame import (
    TvScreenRect,
    _build_content_scale_chain,
    _build_image_tv_frame_filter,
    composite_content_in_tv_frame,
    probe_stream_duration,
    resolve_intro_frame_config,
)
from src.video.tv_speaker_audio import TvSpeakerAudioConfig

pytestmark = pytest.mark.signature


def _frame_cfg(
    tv_frame: Path,
    *,
    content_fit: str = "fill",
    use_screen_mask: bool = True,
    tv_speaker_audio: TvSpeakerAudioConfig | None = None,
) -> "IntroFrameConfig":
    from src.video.intro_tv_frame import IntroFrameConfig

    return IntroFrameConfig(
        tv_frame_image=tv_frame,
        tv_frame_size=(1280, 720),
        screen=TvScreenRect(x=384, y=158, w=537, h=394),
        screen_ref=TvScreenRect(x=384, y=158, w=537, h=394),
        output_size=(1920, 1080),
        fps=30,
        codec="libx264",
        crf=18,
        pattern_path=None,
        pattern_audio_volume=0.0,
        screen_mask_path=None,
        screen_mask_corner_radius=20.0,
        screen_mask_feather=1.5,
        screen_edge_bow={"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0},
        content_fit=content_fit,  # type: ignore[arg-type]
        use_screen_mask=use_screen_mask,
        tv_speaker_audio=tv_speaker_audio or TvSpeakerAudioConfig(enabled=False),
    )


class TestResolveIntroFrameConfig:
    def test_enabled_with_tv_frame(self, tmp_path, monkeypatch):
        frame = tmp_path / "tv-frame.png"
        frame.write_bytes(b"fake")
        monkeypatch.setattr(
            "src.video.intro_tv_frame.resolve_project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.video.intro_tv_frame.probe_image_size",
            lambda p: (2193, 1233),
        )
        cfg = {
            "intro": {"enabled": True, "tv_frame": "media/tv-frame.png"},
            "video": {"resolution": [1920, 1080], "fps": 30, "codec": "libx264", "crf": 20},
        }
        (tmp_path / "media").mkdir()
        (tmp_path / "media" / "tv-frame.png").write_bytes(b"fake")

        result = resolve_intro_frame_config(cfg)
        assert result is not None
        assert result.tv_frame_image.name == "tv-frame.png"
        assert result.content_fit == "fill"


class TestBuildImageTvFrameFilter:
    def test_static_frame_with_mask(self):
        fc = _build_image_tv_frame_filter(
            narr_duration=15.0,
            fps=30,
            tw=1920,
            th=1080,
            screen=TvScreenRect(x=576, y=237, w=806, h=591),
            content_fit="fill",
            use_screen_mask=True,
        )
        assert "alphamerge" in fc
        assert "format=yuv420p[outv]" in fc


class TestCompositeContentInTvFrame:
    def test_ffmpeg_uses_tv_frame_image(self, tmp_path, monkeypatch):
        content = tmp_path / "content.mp4"
        content.write_bytes(b"x")
        output = tmp_path / "out.mp4"
        tv_frame = tmp_path / "tv-frame.png"
        tv_frame.write_bytes(b"png")
        captured: dict = {}

        monkeypatch.setattr(
            "src.video.intro_tv_frame.probe_stream_duration",
            lambda path, stream="a:0": 14.0,
        )

        def fake_run(cmd, description):
            captured["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"mp4")

        monkeypatch.setattr("src.video.intro_tv_frame._run", fake_run)

        composite_content_in_tv_frame(
            content, output, _frame_cfg(tv_frame)
        )

        cmd = captured["cmd"]
        assert str(tv_frame) in cmd
        assert cmd[cmd.index("-t") + 1].startswith("14.0")

    def test_applies_tv_speaker_audio_filter(self, tmp_path, monkeypatch):
        content = tmp_path / "content.mp4"
        content.write_bytes(b"x")
        output = tmp_path / "out.mp4"
        tv_frame = tmp_path / "tv-frame.png"
        tv_frame.write_bytes(b"png")
        captured: dict = {}

        monkeypatch.setattr(
            "src.video.intro_tv_frame.probe_stream_duration",
            lambda path, stream="a:0": 12.0,
        )

        def fake_run(cmd, description):
            captured["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"mp4")

        monkeypatch.setattr("src.video.intro_tv_frame._run", fake_run)

        composite_content_in_tv_frame(
            content,
            output,
            _frame_cfg(
                tv_frame,
                use_screen_mask=False,
                tv_speaker_audio=TvSpeakerAudioConfig(enabled=True),
            ),
        )

        fc = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        assert "highpass=f=200" in fc
        assert "[aout]" in fc
