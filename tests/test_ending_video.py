"""片尾合成测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ending_video import compose_ending
from src.video.intro_content import resolve_ending_config

pytestmark = pytest.mark.signature


def _touch(path: Path, size: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


class TestResolveEndingConfig:
    def test_defaults(self, tmp_path, monkeypatch):
        clip = tmp_path / "media" / "thanks.mp4"
        shutdown = tmp_path / "media" / "off.mp4"
        _touch(clip)
        _touch(shutdown)
        monkeypatch.setattr(
            "src.video.intro_content.resolve_project_root",
            lambda: tmp_path,
        )
        path, text, off = resolve_ending_config({})
        assert path == clip
        assert text == "故事讲完了，感谢收听。我们下次再见"
        assert off == shutdown

    def test_missing_clip_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.video.intro_content.resolve_project_root",
            lambda: tmp_path,
        )
        with pytest.raises(FileNotFoundError, match="片尾短片不存在"):
            resolve_ending_config({})

    def test_shutdown_disabled(self, tmp_path, monkeypatch):
        clip = tmp_path / "media" / "thanks.mp4"
        _touch(clip)
        monkeypatch.setattr(
            "src.video.intro_content.resolve_project_root",
            lambda: tmp_path,
        )
        _, _, off = resolve_ending_config({"shutdown_clip": False})
        assert off is None


class TestComposeEnding:
    def test_compose_ending_end_to_end(self, tmp_path, monkeypatch):
        ws = tmp_path / "demo"
        ws.mkdir()
        clip = tmp_path / "media" / "thanks.mp4"
        shutdown = tmp_path / "media" / "off.mp4"
        _touch(clip)
        _touch(shutdown)
        tts_calls: list[str] = []

        class FakeTTSTool:
            def __init__(self, config):
                pass

            def run(self, text, audio_path, srt_path, rate=None, volume=None):
                tts_calls.append(text)
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"\x00" * 100)
                srt_path.write_text("", encoding="utf-8")
                return audio_path, srt_path

        build_calls: list[tuple[Path, Path]] = []
        tv_calls: dict = {}
        append_calls: list[tuple[Path, Path]] = []

        def fake_build(video_path, audio_path, output_path, **kwargs):
            build_calls.append((video_path, audio_path))
            output_path.write_bytes(b"content")
            return output_path

        def fake_tv(content, output, frame_cfg):
            tv_calls["content"] = content
            output.write_bytes(b"tv")
            return output

        def fake_append(first, second, output_path, **kwargs):
            append_calls.append((first, second, kwargs.get("second_audio_volume")))
            output_path.write_bytes(b"final")
            return output_path

        def fake_prepend(input_path, output_path, **kwargs):
            output_path.write_bytes(b"lead")
            return output_path

        from src.video.intro_tv_frame import IntroFrameConfig, TvScreenRect
        from src.video.tv_speaker_audio import TvSpeakerAudioConfig

        fake_cfg = IntroFrameConfig(
            tv_frame_image=tmp_path / "tv-frame.png",
            tv_frame_size=(1280, 720),
            screen=TvScreenRect(0, 0, 100, 100),
            screen_ref=TvScreenRect(0, 0, 100, 100),
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
            content_fit="fill",
            use_screen_mask=True,
            tv_speaker_audio=TvSpeakerAudioConfig(enabled=False),
        )

        monkeypatch.setattr(
            "src.video.intro_content.resolve_project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr("src.tools.tts_tool.TTSTool", FakeTTSTool)
        monkeypatch.setattr(
            "src.video.intro_content.build_focus_tagline_clip", fake_build
        )
        monkeypatch.setattr(
            "src.video.intro_tv_frame.resolve_intro_frame_config",
            lambda cfg: fake_cfg,
        )
        monkeypatch.setattr(
            "src.video.intro_tv_frame.composite_content_in_tv_frame", fake_tv
        )
        monkeypatch.setattr(
            "src.video.intro_content.append_video_clip", fake_append
        )
        monkeypatch.setattr(
            "src.video.intro_content.prepend_black_lead", fake_prepend
        )

        cfg = {
            "video": {"resolution": [1920, 1080]},
            "intro": {},
            "ending": {},
            "tts": {},
            "subtitle": {},
        }
        out = compose_ending(ws, cfg)

        assert out == ws / "intro" / "ending.mp4"
        assert out.read_bytes() == b"final"
        assert tts_calls == ["故事讲完了，感谢收听。我们下次再见"]
        assert build_calls[0][0] == clip
        assert tv_calls["content"] == ws / "intro" / "ending_content.mp4"
        assert append_calls[0][0] == ws / "intro" / "ending_with_lead.mp4"
        assert append_calls[0][1] == shutdown
        assert append_calls[0][2] == 0.5

    def test_no_tv_frame_skips_composite(self, tmp_path, monkeypatch):
        ws = tmp_path / "demo"
        ws.mkdir()
        clip = tmp_path / "media" / "thanks.mp4"
        shutdown = tmp_path / "media" / "off.mp4"
        _touch(clip)
        _touch(shutdown)

        class FakeTTSTool:
            def __init__(self, config):
                pass

            def run(self, text, audio_path, srt_path, rate=None, volume=None):
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"a")
                srt_path.write_text("", encoding="utf-8")
                return audio_path, srt_path

        def fake_build(video_path, audio_path, output_path, **kwargs):
            output_path.write_bytes(b"content")
            return output_path

        append_calls: list[Path] = []

        def fake_append(first, second, output_path, **kwargs):
            append_calls.append(first)
            output_path.write_bytes(b"final")
            return output_path

        def fake_prepend(input_path, output_path, **kwargs):
            output_path.write_bytes(b"lead")
            return output_path

        monkeypatch.setattr(
            "src.video.intro_content.resolve_project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr("src.tools.tts_tool.TTSTool", FakeTTSTool)
        monkeypatch.setattr(
            "src.video.intro_content.build_focus_tagline_clip", fake_build
        )
        monkeypatch.setattr(
            "src.video.intro_tv_frame.resolve_intro_frame_config",
            lambda cfg: None,
        )
        monkeypatch.setattr(
            "src.video.intro_content.append_video_clip", fake_append
        )
        monkeypatch.setattr(
            "src.video.intro_content.prepend_black_lead", fake_prepend
        )

        out = compose_ending(ws, {"video": {}, "ending": {}, "tts": {}, "subtitle": {}}, tv_frame=False)
        assert append_calls[0] == ws / "intro" / "ending_with_lead.mp4"
        assert out == ws / "intro" / "ending.mp4"
