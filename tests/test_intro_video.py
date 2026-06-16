"""片头合成测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.intro_video import (
    collect_intro_images,
    compose_intro,
    generate_intro_assets,
    split_intro_text,
)

pytestmark = pytest.mark.signature


def _touch(path: Path, size: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


class TestSplitIntroText:
    def test_split_by_chinese_period(self):
        text = "第一句。第二句！第三句？"
        assert split_intro_text(text) == ["第一句。", "第二句！", "第三句？"]

    def test_single_sentence_without_trailing_punct(self):
        assert split_intro_text("只有一句") == ["只有一句"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="不能为空"):
            split_intro_text("   ")


class TestCollectIntroImages:
    def test_default_first_n_images(self, tmp_path):
        ws = tmp_path / "proj"
        for i in range(3):
            _touch(ws / "images" / f"{i:04d}.png")
        imgs = collect_intro_images(ws, 2)
        assert [p.name for p in imgs] == ["0000.png", "0001.png"]

    def test_custom_indices(self, tmp_path):
        ws = tmp_path / "proj"
        _touch(ws / "images" / "0002.png")
        _touch(ws / "images" / "0005.png")
        imgs = collect_intro_images(ws, 2, indices=[2, 5])
        assert [p.name for p in imgs] == ["0002.png", "0005.png"]

    def test_missing_image_raises(self, tmp_path):
        ws = tmp_path / "proj"
        with pytest.raises(FileNotFoundError, match="0000.png"):
            collect_intro_images(ws, 1)


class TestGenerateIntroAssets:
    def test_calls_tts_per_segment(self, tmp_path, monkeypatch):
        calls: list[tuple[str, Path, Path]] = []

        class FakeTTSTool:
            def __init__(self, config):
                pass

            def run(self, text, audio_path, srt_path, rate=None, volume=None):
                calls.append((text, audio_path, srt_path))
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"\x00" * 100)
                srt_path.write_text("1\n", encoding="utf-8")
                return audio_path, srt_path

        monkeypatch.setattr("src.tools.tts_tool.TTSTool", FakeTTSTool)

        intro_dir = tmp_path / "intro"
        cfg = {"tts": {"voice": "x"}, "subtitle": {"enabled": True}}
        result = generate_intro_assets(["A。", "B。"], intro_dir, cfg)

        assert len(calls) == 2
        assert calls[0][0] == "A。"
        assert len(result) == 2
        assert result[0]["audio"].name == "0000.mp3"
        assert result[1]["srt"].name == "0001.srt"


class TestComposeIntro:
    def test_compose_intro_end_to_end(self, tmp_path, monkeypatch):
        ws = tmp_path / "demo"
        _touch(ws / "images" / "0000.png")
        _touch(ws / "images" / "0001.png")

        tts_calls: list[str] = []
        assemble_calls: dict = {}

        class FakeTTSTool:
            def __init__(self, config):
                pass

            def run(self, text, audio_path, srt_path, rate=None, volume=None):
                tts_calls.append(text)
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"\x00" * 100)
                srt_path.write_text("1\n", encoding="utf-8")
                return audio_path, srt_path

        class FakeAssembleTool:
            def __init__(self, config):
                pass

            def run(self, **kwargs):
                assemble_calls.update(kwargs)
                kwargs["output_path"].write_bytes(b"mp4")
                return kwargs["output_path"]

        monkeypatch.setattr("src.tools.tts_tool.TTSTool", FakeTTSTool)
        monkeypatch.setattr(
            "src.tools.video_assemble_tool.VideoAssembleTool", FakeAssembleTool
        )
        monkeypatch.setattr(
            "src.video.intro_tv_frame.resolve_intro_frame_config", lambda cfg: None
        )

        cfg = {
            "project": {"default_output": str(tmp_path / "out")},
            "video": {"resolution": [1920, 1080]},
            "intro": {"focus_clip": False},
            "tts": {},
            "subtitle": {"enabled": True},
        }
        text = "开场白。第二句。"
        out = compose_intro(ws, text, cfg)

        assert out == ws / "intro" / "intro.mp4"
        assert tts_calls == ["开场白。", "第二句。"]
        assert len(assemble_calls["images"]) == 2
        assert assemble_calls["workspace"] == ws

    def test_compose_intro_with_tv_frame(self, tmp_path, monkeypatch):
        ws = tmp_path / "demo"
        _touch(ws / "images" / "0000.png")

        class FakeTTSTool:
            def __init__(self, config):
                pass

            def run(self, text, audio_path, srt_path, rate=None, volume=None):
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"\x00" * 100)
                srt_path.write_text("", encoding="utf-8")
                return audio_path, srt_path

        class FakeAssembleTool:
            def __init__(self, config):
                pass

            def run(self, **kwargs):
                kwargs["output_path"].write_bytes(b"content")
                return kwargs["output_path"]

        tv_calls: dict = {}

        def fake_tv(content, output, frame_cfg):
            tv_calls["content"] = content
            tv_calls["output"] = output
            output.write_bytes(b"tv")
            return output

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

        monkeypatch.setattr("src.tools.tts_tool.TTSTool", FakeTTSTool)
        monkeypatch.setattr(
            "src.tools.video_assemble_tool.VideoAssembleTool", FakeAssembleTool
        )
        monkeypatch.setattr(
            "src.video.intro_tv_frame.resolve_intro_frame_config", lambda cfg: fake_cfg
        )
        monkeypatch.setattr(
            "src.video.intro_tv_frame.composite_content_in_tv_frame", fake_tv
        )

        cfg = {"project": {}, "intro": {"focus_clip": False}, "tts": {}, "subtitle": {}}
        out = compose_intro(ws, "片头。", cfg)
        assert out.read_bytes() == b"tv"
        assert tv_calls["content"] == ws / "intro" / "story_part.mp4"

    def test_compose_intro_with_focus_clip(self, tmp_path, monkeypatch):
        ws = tmp_path / "demo"
        _touch(ws / "images" / "0000.png")
        focus = tmp_path / "focus.mp4"
        focus.write_bytes(b"vid")
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

        class FakeAssembleTool:
            def __init__(self, config):
                pass

            def run(self, **kwargs):
                kwargs["output_path"].write_bytes(b"story")
                return kwargs["output_path"]

        concat_calls: list[tuple[Path, Path]] = []

        def fake_build(video_path, audio_path, output_path, **kwargs):
            output_path.write_bytes(b"focus")
            return output_path

        def fake_concat(first, second, output_path, **kwargs):
            concat_calls.append((first, second))
            output_path.write_bytes(b"content")
            return output_path

        monkeypatch.setattr("src.tools.tts_tool.TTSTool", FakeTTSTool)
        monkeypatch.setattr(
            "src.tools.video_assemble_tool.VideoAssembleTool", FakeAssembleTool
        )
        monkeypatch.setattr(
            "src.video.intro_tv_frame.resolve_intro_frame_config", lambda cfg: None
        )
        monkeypatch.setattr(
            "src.video.intro_content.build_focus_tagline_clip", fake_build
        )
        monkeypatch.setattr(
            "src.video.intro_content.concat_content_clips", fake_concat
        )

        cfg = {
            "project": {"default_output": str(tmp_path / "out")},
            "video": {"resolution": [1920, 1080], "codec": "libx264", "crf": 18},
            "intro": {
                "focus_clip": str(focus),
                "focus_tagline": "关掉杂念，故事开始咯",
            },
            "tts": {},
            "subtitle": {},
        }
        out = compose_intro(ws, "开场。", cfg)
        assert out == ws / "intro" / "intro.mp4"
        assert tts_calls == ["开场。", "关掉杂念，故事开始咯"]
        assert concat_calls[0][0] == ws / "intro" / "story_part.mp4"
        assert concat_calls[0][1] == ws / "intro" / "focus_part.mp4"

    def test_no_split_single_segment(self, tmp_path, monkeypatch):
        ws = tmp_path / "demo"
        _touch(ws / "images" / "0000.png")

        class FakeTTSTool:
            def __init__(self, config):
                pass

            def run(self, text, audio_path, srt_path, rate=None, volume=None):
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"\x00" * 100)
                srt_path.write_text("", encoding="utf-8")
                return audio_path, srt_path

        class FakeAssembleTool:
            def __init__(self, config):
                pass

            def run(self, **kwargs):
                kwargs["output_path"].write_bytes(b"mp4")
                return kwargs["output_path"]

        monkeypatch.setattr("src.tools.tts_tool.TTSTool", FakeTTSTool)
        monkeypatch.setattr(
            "src.tools.video_assemble_tool.VideoAssembleTool", FakeAssembleTool
        )
        monkeypatch.setattr(
            "src.video.intro_tv_frame.resolve_intro_frame_config", lambda cfg: None
        )

        cfg = {
            "project": {},
            "intro": {"focus_clip": False},
            "tts": {},
            "subtitle": {},
        }
        compose_intro(ws, "整段不分句", cfg, split_sentences=False)
        assert (ws / "intro" / "0000.mp3").exists()

    def test_intro_does_not_pollute_main_tmp_video(self, tmp_path, monkeypatch):
        """片头合成须使用 intro/tmp_video，不得覆盖正片 tmp_video/clip_0000。"""
        ws = tmp_path / "demo"
        _touch(ws / "images" / "0008.png")

        main_tmp = ws / "tmp_video"
        main_tmp.mkdir(parents=True)
        main_marker = main_tmp / "clip_0000.mp4"
        main_marker.write_bytes(b"MAIN_SEGMENT_0")

        captured: dict = {}

        class FakeTTSTool:
            def __init__(self, config):
                pass

            def run(self, text, audio_path, srt_path, rate=None, volume=None):
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(b"\x00" * 100)
                srt_path.write_text("1\n", encoding="utf-8")
                return audio_path, srt_path

        class FakeAssembleTool:
            def __init__(self, config):
                pass

            def run(self, **kwargs):
                captured.update(kwargs)
                tmp = kwargs.get("tmp_dir")
                assert tmp is not None
                tmp.mkdir(parents=True, exist_ok=True)
                (tmp / "clip_0000.mp4").write_bytes(b"INTRO_CLIP")
                kwargs["output_path"].write_bytes(b"mp4")
                return kwargs["output_path"]

        monkeypatch.setattr("src.tools.tts_tool.TTSTool", FakeTTSTool)
        monkeypatch.setattr(
            "src.tools.video_assemble_tool.VideoAssembleTool", FakeAssembleTool
        )
        monkeypatch.setattr(
            "src.video.intro_tv_frame.resolve_intro_frame_config", lambda cfg: None
        )

        cfg = {
            "project": {},
            "intro": {"focus_clip": False},
            "video": {"resolution": [1920, 1080]},
            "tts": {},
            "subtitle": {},
        }
        compose_intro(
            ws,
            "片头一句。",
            cfg,
            image_indices=[8],
            split_sentences=False,
        )

        assert captured["tmp_dir"] == ws / "intro" / "tmp_video"
        assert main_marker.read_bytes() == b"MAIN_SEGMENT_0"
        assert not (ws / "intro" / "tmp_video").exists()
