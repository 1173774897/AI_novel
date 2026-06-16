"""即梦 CLI 片头合成测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.video.jimeng_intro_compositor import (
    JimengIntroSettings,
    _build_multimodal_command,
    compute_jimeng_output_duration,
    fits_jimeng_duration,
    jimeng_safe_content_duration,
    resolve_jimeng_intro_settings,
)

pytestmark = pytest.mark.signature


class TestJimengDurationHelpers:
    def test_fits_within_limit(self):
        assert fits_jimeng_duration(14.5, 0.5)
        assert not fits_jimeng_duration(15.1, 0.0)
        assert not fits_jimeng_duration(1.0, 0.0)

    def test_safe_content_duration_leaves_margin(self):
        assert jimeng_safe_content_duration(15.0) == 14.95

    def test_compute_output_duration_capped(self):
        assert compute_jimeng_output_duration(12.0, 0.5) == 12
        assert compute_jimeng_output_duration(14.5, 0.5) == 15


class TestResolveJimengIntroSettings:
    def test_merges_imagegen_cli_command(self):
        cfg = {
            "intro": {"jimeng_cli": {"model_version": "seedance2.0_vip"}},
            "imagegen": {"cli_command": "dreamina"},
        }
        s = resolve_jimeng_intro_settings(cfg)
        assert s.cli_command == "dreamina"
        assert s.model_version == "seedance2.0_vip"


class TestBuildMultimodalCommand:
    def test_includes_image_video_and_audio(self, tmp_path):
        frame = tmp_path / "tv-frame.png"
        content = tmp_path / "content.mp4"
        audio = tmp_path / "narr.m4a"
        for p in (frame, content, audio):
            p.write_bytes(b"x")

        settings = JimengIntroSettings(
            cli_command="dreamina",
            model_version="seedance2.0fast",
            video_resolution="720p",
            ratio="16:9",
            poll=300,
            timeout=300.0,
            prompt="embed in tv",
            extra_args=(),
        )
        cmd = _build_multimodal_command(
            settings,
            tv_frame_image=frame,
            content_video=content,
            audio_path=audio,
            duration=15,
        )
        assert cmd[:2] == ["dreamina", "multimodal2video"]
        assert "--image" in cmd and str(frame) in cmd
        assert "--video" in cmd and str(content) in cmd
        assert "--audio" in cmd
        assert "--prompt" in cmd and "embed in tv" in cmd
        assert "--duration=15" in cmd
        assert "--poll=0" in cmd


class TestCompositeViaJimengCli:
    def test_submits_and_downloads_video(self, tmp_path, monkeypatch):
        from src.video.jimeng_intro_compositor import composite_via_jimeng_cli

        frame = tmp_path / "tv-frame.png"
        content = tmp_path / "content.mp4"
        audio = tmp_path / "narr.m4a"
        out = tmp_path / "final.mp4"
        work = tmp_path / "work"
        for p in (frame, content, audio):
            p.write_bytes(b"x")

        settings = JimengIntroSettings(
            cli_command="dreamina",
            model_version="seedance2.0fast",
            video_resolution="720p",
            ratio="16:9",
            poll=60,
            timeout=60.0,
            prompt="test",
            extra_args=(),
        )

        downloaded = work / "result.mp4"

        def fake_run(cmd, timeout):
            if "multimodal2video" in cmd:
                payload = {"gen_status": "success", "result_json": {"videos": [{"path": str(downloaded)}]}}
                result = type("R", (), {})()
                result.stdout = json.dumps(payload)
                result.returncode = 0
                return result
            raise AssertionError(cmd)

        monkeypatch.setattr("src.video.jimeng_intro_compositor.shutil.which", lambda _: "/usr/bin/dreamina")
        monkeypatch.setattr("src.video.jimeng_intro_compositor._run_cli", fake_run)

        def fake_resolve(data, download_dir):
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            downloaded.write_bytes(b"video")
            return downloaded

        monkeypatch.setattr(
            "src.video.jimeng_intro_compositor._resolve_output_video", fake_resolve
        )

        result = composite_via_jimeng_cli(
            tv_frame_image=frame,
            content_video=content,
            narration_audio=audio,
            output_path=out,
            settings=settings,
            duration=15,
            work_dir=work,
        )
        assert result == out
        assert out.read_bytes() == b"video"
