"""即梦 CLI 视频后端测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.videogen.jimeng_cli_backend import (
    JimengCliVideoBackend,
    _clamp_dreamina_duration,
    merge_jimeng_cli_videogen_config,
)
from src.videogen.video_generator import create_video_generator


@pytest.mark.signature
class TestJimengCliVideoHelpers:
    def test_clamp_dreamina_duration(self):
        assert _clamp_dreamina_duration(3) == 4
        assert _clamp_dreamina_duration(10) == 10
        assert _clamp_dreamina_duration(20) == 15

    def test_merge_inherits_imagegen_cli_fields(self):
        merged = merge_jimeng_cli_videogen_config(
            {"backend": "jimeng-cli"},
            {
                "cli_flavor": "dreamina",
                "cli_command": "dreamina",
                "ratio": "16:9",
            },
        )
        assert merged["cli_flavor"] == "dreamina"
        assert merged["cli_command"] == "dreamina"
        assert merged["ratio"] == "16:9"

    def test_merge_does_not_inherit_imagegen_timeout(self):
        merged = merge_jimeng_cli_videogen_config(
            {"backend": "jimeng-cli"},
            {"timeout": 300, "cli_command": "dreamina"},
        )
        assert merged["poll_timeout"] == 21600
        assert "timeout" not in merged or merged.get("timeout") != 300

    def test_is_concurrency_limit_error(self):
        from src.videogen.jimeng_cli_backend import is_concurrency_limit_error

        assert is_concurrency_limit_error("api error: ret=1310, message=ExceedConcurrencyLimit")
        assert not is_concurrency_limit_error("timeout")


@pytest.mark.signature
class TestJimengCliVideoBackendDreamina:
    def _make_backend(self, **overrides) -> JimengCliVideoBackend:
        config = {
            "cli_flavor": "dreamina",
            "cli_command": "dreamina",
            "model_version": "seedance2.0fast",
            "ratio": "16:9",
            "video_resolution": "720p",
            "duration": 5,
            "request_interval": 0,
            "timeout": 30,
            **overrides,
        }
        return JimengCliVideoBackend(config)

    def test_build_text2video_command(self):
        backend = self._make_backend()
        cmd = backend._build_dreamina_text2video_command("a cat runs", 5)
        assert cmd[:3] == ["dreamina", "text2video", "--prompt=a cat runs"]
        assert "--ratio=16:9" in cmd
        assert "--model_version=seedance2.0fast" in cmd

    def test_build_image2video_command(self, tmp_path: Path):
        image = tmp_path / "first.png"
        image.write_bytes(b"png")
        backend = self._make_backend()
        cmd = backend._build_dreamina_image2video_command("push in", image, 6)
        assert cmd[:3] == ["dreamina", "image2video", f"--image={image}"]

    def test_build_multimodal_command(self, tmp_path: Path):
        image = tmp_path / "char.png"
        video = tmp_path / "clip.mp4"
        audio = tmp_path / "clip.m4a"
        image.write_bytes(b"png")
        video.write_bytes(b"mp4")
        audio.write_bytes(b"aac")
        backend = self._make_backend()
        cmd = backend._build_dreamina_multimodal_command(
            "replace character",
            image_paths=[image],
            video_paths=[video],
            audio_paths=[audio],
            duration=8,
        )
        assert cmd[0:2] == ["dreamina", "multimodal2video"]
        assert "--prompt=replace character" in cmd
        assert "--image" in cmd and str(image) in cmd
        assert "--video" in cmd and str(video) in cmd
        assert "--audio" in cmd and str(audio) in cmd
        assert "--duration=8" in cmd

    def test_generate_multimodal_rejects_jimeng_flavor(self, tmp_path: Path):
        backend = JimengCliVideoBackend(
            {
                "cli_flavor": "jimeng",
                "cli_command": "jimeng",
                "duration": 5,
            }
        )
        with pytest.raises(RuntimeError, match="multimodal2video"):
            backend.generate_multimodal(
                "test",
                image_paths=[tmp_path / "a.png"],
                video_paths=[tmp_path / "b.mp4"],
                output_dir=tmp_path,
            )

    @pytest.mark.skip(reason="i2i 在 imagegen 后端，非 video 后端")
    def test_build_dreamina_i2i_command(self, tmp_path: Path):
        ref = tmp_path / "anchor.png"
        ref.write_bytes(b"png")
        backend = self._make_backend()
        cmd = backend._build_dreamina_i2i_command("same cat, new pose", [ref])
        assert cmd[1] == "image2image"
        assert f"--images={ref}" in cmd
        assert "--prompt=same cat, new pose" in cmd

    @patch("src.videogen.jimeng_cli_backend._run_cli")
    @patch("src.video.jimeng_intro_compositor.shutil.which", return_value="/usr/local/bin/dreamina")
    def test_generate_text2video_success(self, _which, mock_run, tmp_path: Path):
        video = tmp_path / "out.mp4"
        video.write_bytes(b"fake-video")
        payload = {
            "gen_status": "success",
            "result_json": {"videos": [{"path": str(video)}]},
        }
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(payload)

        backend = self._make_backend()
        result = backend.generate("a cat runs", duration=5, output_dir=tmp_path)

        assert result.video_path == video
        assert result.duration == 5.0
        assert result.pending is False

    @patch("src.videogen.jimeng_cli_backend._run_cli")
    @patch("src.video.jimeng_intro_compositor.shutil.which", return_value="/usr/local/bin/dreamina")
    def test_async_submit_returns_pending(self, _which, mock_run, tmp_path: Path):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(
            {"gen_status": "querying", "submit_id": "abc-123"}
        )
        backend = self._make_backend(async_submit=True)
        result = backend.generate("cat", duration=5, output_dir=tmp_path)
        assert result.pending is True
        assert result.submit_id == "abc-123"
        cmd = mock_run.call_args[0][0]
        assert any(arg == "--poll=0" for arg in cmd)


@pytest.mark.signature
class TestJimengCliVideoBackendJimeng:
    def _make_backend(self, **overrides) -> JimengCliVideoBackend:
        config = {
            "cli_flavor": "jimeng",
            "cli_command": "jimeng",
            "model": "jimeng-video-seedance-2.0-fast",
            "ratio": "16:9",
            "video_resolution": "720p",
            "duration": 5,
            "region": "cn",
            "request_interval": 0,
            "timeout": 30,
            **overrides,
        }
        return JimengCliVideoBackend(config)

    def test_build_jimeng_text_to_video_command(self, tmp_path: Path):
        backend = self._make_backend()
        cmd = backend._build_jimeng_command("fox in snow", None, 5, tmp_path)
        assert cmd[0:4] == ["jimeng", "video", "generate", "--prompt"]
        assert "fox in snow" in cmd
        assert cmd[cmd.index("--mode") + 1] == "text_to_video"

    def test_build_jimeng_image_to_video_command(self, tmp_path: Path):
        image = tmp_path / "first.png"
        image.write_bytes(b"png")
        backend = self._make_backend()
        cmd = backend._build_jimeng_command("push in", image, 5, tmp_path)
        assert cmd[cmd.index("--mode") + 1] == "image_to_video"
        assert "--image-file" in cmd
        assert str(image) in cmd

    @patch("src.videogen.jimeng_cli_backend._run_cli")
    @patch("src.video.jimeng_intro_compositor.shutil.which", return_value="/usr/bin/jimeng")
    def test_generate_jimeng_success(self, _which, mock_run, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake-video")
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps(
            {"output": {"path": str(video)}}
        )

        backend = self._make_backend()
        backend._output_dir = str(tmp_path)
        result = backend.generate("fox in snow", duration=5)

        assert result.video_path == video


@pytest.mark.signature
class TestJimengCliVideoFactory:
    def test_create_video_generator_jimeng_cli(self):
        gen = create_video_generator(
            {
                "backend": "jimeng-cli",
                "cli_flavor": "dreamina",
                "cli_command": "dreamina",
            }
        )
        assert isinstance(gen, JimengCliVideoBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown video backend"):
            create_video_generator({"backend": "not-real"})
