"""即梦 CLI 图片后端测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.imagegen.image_generator import create_image_generator
from src.imagegen.jimeng_cli_backend import (
    JimengCliBackend,
    JimengGenerationError,
    _find_image_paths,
    _newest_image_in_dir,
    _normalize_model_version,
    _parse_cli_json,
    _ratio_from_size,
)


@pytest.mark.signature
class TestJimengCliHelpers:
    def test_ratio_from_size_portrait(self):
        assert _ratio_from_size(1024, 1792) == "9:16"

    def test_ratio_from_size_landscape(self):
        assert _ratio_from_size(1920, 1080) == "16:9"

    def test_ratio_from_size_square(self):
        assert _ratio_from_size(1024, 1024) == "1:1"

    def test_ratio_from_size_invalid(self):
        assert _ratio_from_size(0, 0) == "9:16"

    def test_normalize_model_version(self):
        assert _normalize_model_version("jimeng-4.5") == "4.5"
        assert _normalize_model_version("4.5") == "4.5"

    def test_parse_cli_json(self):
        assert _parse_cli_json('{"gen_status":"success"}')["gen_status"] == "success"

    def test_parse_cli_json_invalid(self):
        with pytest.raises(RuntimeError, match="非 JSON"):
            _parse_cli_json("not json")

    def test_find_image_paths_nested(self):
        payload = {
            "data": {
                "images": [{"path": "/tmp/out/001.png"}],
            }
        }
        assert _find_image_paths(payload) == ["/tmp/out/001.png"]

    def test_find_image_paths_empty(self):
        assert _find_image_paths({"status": "ok"}) == []

    def test_newest_image_in_dir(self, tmp_path: Path):
        older = tmp_path / "old.png"
        newer = tmp_path / "nested" / "new.jpg"
        newer.parent.mkdir()
        older.write_bytes(b"old")
        newer.write_bytes(b"new")
        import os
        import time

        now = time.time()
        os.utime(older, (now - 10, now - 10))
        os.utime(newer, (now, now))

        assert _newest_image_in_dir(tmp_path) == newer

    def test_newest_image_in_dir_empty(self, tmp_path: Path):
        assert _newest_image_in_dir(tmp_path) is None


@pytest.mark.signature
class TestJimengCliBackendJimengFlavor:
    def _make_backend(self, **overrides) -> JimengCliBackend:
        config = {
            "cli_flavor": "jimeng",
            "cli_command": "jimeng",
            "model": "jimeng-4.5",
            "ratio": "9:16",
            "resolution": "2k",
            "region": "cn",
            "width": 1024,
            "height": 1792,
            "request_interval": 0,
            "timeout": 30,
            **overrides,
        }
        return JimengCliBackend(config)

    def test_build_command_includes_core_flags(self, tmp_path: Path):
        backend = self._make_backend(negative_prompt="nsfw")
        cmd = backend._build_jimeng_command("a cat", tmp_path)
        assert cmd[:4] == ["jimeng", "image", "generate", "--prompt"]
        assert "a cat" in cmd
        assert "--model" in cmd and "jimeng-4.5" in cmd
        assert "--output-dir" in cmd and str(tmp_path) in cmd
        assert "--wait" in cmd
        assert "--json" in cmd
        assert "--negative-prompt" in cmd and "nsfw" in cmd

    @patch("src.imagegen.jimeng_cli_backend.subprocess.run")
    @patch("src.imagegen.jimeng_cli_backend.shutil.which", return_value="/usr/bin/jimeng")
    def test_generate_success_from_json_path(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ):
        image_path = tmp_path / "result.png"
        Image.new("RGB", (64, 64), color="red").save(image_path)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"path": str(image_path)}),
            stderr="",
        )

        backend = self._make_backend(output_dir=str(tmp_path))
        image = backend.generate("sunset city")

        assert image.size == (64, 64)
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0][0] == "jimeng"


@pytest.mark.signature
class TestJimengCliBackendDreaminaFlavor:
    def _make_backend(self, **overrides) -> JimengCliBackend:
        config = {
            "cli_flavor": "dreamina",
            "cli_command": "dreamina",
            "model": "4.5",
            "ratio": "9:16",
            "resolution": "2k",
            "width": 1024,
            "height": 1792,
            "request_interval": 0,
            "timeout": 30,
            **overrides,
        }
        return JimengCliBackend(config)

    def test_build_dreamina_command(self):
        backend = self._make_backend()
        cmd = backend._build_dreamina_command("a cat portrait")
        assert cmd[0:2] == ["dreamina", "text2image"]
        assert "--prompt=a cat portrait" in cmd
        assert "--ratio=9:16" in cmd
        assert "--resolution_type=2k" in cmd
        assert "--model_version=4.5" in cmd
        assert "--poll=30" in cmd

    def test_build_dreamina_command_normalizes_model(self):
        backend = self._make_backend(model="jimeng-5.0")
        cmd = backend._build_dreamina_command("test")
        assert "--model_version=5.0" in cmd

    @patch("src.imagegen.jimeng_cli_backend.subprocess.run")
    @patch("src.imagegen.jimeng_cli_backend.shutil.which", return_value="/usr/local/bin/dreamina")
    def test_generate_success_from_image_url(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ):
        payload = {
            "submit_id": "abc",
            "gen_status": "success",
            "result_json": {
                "images": [{"image_url": "https://example.com/out.png"}],
            },
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )
        fake_image = Image.new("RGB", (48, 80), color="green")

        backend = self._make_backend(output_dir=str(tmp_path))
        with patch(
            "src.imagegen.jimeng_cli_backend._download_image_url",
            return_value=fake_image,
        ) as mock_download:
            image = backend.generate("portrait")

        assert image.size == (48, 80)
        mock_download.assert_called_once_with("https://example.com/out.png")
        assert mock_run.call_args.args[0][0:2] == ["dreamina", "text2image"]

    @patch("src.imagegen.jimeng_cli_backend.subprocess.run")
    @patch("src.imagegen.jimeng_cli_backend.shutil.which", return_value="/usr/local/bin/dreamina")
    def test_generate_polls_query_result_when_querying(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ):
        image_path = tmp_path / "done.png"
        Image.new("RGB", (20, 30), color="blue").save(image_path)
        mock_run.side_effect = [
            MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "submit_id": "task-1",
                        "gen_status": "querying",
                    }
                ),
                stderr="",
            ),
            MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "submit_id": "task-1",
                        "gen_status": "success",
                        "result_json": {
                            "images": [{"path": str(image_path)}],
                        },
                    }
                ),
                stderr="",
            ),
        ]

        backend = self._make_backend(output_dir=str(tmp_path))
        with patch("src.imagegen.jimeng_cli_backend.time.sleep"):
            image = backend.generate("wait me")

        assert image.size == (20, 30)
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[1].args[0][1:3] == ["query_result", "--submit_id=task-1"]

    @patch("src.imagegen.jimeng_cli_backend.shutil.which", return_value=None)
    def test_generate_missing_dreamina_cli_hint(self, _which: MagicMock):
        backend = self._make_backend()
        with pytest.raises(RuntimeError, match="curl -s https://jimeng.jianying.com/cli"):
            backend.generate("test")


@pytest.mark.signature
class TestJimengCliFactory:
    def test_create_image_generator_jimeng_cli_defaults_to_dreamina(self):
        gen = create_image_generator(
            {
                "backend": "jimeng-cli",
                "model": "",
                "width": 1024,
                "height": 1792,
            }
        )
        assert isinstance(gen, JimengCliBackend)
        assert gen._flavor == "dreamina"
        assert gen._command == "dreamina"
        assert gen._model == "4.5"
        assert gen._ratio == "9:16"

    def test_create_image_generator_coerces_float_model(self):
        gen = create_image_generator(
            {
                "backend": "jimeng-cli",
                "model": 4.5,
                "ratio": "16:9",
            }
        )
        assert isinstance(gen, JimengCliBackend)
        assert gen._model == "4.5"
        assert gen._ratio == "16:9"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown image backend"):
            create_image_generator({"backend": "not-a-backend"})
