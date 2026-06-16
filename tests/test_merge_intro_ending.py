"""片头/正片/片尾拼接测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.video.full_video_merge import (
    merge_intro_main_ending,
    resolve_default_paths,
    resolve_project_stem,
)

pytestmark = pytest.mark.signature


class TestResolvePaths:
    def test_stem_from_agent_state(self, tmp_path):
        ws = tmp_path / "demo"
        ws.mkdir()
        (ws / "agent_state.json").write_text(
            json.dumps({"input_file": "input/无尽恶意.txt"}),
            encoding="utf-8",
        )
        assert resolve_project_stem(ws) == "无尽恶意"

    def test_stem_fallback_workspace_name(self, tmp_path):
        ws = tmp_path / "my_project"
        ws.mkdir()
        assert resolve_project_stem(ws) == "my_project"

    def test_default_paths(self, tmp_path):
        ws = tmp_path / "proj"
        ws.mkdir()
        (ws / "intro").mkdir()
        cfg = {"project": {"default_output": str(tmp_path / "out")}}
        intro, main, ending = resolve_default_paths(
            ws, cfg, project_root=tmp_path,
        )
        assert intro == ws / "intro" / "intro.mp4"
        assert main == tmp_path / "out" / "proj.mp4"
        assert ending == ws / "intro" / "ending.mp4"


class TestMergeIntroMainEnding:
    def _touch_mp4(self, path: Path, size: int = 200) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * size)

    def test_single_clip_copy(self, tmp_path):
        src = tmp_path / "main.mp4"
        dst = tmp_path / "out.mp4"
        self._touch_mp4(src)
        merge_intro_main_ending([src], dst, {})
        assert dst.read_bytes() == src.read_bytes()

    def test_empty_clips_raises(self, tmp_path):
        with pytest.raises(ValueError, match="至少"):
            merge_intro_main_ending([], tmp_path / "out.mp4", {})

    def test_missing_clip_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            merge_intro_main_ending(
                [tmp_path / "missing.mp4"],
                tmp_path / "out.mp4",
                {},
            )

    def test_three_part_normalizes_short_clips_then_concat(self, tmp_path, monkeypatch):
        intro = tmp_path / "intro.mp4"
        main = tmp_path / "main.mp4"
        ending = tmp_path / "ending.mp4"
        out = tmp_path / "full.mp4"
        for p in (intro, main, ending):
            self._touch_mp4(p)

        norm_calls: list[Path] = []
        concat_calls: list[list[Path]] = []

        monkeypatch.setattr(
            "src.video.full_video_merge._find_main_clip_index", lambda clips: 1
        )
        monkeypatch.setattr(
            "src.video.full_video_merge.probe_video_stream_size",
            lambda _p: (1920, 1080),
        )
        monkeypatch.setattr(
            "src.video.full_video_merge._probe_audio_spec",
            lambda path: (24000, 1) if path == main else (48000, 2),
        )

        def fake_normalize(src, dst, **kwargs):
            norm_calls.append(src)
            dst.write_bytes(b"norm")
            return dst

        def fake_concat(clips, output_path, tmp_dir):
            concat_calls.append(list(clips))
            output_path.write_bytes(b"merged")
            return output_path

        monkeypatch.setattr(
            "src.video.full_video_merge._normalize_clip", fake_normalize
        )
        monkeypatch.setattr(
            "src.video.full_video_merge._concat_copy", fake_concat
        )

        merge_intro_main_ending(
            [intro, main, ending],
            out,
            {"video": {"codec": "libx264", "crf": 18, "fps": 30}},
            tmp_dir=tmp_path / "tmp",
        )

        assert set(norm_calls) == {intro, ending}
        assert len(concat_calls) == 1
        assert concat_calls[0][1] == main
        assert out.read_bytes() == b"merged"

    def test_two_part_skips_normalize_when_main_matches(self, tmp_path, monkeypatch):
        main = tmp_path / "main.mp4"
        ending = tmp_path / "ending.mp4"
        out = tmp_path / "full.mp4"
        self._touch_mp4(main)
        self._touch_mp4(ending)

        monkeypatch.setattr(
            "src.video.full_video_merge._find_main_clip_index", lambda clips: 0
        )
        monkeypatch.setattr(
            "src.video.full_video_merge.probe_video_stream_size",
            lambda _p: (1920, 1080),
        )
        monkeypatch.setattr(
            "src.video.full_video_merge._probe_audio_spec",
            lambda _p: (24000, 1),
        )
        monkeypatch.setattr(
            "src.video.full_video_merge._clip_matches_target",
            lambda path, **kwargs: path == main,
        )
        concat_inputs: list[list[Path]] = []

        def fake_concat(clips, output_path, tmp_dir):
            concat_inputs.append(list(clips))
            output_path.write_bytes(b"ok")
            return output_path

        def fake_normalize(src, dst, **kwargs):
            dst.write_bytes(b"norm")
            return dst

        monkeypatch.setattr(
            "src.video.full_video_merge._normalize_clip", fake_normalize
        )
        monkeypatch.setattr(
            "src.video.full_video_merge._concat_copy", fake_concat
        )
        merge_intro_main_ending([main, ending], out, {})
        assert concat_inputs[0][0] == main
        assert out.read_bytes() == b"ok"


class TestMergeScriptCLI:
    @staticmethod
    def _touch_mp4(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 200)

    def test_cli_missing_main(self, tmp_path):
        from scripts.merge_intro_ending import main

        ws = tmp_path / "proj"
        ws.mkdir()
        (ws / "intro").mkdir()
        self._touch_mp4(ws / "intro" / "intro.mp4")
        self._touch_mp4(ws / "intro" / "ending.mp4")
        assert main([str(ws)]) == 2

    def test_cli_success(self, tmp_path, monkeypatch):
        from scripts.merge_intro_ending import main

        ws = tmp_path / "无尽恶意"
        ws.mkdir()
        (ws / "intro").mkdir()
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        main_mp4 = out_dir / "无尽恶意.mp4"
        self._touch_mp4(main_mp4)
        self._touch_mp4(ws / "intro" / "intro.mp4")
        self._touch_mp4(ws / "intro" / "ending.mp4")
        (ws / "agent_state.json").write_text(
            json.dumps({"input_file": "input/无尽恶意.txt"}),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "src.video.full_video_merge.merge_intro_main_ending",
            lambda clips, output_path, config, tmp_dir=None: (
                output_path.write_bytes(b"full") or output_path
            ),
        )
        monkeypatch.setattr(
            "src.config_manager.load_config",
            lambda _path=None: {
                "project": {"default_output": str(out_dir)},
                "video": {"codec": "libx264", "crf": 18, "fps": 30},
            },
        )

        out_file = out_dir / "完整.mp4"
        rc = main([
            str(ws),
            "--main", str(main_mp4),
            "-o", str(out_file),
        ])
        assert rc == 0
        assert out_file.read_bytes() == b"full"
