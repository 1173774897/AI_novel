"""Agent 断点状态修复测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent_state_repair import (
    dedupe_completed_nodes,
    repair_agent_state_data,
    segment_for_image_count,
    segment_legacy,
)
from src.tools.segment_tool import SegmentTool

pytestmark = pytest.mark.signature


class TestDedupeCompletedNodes:
    def test_dedupe_preserves_order(self):
        assert dedupe_completed_nodes(
            ["director", "content_analyzer", "director", "art_director"]
        ) == ["director", "content_analyzer", "art_director"]

    def test_empty(self):
        assert dedupe_completed_nodes(None) == []
        assert dedupe_completed_nodes([]) == []


class TestSegmentLegacy:
    def test_legacy_matches_image_count_for_jizhi(self):
        from src.config_manager import load_config

        text = Path("input/恶之花/01-极致捧杀.txt").read_text(encoding="utf-8")
        cfg = load_config()
        legacy = segment_legacy(text, cfg)
        current = SegmentTool(cfg).run(text)
        assert len(legacy) == 182
        assert len(current) == 181

    def test_segment_for_image_count_prefers_legacy_when_matching(self):
        text = Path("input/恶之花/01-极致捧杀.txt").read_text(encoding="utf-8")
        cfg = {"segmenter": {"method": "simple", "max_chars": 100, "min_chars": 20}}
        segs = segment_for_image_count(text, cfg, 182)
        assert len(segs) == 182


class TestRepairAgentStateData:
    def test_rebuilds_empty_segments_from_images(self, tmp_path):
        ws = tmp_path / "proj"
        img_dir = ws / "images"
        img_dir.mkdir(parents=True)
        for i in range(3):
            (img_dir / f"{i:04d}.png").write_bytes(b"fake")

        text = "第一句。第二句。第三句很长很长很长很长很长很长很长很长很长。"
        data = {
            "full_text": text,
            "segments": [],
            "images": [],
            "completed_nodes": ["director", "content_analyzer", "art_director"],
        }
        cfg = {"segmenter": {"method": "simple", "max_chars": 10, "min_chars": 2}}
        repaired = repair_agent_state_data(data, cfg, ws)

        assert len(repaired["segments"]) >= 1
        assert len(repaired["images"]) == 3
        assert repaired["images"][0].endswith("0000.png")

    def test_clears_voice_director_when_audio_incomplete(self, tmp_path):
        ws = tmp_path / "proj"
        (ws / "images").mkdir(parents=True)
        (ws / "audio").mkdir(parents=True)
        (ws / "images" / "0000.png").write_bytes(b"x")
        (ws / "images" / "0001.png").write_bytes(b"x")
        (ws / "audio" / "0000.mp3").write_bytes(b"x" * 200)

        data = {
            "full_text": "甲。乙。",
            "segments": [{"text": "甲。", "index": 0}, {"text": "乙。", "index": 1}],
            "images": [],
            "completed_nodes": [
                "director",
                "content_analyzer",
                "art_director",
                "voice_director",
            ],
        }
        cfg = {"segmenter": {"method": "simple", "max_chars": 100, "min_chars": 1}}
        repaired = repair_agent_state_data(data, cfg, ws)

        assert "voice_director" not in repaired["completed_nodes"]
        assert len(repaired["audio_files"]) == 1

    def test_dedupes_completed_nodes(self, tmp_path):
        ws = tmp_path / "proj"
        ws.mkdir()
        data = {
            "full_text": "",
            "segments": [{"text": "x", "index": 0}],
            "completed_nodes": ["director", "director", "content_analyzer"],
        }
        cfg = {"segmenter": {"method": "simple", "max_chars": 100, "min_chars": 1}}
        repaired = repair_agent_state_data(data, cfg, ws)
        assert repaired["completed_nodes"] == ["director", "content_analyzer"]


class TestAgentPipelineLoadRepair:
    @pytest.fixture
    def mock_cfg(self):
        return {
            "project": {"default_workspace": "workspace", "default_output": "output"},
            "segmenter": {"method": "simple", "max_chars": 100, "min_chars": 20},
        }

    def test_load_state_auto_repairs_empty_segments(self, tmp_path, mock_cfg, monkeypatch):
        from unittest.mock import patch

        ws = tmp_path / "01-test"
        img_dir = ws / "images"
        img_dir.mkdir(parents=True)
        text = Path("input/恶之花/01-极致捧杀.txt").read_text(encoding="utf-8")
        for i in range(182):
            (img_dir / f"{i:04d}.png").write_bytes(b"x")

        state = {
            "input_file": "input/恶之花/01-极致捧杀.txt",
            "workspace": str(ws),
            "full_text": text,
            "segments": [],
            "images": [],
            "completed_nodes": ["director", "content_analyzer", "art_director"],
        }
        (ws / "agent_state.json").write_text(
            json.dumps(state, ensure_ascii=False), encoding="utf-8"
        )

        with patch("src.agent_pipeline.load_config", return_value=mock_cfg):
            from src.agent_pipeline import AgentPipeline

            pipe = AgentPipeline(
                Path("input/恶之花/01-极致捧杀.txt"),
                workspace=ws,
                exact_workspace=True,
                resume=True,
            )
            loaded = pipe._load_state()

        assert loaded is not None
        assert len(loaded["segments"]) == 182
        assert len(loaded["images"]) == 182
