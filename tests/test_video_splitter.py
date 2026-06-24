"""video_splitter 单元测试。"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.video.video_splitter import (
    JIMENG_MAX_SEGMENT_SEC,
    JIMENG_MIN_SEGMENT_SEC,
    JIMENG_SAFE_MAX_CLIP_SEC,
    JIMENG_UPLOAD_MAX_SEC,
    clamp_clip_duration,
    ensure_clip_uploadable,
    plan_segment_spans,
)

pytestmark = pytest.mark.signature


class TestPlanSegmentSpans:
    def test_empty_duration(self):
        assert plan_segment_spans(0) == []

    def test_single_short_clip(self):
        spans = plan_segment_spans(12.0)
        assert len(spans) == 1
        assert spans[0].start_sec == 0.0
        assert spans[0].end_sec == 12.0

    def test_exactly_max_one_segment(self):
        spans = plan_segment_spans(JIMENG_SAFE_MAX_CLIP_SEC)
        assert len(spans) == 1
        assert spans[0].duration_sec == pytest.approx(JIMENG_SAFE_MAX_CLIP_SEC)

    def test_clamp_clip_duration(self):
        assert clamp_clip_duration(15.5) == pytest.approx(JIMENG_SAFE_MAX_CLIP_SEC)
        assert clamp_clip_duration(10) == 10.0

    @patch("src.video.video_splitter.probe_stream_duration", return_value=15.015)
    @patch("src.video.video_splitter.subprocess.run")
    def test_ensure_clip_uploadable_trims_over_limit(
        self, mock_run, mock_probe, tmp_path: Path
    ):
        clip = tmp_path / "source.mp4"
        clip.write_bytes(b"x")
        trim_out = tmp_path / "source_trim.mp4"

        def fake_run(cmd, **kwargs):
            import subprocess as sp
            if "-t" in cmd:
                trim_out.write_bytes(b"trimmed")
            return sp.CompletedProcess(cmd, 0, "", "")

        import subprocess
        mock_run.side_effect = fake_run
        ensure_clip_uploadable(clip)
        assert mock_run.called
        assert clip.is_file()

    def test_splits_long_video(self):
        spans = plan_segment_spans(40.0, max_seg=15.0, min_seg=4.0)
        assert len(spans) >= 2
        assert all(s.duration_sec >= 4.0 - 0.01 for s in spans)
        assert all(s.duration_sec <= 15.0 + 0.01 for s in spans)
        assert spans[0].start_sec == 0.0
        assert spans[-1].end_sec == pytest.approx(40.0)

    def test_merges_short_tail(self):
        # 18s: 14+4，避免最后一段 < min_seg
        spans = plan_segment_spans(18.0, max_seg=15.0, min_seg=4.0)
        assert len(spans) == 2
        assert spans[-1].duration_sec == pytest.approx(4.0)
        assert spans[-1].end_sec == pytest.approx(18.0)


class TestExtractLastFrame:
    @patch("src.video.video_splitter.probe_video_duration", return_value=4.06)
    @patch("src.video.video_splitter.subprocess.run")
    def test_short_clip_uses_duration_seek_not_tight_sseof(
        self, mock_run, mock_probe, tmp_path: Path
    ):
        import subprocess as sp

        from src.video.video_splitter import extract_last_frame

        video = tmp_path / "short.mp4"
        video.write_bytes(b"mp4")
        out = tmp_path / "last.png"

        def fake_run(cmd, **kwargs):
            if "-ss" in cmd and "3.940" in cmd:
                out.write_bytes(b"png")
                return sp.CompletedProcess(cmd, 0, "", "")
            return sp.CompletedProcess(cmd, 1, "", "no frame")

        mock_run.side_effect = fake_run
        result = extract_last_frame(video, out)
        assert result == out
        first_cmd = mock_run.call_args_list[0][0][0]
        ss_idx = first_cmd.index("-ss")
        assert first_cmd[ss_idx + 1] == "3.940"
        assert "-sseof" not in first_cmd

    @patch("src.video.video_splitter.probe_video_duration", return_value=4.06)
    @patch("src.video.video_splitter.subprocess.run")
    def test_falls_back_to_sseof_when_duration_seek_fails(
        self, mock_run, mock_probe, tmp_path: Path
    ):
        import subprocess as sp

        from src.video.video_splitter import extract_last_frame

        video = tmp_path / "short.mp4"
        video.write_bytes(b"mp4")
        out = tmp_path / "last.png"
        calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            calls["n"] += 1
            if "-sseof" in cmd:
                out.write_bytes(b"png")
                return sp.CompletedProcess(cmd, 0, "", "")
            return sp.CompletedProcess(cmd, 1, "", "no frame")

        mock_run.side_effect = fake_run
        extract_last_frame(video, out)
        assert calls["n"] == 2
        assert "-sseof" in mock_run.call_args_list[1][0][0]
