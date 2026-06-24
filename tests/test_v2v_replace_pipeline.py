"""V2VReplacePipeline 测试（Mock 即梦，不调用 CLI）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.v2v_replace.models import V2VExtension, V2VReplaceJob, V2VSegment
from src.v2v_replace_pipeline import V2VReplacePipeline
from src.videogen.video_generator import VideoResult

pytestmark = pytest.mark.signature


@pytest.fixture
def mock_video_file(tmp_path: Path) -> Path:
    """最小可 ffprobe 的占位：用真实短 mp4 更好，此处 mock probe。"""
    p = tmp_path / "source.mp4"
    p.write_bytes(b"fake")
    return p


@pytest.fixture
def mock_char_image(tmp_path: Path) -> Path:
    p = tmp_path / "char.png"
    p.write_bytes(b"png")
    return p


class TestV2VReplacePipeline:
    @patch("src.v2v_replace_pipeline.probe_video_duration", return_value=10.0)
    @patch("src.v2v_replace_pipeline.extract_video_clip")
    @patch("src.v2v_replace_pipeline.extract_audio_clip", return_value=None)
    @patch("src.v2v_replace_pipeline.plan_segment_spans")
    @patch.object(V2VReplacePipeline, "_create_backend")
    @patch.object(V2VReplacePipeline, "_process_segments")
    def test_run_creates_job_and_delegates(
        self,
        mock_process,
        mock_backend,
        mock_plan,
        mock_audio,
        mock_extract,
        mock_probe,
        tmp_path,
        mock_video_file,
        mock_char_image,
    ):
        from src.video.video_splitter import VideoSegmentSpan

        mock_plan.return_value = [VideoSegmentSpan(id=1, start_sec=0.0, end_sec=10.0)]
        mock_process.return_value = {"status": "completed", "video_path": "/out.mp4"}

        clip_out = tmp_path / "clip.mp4"
        mock_extract.return_value = clip_out

        pipe = V2VReplacePipeline(
            config={
                "llm": {},
                "imagegen": {},
                "director": {"videogen": {"backend": "jimeng-cli"}},
                "v2v_replace": {},
            },
            workspace=tmp_path / "ws",
        )
        result = pipe.run(mock_video_file, mock_char_image)
        assert result["status"] == "completed"
        mock_process.assert_called_once()
        run_dir = Path(mock_process.call_args[0][0])
        assert (run_dir / "job.json").is_file()

    def test_assemble_requires_all_outputs(self, tmp_path: Path):
        job = V2VReplaceJob(
            run_id="abc",
            source_video="/x.mp4",
            character_image="/c.png",
            prompt="test",
            segments=[
                V2VSegment(
                    id=1,
                    start_sec=0,
                    end_sec=5,
                    output_clip=str(tmp_path / "missing.mp4"),
                    status="done",
                )
            ],
        )
        pipe = V2VReplacePipeline(config={"director": {"videogen": {}}})
        with pytest.raises(RuntimeError, match="缺少输出视频"):
            pipe._assemble(tmp_path, job)

    @patch("src.v2v_replace_pipeline.ensure_clip_uploadable", side_effect=lambda p: p)
    @patch("src.v2v_replace_pipeline.probe_video_duration", return_value=5.0)
    @patch.object(V2VReplacePipeline, "_create_backend")
    def test_submit_segment_calls_multimodal(
        self, mock_factory, mock_probe, mock_ensure, tmp_path: Path
    ):
        backend = MagicMock()
        out = tmp_path / "segments" / "seg_001" / "output.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        backend.generate_multimodal.return_value = VideoResult(
            video_path=out,
            duration=5.0,
            width=1280,
            height=720,
            pending=False,
        )
        mock_factory.return_value = backend

        source = tmp_path / "segments" / "seg_001" / "source.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"v")
        char = tmp_path / "character_ref.png"
        char.write_bytes(b"i")

        job = V2VReplaceJob(
            run_id="t1",
            source_video=str(tmp_path / "in.mp4"),
            character_image=str(char),
            prompt="replace hero",
        )
        seg = V2VSegment(
            id=1,
            start_sec=0,
            end_sec=5,
            source_clip=str(source),
        )
        pipe = V2VReplacePipeline(config={"director": {"videogen": {}}})
        pipe._submit_segment(tmp_path, job, seg, backend)

        backend.generate_multimodal.assert_called_once()
        kwargs = backend.generate_multimodal.call_args.kwargs
        assert kwargs["image_paths"] == [char]
        assert kwargs["video_paths"] == [source]
        assert backend.generate_multimodal.call_args.args[0] == "replace hero"
        assert seg.status == "done"
        assert seg.output_clip

    @patch("src.v2v_replace_pipeline.extract_last_frame")
    @patch("src.v2v_replace_pipeline.ensure_clip_uploadable", side_effect=lambda p: p)
    @patch("src.v2v_replace_pipeline.probe_video_duration", return_value=5.0)
    @patch.object(V2VReplacePipeline, "_create_backend")
    def test_submit_segment_second_uses_prev_output_anchor(
        self, mock_factory, mock_probe, mock_ensure, mock_extract, tmp_path: Path
    ):
        backend = MagicMock()
        out2 = tmp_path / "segments" / "seg_002" / "output.mp4"
        out2.parent.mkdir(parents=True, exist_ok=True)
        out2.write_bytes(b"mp4")
        backend.generate_multimodal.return_value = VideoResult(
            video_path=out2,
            duration=5.0,
            width=1280,
            height=720,
            pending=False,
        )
        mock_factory.return_value = backend

        prev_out = tmp_path / "segments" / "seg_001" / "output.mp4"
        prev_out.parent.mkdir(parents=True, exist_ok=True)
        prev_out.write_bytes(b"prev")
        source = tmp_path / "segments" / "seg_002" / "source.mp4"
        source.write_bytes(b"v")
        char = tmp_path / "character_ref.png"
        char.write_bytes(b"i")
        anchor = tmp_path / "segments" / "seg_002" / "anchor_frame.png"

        def fake_extract(src, dst):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"anchor")
            return dst

        mock_extract.side_effect = fake_extract

        job = V2VReplaceJob(
            run_id="t1",
            source_video=str(tmp_path / "in.mp4"),
            character_image=str(char),
            prompt="replace hero",
            segment_anchor=True,
            segments=[
                V2VSegment(
                    id=1,
                    start_sec=0,
                    end_sec=5,
                    output_clip=str(prev_out),
                    status="done",
                ),
            ],
        )
        seg2 = V2VSegment(
            id=2,
            start_sec=5,
            end_sec=10,
            source_clip=str(source),
        )
        pipe = V2VReplacePipeline(config={"director": {"videogen": {}}})
        pipe._submit_segment(tmp_path, job, seg2, backend)

        mock_extract.assert_called_once_with(prev_out, anchor)
        kwargs = backend.generate_multimodal.call_args.kwargs
        assert kwargs["image_paths"] == [anchor, char]
        assert "衔接锚点" in backend.generate_multimodal.call_args.args[0]
        assert seg2.anchor_frame == str(anchor)
        assert seg2.status == "done"

    @patch("src.v2v_replace_pipeline.ensure_clip_uploadable", side_effect=lambda p: p)
    @patch("src.v2v_replace_pipeline.probe_video_duration", return_value=5.0)
    @patch.object(V2VReplacePipeline, "_create_backend")
    def test_submit_segment_anchor_disabled_skips_prev_frame(
        self, mock_factory, mock_probe, mock_ensure, tmp_path: Path
    ):
        backend = MagicMock()
        out2 = tmp_path / "segments" / "seg_002" / "output.mp4"
        out2.parent.mkdir(parents=True, exist_ok=True)
        out2.write_bytes(b"mp4")
        backend.generate_multimodal.return_value = VideoResult(
            video_path=out2,
            duration=5.0,
            width=1280,
            height=720,
            pending=False,
        )
        mock_factory.return_value = backend

        prev_out = tmp_path / "segments" / "seg_001" / "output.mp4"
        prev_out.parent.mkdir(parents=True, exist_ok=True)
        prev_out.write_bytes(b"prev")
        source = tmp_path / "segments" / "seg_002" / "source.mp4"
        source.write_bytes(b"v")
        char = tmp_path / "character_ref.png"
        char.write_bytes(b"i")

        job = V2VReplaceJob(
            run_id="t1",
            source_video=str(tmp_path / "in.mp4"),
            character_image=str(char),
            prompt="replace hero",
            segment_anchor=False,
            segments=[
                V2VSegment(
                    id=1,
                    start_sec=0,
                    end_sec=5,
                    output_clip=str(prev_out),
                    status="done",
                ),
            ],
        )
        seg2 = V2VSegment(id=2, start_sec=5, end_sec=10, source_clip=str(source))
        pipe = V2VReplacePipeline(config={"director": {"videogen": {}}})
        with patch("src.v2v_replace_pipeline.extract_last_frame") as mock_extract:
            pipe._submit_segment(tmp_path, job, seg2, backend)
            mock_extract.assert_not_called()

        kwargs = backend.generate_multimodal.call_args.kwargs
        assert kwargs["image_paths"] == [char]
        assert backend.generate_multimodal.call_args.args[0] == "replace hero"

    def test_previous_segment_ready(self, tmp_path: Path):
        prev_out = tmp_path / "o1.mp4"
        prev_out.write_bytes(b"x")
        job = V2VReplaceJob(
            run_id="x",
            source_video="/a.mp4",
            character_image="/c.png",
            prompt="p",
            segments=[
                V2VSegment(
                    id=1,
                    start_sec=0,
                    end_sec=5,
                    output_clip=str(prev_out),
                    status="done",
                ),
                V2VSegment(id=2, start_sec=5, end_sec=10),
            ],
        )
        assert V2VReplacePipeline._previous_segment_ready(job, job.segments[0]) is True
        assert V2VReplacePipeline._previous_segment_ready(job, job.segments[1]) is True
        job.segments[0].output_clip = ""
        assert V2VReplacePipeline._previous_segment_ready(job, job.segments[1]) is False

    @patch("src.v2v_replace_pipeline.concat_videos")
    def test_assemble_appends_extension_clip(self, mock_concat, tmp_path: Path):
        seg_out = tmp_path / "seg1.mp4"
        ext_out = tmp_path / "ext.mp4"
        seg_out.write_bytes(b"seg")
        ext_out.write_bytes(b"ext")
        mock_concat.return_value = tmp_path / "final.mp4"

        job = V2VReplaceJob(
            run_id="abc",
            source_video="/x.mp4",
            character_image="/c.png",
            prompt="test",
            segments=[
                V2VSegment(
                    id=1,
                    start_sec=0,
                    end_sec=5,
                    output_clip=str(seg_out),
                    status="done",
                )
            ],
            extension=V2VExtension(
                prompt="continue scene",
                output_clip=str(ext_out),
                status="done",
            ),
        )
        pipe = V2VReplacePipeline(config={"director": {"videogen": {}}})
        pipe._assemble(tmp_path, job)
        clips = mock_concat.call_args[0][0]
        assert clips == [seg_out, ext_out]

    @patch("src.v2v_replace_pipeline.extract_last_frame")
    @patch("src.v2v_replace_pipeline.ensure_clip_uploadable", side_effect=lambda p: p)
    @patch("src.v2v_replace_pipeline.probe_video_duration", return_value=5.0)
    @patch.object(V2VReplacePipeline, "_create_backend")
    def test_process_extension_uses_last_segment_frame_as_anchor(
        self, mock_factory, mock_probe, mock_ensure, mock_extract, tmp_path: Path
    ):
        backend = MagicMock()
        ext_out = tmp_path / "extension" / "output.mp4"
        ext_out.parent.mkdir(parents=True, exist_ok=True)
        ext_out.write_bytes(b"mp4")
        backend.generate.return_value = VideoResult(
            video_path=ext_out,
            duration=5.0,
            width=1280,
            height=720,
            pending=False,
        )
        mock_factory.return_value = backend

        last_seg_out = tmp_path / "segments" / "seg_002" / "output.mp4"
        last_seg_out.parent.mkdir(parents=True, exist_ok=True)
        last_seg_out.write_bytes(b"last")

        job = V2VReplaceJob(
            run_id="t2",
            source_video=str(tmp_path / "in.mp4"),
            character_image=str(tmp_path / "char.png"),
            prompt="replace",
            segments=[
                V2VSegment(
                    id=1,
                    start_sec=0,
                    end_sec=5,
                    output_clip=str(tmp_path / "seg1.mp4"),
                    status="done",
                ),
                V2VSegment(
                    id=2,
                    start_sec=5,
                    end_sec=10,
                    output_clip=str(last_seg_out),
                    status="done",
                ),
            ],
            extension=V2VExtension(prompt="狸花猫继续探索", duration_sec=5.0, status="pending"),
        )
        pipe = V2VReplacePipeline(config={"director": {"videogen": {}}})
        pipe._process_extension(tmp_path, job, backend, progress_callback=None, start_pct=0.1)

        mock_extract.assert_called_once()
        assert mock_extract.call_args[0][0] == last_seg_out
        backend.generate.assert_called_once()
        kwargs = backend.generate.call_args.kwargs
        assert kwargs["image_path"] == tmp_path / "extension" / "anchor_frame.png"
        assert job.extension.status == "done"
        assert job.extension.output_clip

    def test_pending_count_includes_extension(self):
        job = V2VReplaceJob(
            run_id="x",
            source_video="/a.mp4",
            character_image="/c.png",
            prompt="p",
            segments=[
                V2VSegment(id=1, start_sec=0, end_sec=5, output_clip="/o.mp4", status="done")
            ],
            extension=V2VExtension(prompt="extend", status="pending"),
        )
        assert job.pending_count() == 1

    @patch("src.v2v_replace_pipeline.probe_video_duration", return_value=10.0)
    @patch("src.v2v_replace_pipeline.extract_video_clip")
    @patch("src.v2v_replace_pipeline.extract_audio_clip", return_value=None)
    @patch("src.v2v_replace_pipeline.plan_segment_spans")
    @patch.object(V2VReplacePipeline, "_create_backend")
    @patch.object(V2VReplacePipeline, "_process_segments")
    def test_run_stores_extension_in_job(
        self,
        mock_process,
        mock_backend,
        mock_plan,
        mock_audio,
        mock_extract,
        mock_probe,
        tmp_path,
        mock_video_file,
        mock_char_image,
    ):
        from src.video.video_splitter import VideoSegmentSpan

        mock_plan.return_value = [VideoSegmentSpan(id=1, start_sec=0.0, end_sec=10.0)]
        mock_process.return_value = {"status": "completed", "video_path": "/out.mp4"}
        mock_extract.return_value = tmp_path / "clip.mp4"

        pipe = V2VReplacePipeline(
            config={
                "llm": {},
                "imagegen": {},
                "director": {"videogen": {"backend": "jimeng-cli"}},
                "v2v_replace": {},
            },
            workspace=tmp_path / "ws",
        )
        pipe.run(
            mock_video_file,
            mock_char_image,
            extension_prompt="继续走动",
            extension_duration=4.0,
        )
        run_dir = Path(mock_process.call_args[0][0])
        job = V2VReplaceJob.model_validate_json(
            (run_dir / "job.json").read_text(encoding="utf-8")
        )
        assert job.extension is not None
        assert job.extension.prompt == "继续走动"
        assert job.extension.duration_sec == 4.0
        assert job.extension.status == "pending"
