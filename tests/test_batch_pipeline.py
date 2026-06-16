"""BatchPipeline 文件发现与批量调度测试。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.batch_pipeline import (
    BatchPipeline,
    BatchResult,
    discover_txt_files,
    slice_batch_files,
)

pytestmark = pytest.mark.signature


@pytest.fixture
def txt_dir(tmp_path):
    d = tmp_path / "stories"
    d.mkdir()
    (d / "03-c.txt").write_text("c", encoding="utf-8")
    (d / "01-a.txt").write_text("a", encoding="utf-8")
    (d / "02-b.txt").write_text("b", encoding="utf-8")
    (d / ".hidden.txt").write_text("x", encoding="utf-8")
    (d / "readme.md").write_text("md", encoding="utf-8")
    return d


def test_discover_txt_sorted(txt_dir):
    files = discover_txt_files(txt_dir)
    assert [p.name for p in files] == ["01-a.txt", "02-b.txt", "03-c.txt"]


def test_discover_empty_dir_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="没有可处理的"):
        slice_batch_files(discover_txt_files(empty))


def test_slice_by_index(txt_dir):
    files = discover_txt_files(txt_dir)
    items = slice_batch_files(files, start_index=2, end_index=3)
    assert [i.file.name for i in items] == ["02-b.txt", "03-c.txt"]
    assert [i.index for i in items] == [2, 3]


def test_slice_by_filename_stem(txt_dir):
    files = discover_txt_files(txt_dir)
    items = slice_batch_files(files, start_file="02-b", end_file="03-c.txt")
    assert [i.file.name for i in items] == ["02-b.txt", "03-c.txt"]


def test_slice_invalid_file_raises(txt_dir):
    files = discover_txt_files(txt_dir)
    with pytest.raises(ValueError, match="不在目录内"):
        slice_batch_files(files, start_file="missing.txt")


@patch("src.batch_pipeline.BatchPipeline._run_one")
def test_batch_run_all_success(mock_run_one, txt_dir, tmp_path):
    mock_run_one.side_effect = [
        tmp_path / "out" / "01-a.mp4",
        tmp_path / "out" / "02-b.mp4",
        tmp_path / "out" / "03-c.mp4",
    ]
    pipe = BatchPipeline(txt_dir, mode="agent")
    result = pipe.run()
    assert len(result.succeeded) == 3
    assert result.failed == []
    assert mock_run_one.call_count == 3


@patch("src.batch_pipeline.BatchPipeline._run_one")
def test_batch_continue_on_error(mock_run_one, txt_dir, tmp_path):
    mock_run_one.side_effect = [
        tmp_path / "01-a.mp4",
        RuntimeError("boom"),
        tmp_path / "03-c.mp4",
    ]
    pipe = BatchPipeline(txt_dir, continue_on_error=True)
    result = pipe.run()
    assert len(result.succeeded) == 2
    assert len(result.failed) == 1
    assert result.failed[0][0].name == "02-b.txt"


@patch("src.batch_pipeline.BatchPipeline._run_one")
def test_batch_abort_on_error(mock_run_one, txt_dir):
    mock_run_one.side_effect = RuntimeError("fail")
    pipe = BatchPipeline(txt_dir)
    with pytest.raises(RuntimeError, match="fail"):
        pipe.run()


@patch("src.agent_pipeline.AgentPipeline.run")
def test_batch_uses_agent_pipeline(mock_agent_run, txt_dir, tmp_path):
    mock_agent_run.return_value = tmp_path / "01-a.mp4"
    pipe = BatchPipeline(
        txt_dir,
        mode="agent",
        workspace_base=tmp_path / "ws",
        output_dir=tmp_path / "out",
        start_index=1,
        end_index=1,
    )
    result = pipe.run()
    assert isinstance(result, BatchResult)
    assert len(result.succeeded) == 1
    mock_agent_run.assert_called_once()


@patch("src.pipeline.Pipeline.run")
def test_batch_uses_classic_pipeline(mock_classic_run, txt_dir, tmp_path):
    mock_classic_run.return_value = tmp_path / "01-a.mp4"
    pipe = BatchPipeline(txt_dir, mode="classic", start_index=1, end_index=1)
    result = pipe.run()
    assert len(result.succeeded) == 1
    mock_classic_run.assert_called_once()
