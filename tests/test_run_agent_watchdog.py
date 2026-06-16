"""run_agent_watchdog 守护进程测试。"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.run_agent_watchdog import (
    PipelineWatchdog,
    _find_other_pipeline_pid,
    _pid_alive,
)

pytestmark = pytest.mark.signature


def _make_watchdog(
    tmp_path: Path,
    *,
    command: list[str],
    delay: float = 0.1,
    max_retries: int = 0,
) -> PipelineWatchdog:
    input_file = tmp_path / "story.txt"
    input_file.write_text("hello", encoding="utf-8")
    ws = tmp_path / "workspace" / "story"
    ws.mkdir(parents=True, exist_ok=True)
    return PipelineWatchdog(
        input_file,
        delay=delay,
        max_retries=max_retries,
        workspace=str(ws),
        log_file=ws / ".watchdog.log",
        lock_file=ws / ".watchdog.lock",
        stop_file=ws / ".watchdog_stop",
        command=command,
        cwd=tmp_path,
    )


@pytest.mark.signature
def test_pid_alive_current_process():
    assert _pid_alive(os.getpid())


@pytest.mark.signature
def test_pid_alive_dead_process():
    assert not _pid_alive(999_999_999)


@pytest.mark.signature
def test_watchdog_exits_on_child_success(tmp_path: Path):
    cmd = [sys.executable, "-c", "raise SystemExit(0)"]
    wd = _make_watchdog(tmp_path, command=cmd)
    assert wd.run() == 0
    assert "成功完成" in wd.log_file.read_text(encoding="utf-8")


@pytest.mark.signature
def test_watchdog_retries_then_succeeds(tmp_path: Path):
    script = tmp_path / "flip.py"
    script.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            flag = Path(__import__('sys').argv[1])
            if not flag.exists():
                flag.write_text('1')
                raise SystemExit(1)
            raise SystemExit(0)
            """
        ),
        encoding="utf-8",
    )
    flag = tmp_path / "tried.flag"
    cmd = [sys.executable, str(script), str(flag)]
    wd = _make_watchdog(tmp_path, command=cmd, delay=0.05)
    assert wd.run() == 0
    log = wd.log_file.read_text(encoding="utf-8")
    assert "第 1 次运行" in log
    assert "第 2 次运行" in log


@pytest.mark.signature
def test_watchdog_respects_max_retries(tmp_path: Path):
    cmd = [sys.executable, "-c", "raise SystemExit(7)"]
    wd = _make_watchdog(tmp_path, command=cmd, max_retries=2, delay=0.05)
    assert wd.run() == 7
    log = wd.log_file.read_text(encoding="utf-8")
    assert "最大重试次数 2" in log


@pytest.mark.signature
def test_watchdog_stop_file(tmp_path: Path):
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    wd = _make_watchdog(tmp_path, command=cmd, delay=0.05)
    wd.acquire_lock()
    wd.stop_file.write_text("1", encoding="utf-8")
    try:
        assert wd._should_stop() is True
    finally:
        wd.release_lock()


@pytest.mark.signature
def test_watchdog_lock_prevents_duplicate(tmp_path: Path):
    cmd = [sys.executable, "-c", "raise SystemExit(0)"]
    a = _make_watchdog(tmp_path, command=cmd)
    b = _make_watchdog(tmp_path, command=cmd)
    a.acquire_lock()
    try:
        with pytest.raises(RuntimeError, match="已有守护进程"):
            b.acquire_lock()
    finally:
        a.release_lock()


@pytest.mark.signature
def test_find_other_pipeline_pid_ignores_missing(tmp_path: Path):
    input_file = tmp_path / "nope.txt"
    input_file.write_text("x", encoding="utf-8")
    assert _find_other_pipeline_pid(input_file) is None


@pytest.mark.signature
def test_build_command_includes_resume(tmp_path: Path):
    input_file = tmp_path / "无尽恶意.txt"
    input_file.write_text("x", encoding="utf-8")
    wd = PipelineWatchdog(input_file, workspace=str(tmp_path / "ws"))
    cmd = wd.build_command()
    assert "--resume" in cmd
    assert "--mode" in cmd
    assert "agent" in cmd
