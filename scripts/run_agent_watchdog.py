#!/usr/bin/env python3
"""Agent 流水线守护进程：失败或中断后自动 --resume 重试。

用法:
  python scripts/run_agent_watchdog.py input/无尽恶意.txt
  nohup python scripts/run_agent_watchdog.py input/无尽恶意.txt > /tmp/wujin_watchdog.out 2>&1 &

停止:
  kill <watchdog_pid>
  或 touch workspace/<项目名>/.watchdog_stop
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _find_other_pipeline_pid(input_file: Path, exclude_pid: int | None = None) -> int | None:
    """查找同输入文件、非本守护进程子进程的其他 main.py run 实例。"""
    resolved = str(input_file.resolve())
    name = input_file.name
    try:
        out = subprocess.check_output(
            ["pgrep", "-fl", "main.py run"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    for line in out.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_s, cmd = parts
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if exclude_pid and pid == exclude_pid:
            continue
        if pid == os.getpid():
            continue
        if name not in cmd:
            continue
        if resolved in cmd or str(input_file) in cmd or name in cmd:
            return pid
    return None


def _default_workspace(input_file: Path) -> Path:
    return PROJECT_ROOT / "workspace" / input_file.stem


def _default_log_file(input_file: Path) -> Path:
    return _default_workspace(input_file) / ".watchdog.log"


def _default_lock_file(input_file: Path) -> Path:
    return _default_workspace(input_file) / ".watchdog.lock"


def _default_stop_file(input_file: Path) -> Path:
    return _default_workspace(input_file) / ".watchdog_stop"


class PipelineWatchdog:
    """循环执行 Agent 流水线，非零退出时自动带 --resume 重试。"""

    def __init__(
        self,
        input_file: Path,
        *,
        delay: float = 10.0,
        max_retries: int = 0,
        config: str | None = None,
        workspace: str | None = None,
        budget_mode: bool = False,
        log_file: Path | None = None,
        lock_file: Path | None = None,
        stop_file: Path | None = None,
        command: list[str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.input_file = input_file.resolve()
        self.delay = max(0.0, delay)
        self.max_retries = max(0, max_retries)
        self.config = config
        self.workspace = workspace
        self.budget_mode = budget_mode
        self.log_file = log_file or _default_log_file(self.input_file)
        self.lock_file = lock_file or _default_lock_file(self.input_file)
        self.stop_file = stop_file or _default_stop_file(self.input_file)
        self._command_override = command
        self.cwd = cwd or PROJECT_ROOT
        self._child: subprocess.Popen[bytes] | None = None
        self._stop_requested = False

    def build_command(self) -> list[str]:
        if self._command_override is not None:
            return list(self._command_override)
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            "run",
            str(self.input_file),
            "--mode",
            "agent",
            "--resume",
        ]
        if self.config:
            cmd.extend(["--config", self.config])
        if self.workspace:
            cmd.extend(["--workspace", self.workspace])
        if self.budget_mode:
            cmd.append("--budget-mode")
        return cmd

    def log(self, message: str) -> None:
        line = f"[{_utc_now()}] {message}"
        print(line, flush=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _should_stop(self) -> bool:
        return self._stop_requested or self.stop_file.exists()

    def _handle_signal(self, signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        self.log(f"收到 {name}，准备停止守护进程…")
        self._stop_requested = True
        if self._child is not None and self._child.poll() is None:
            self._child.terminate()

    def acquire_lock(self) -> None:
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_file.exists():
            try:
                old_pid = int(self.lock_file.read_text(encoding="utf-8").strip())
            except ValueError:
                old_pid = 0
            if old_pid and _pid_alive(old_pid):
                raise RuntimeError(
                    f"已有守护进程在运行 (pid={old_pid})，锁文件: {self.lock_file}"
                )
        self.lock_file.write_text(str(os.getpid()), encoding="utf-8")

    def release_lock(self) -> None:
        if not self.lock_file.exists():
            return
        try:
            owner = int(self.lock_file.read_text(encoding="utf-8").strip())
        except ValueError:
            owner = 0
        if owner == os.getpid():
            self.lock_file.unlink(missing_ok=True)

    def wait_for_other_pipeline(self) -> None:
        """若用户已手动启动流水线，等待其结束后再接管（自动 --resume）。"""
        while not self._should_stop():
            other = _find_other_pipeline_pid(self.input_file)
            if other is None:
                return
            self.log(
                f"检测到已有流水线 pid={other}，等待结束后自动 --resume 接管…"
            )
            while _pid_alive(other) and not self._should_stop():
                time.sleep(5.0)
            if self._should_stop():
                return
            # 给文件锁/状态落盘留一点时间
            time.sleep(2.0)

    def run_once(self) -> int:
        cmd = self.build_command()
        self.log(f"启动: {' '.join(cmd)}")
        self._child = subprocess.Popen(cmd, cwd=self.cwd)
        return self._child.wait()

    def run(self) -> int:
        if self.stop_file.exists():
            self.stop_file.unlink(missing_ok=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)

        self.acquire_lock()
        attempt = 0
        last_code = 1
        try:
            self.wait_for_other_pipeline()
            if self._should_stop():
                self.log("启动前收到停止请求")
                return 128 + signal.SIGINT
            while not self._should_stop():
                self.wait_for_other_pipeline()
                if self._should_stop():
                    self.log("运行前收到停止请求")
                    return 128 + signal.SIGINT
                attempt += 1
                self.log(f"=== 第 {attempt} 次运行 ===")
                last_code = self.run_once()
                if last_code == 0:
                    self.log("流水线成功完成，守护进程退出")
                    return 0
                if self._should_stop():
                    self.log(f"收到停止请求，退出 (子进程 code={last_code})")
                    return 128 + signal.SIGINT
                if self.max_retries and attempt >= self.max_retries:
                    self.log(
                        f"已达最大重试次数 {self.max_retries}，退出 (code={last_code})"
                    )
                    return last_code
                self.log(
                    f"子进程退出码 {last_code}，{self.delay:.0f}s 后自动 --resume 重试…"
                )
                deadline = time.monotonic() + self.delay
                while time.monotonic() < deadline:
                    if self._should_stop():
                        self.log("等待重试期间收到停止请求")
                        return 128 + signal.SIGINT
                    time.sleep(0.5)
        finally:
            self.release_lock()
        return last_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent 流水线守护：失败/中断后自动 --resume 重试"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="input/无尽恶意.txt",
        help="输入小说路径（默认 input/无尽恶意.txt）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="失败后等待秒数再重试（默认 10）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="最大运行次数，0 表示无限重试（默认 0）",
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径")
    parser.add_argument("--workspace", "-w", default=None, help="工作目录")
    parser.add_argument("--budget-mode", action="store_true", help="省钱模式")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="日志文件（默认 workspace/<书名>/.watchdog.log）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.exists():
        print(f"输入文件不存在: {input_path}", file=sys.stderr)
        return 2

    watchdog = PipelineWatchdog(
        input_path,
        delay=args.delay,
        max_retries=args.max_retries,
        config=args.config,
        workspace=args.workspace,
        budget_mode=args.budget_mode,
        log_file=args.log_file,
    )
    try:
        return watchdog.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
