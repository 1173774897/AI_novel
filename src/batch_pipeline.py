"""批量视频流水线 — 依次处理目录下所有 txt 文件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.logger import log


@dataclass
class BatchItem:
    file: Path
    index: int  # 1-based


@dataclass
class BatchResult:
    succeeded: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)


def discover_txt_files(
    input_dir: Path,
    *,
    pattern: str = "*.txt",
    recursive: bool = False,
) -> list[Path]:
    """列出目录内待处理的 txt 文件（按文件名排序）。"""
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {root}")

    if recursive:
        candidates = sorted(root.rglob(pattern))
    else:
        candidates = sorted(root.glob(pattern))

    files = [
        p.resolve()
        for p in candidates
        if p.is_file() and not p.name.startswith(".")
    ]
    return files


def slice_batch_files(
    files: list[Path],
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    start_file: str | None = None,
    end_file: str | None = None,
) -> list[BatchItem]:
    """按序号或起止文件名切片。"""
    if not files:
        raise ValueError("目录内没有可处理的 txt 文件")

    items = [BatchItem(file=f, index=i) for i, f in enumerate(files, start=1)]

    if start_file or end_file:
        names = [f.name for f in files]
        stems = [f.stem for f in files]

        def _resolve(key: str) -> int:
            if key in names:
                return names.index(key)
            if key in stems:
                return stems.index(key)
            raise ValueError(f"文件不在目录内: {key}")

        i0 = _resolve(start_file) if start_file else 0
        i1 = _resolve(end_file) if end_file else len(files) - 1
        if i0 > i1:
            i0, i1 = i1, i0
        items = items[i0 : i1 + 1]
    elif start_index is not None or end_index is not None:
        i0 = (start_index or 1) - 1
        i1 = (end_index or len(items)) - 1
        if i0 < 0 or i1 >= len(items) or i0 > i1:
            raise ValueError(
                f"序号范围无效: {start_index or 1} ~ {end_index or len(items)} "
                f"(共 {len(items)} 个文件)"
            )
        items = items[i0 : i1 + 1]

    return items


class BatchPipeline:
    """依次对目录内 txt 运行 classic / agent 流水线。"""

    def __init__(
        self,
        input_dir: Path,
        *,
        config_path: Path | None = None,
        output_dir: Path | None = None,
        workspace_base: Path | None = None,
        resume: bool = False,
        mode: str = "agent",
        budget_mode: bool = False,
        quality_threshold: float | None = None,
        pattern: str = "*.txt",
        recursive: bool = False,
        start_index: int | None = None,
        end_index: int | None = None,
        start_file: str | None = None,
        end_file: str | None = None,
        continue_on_error: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.config_path = config_path
        self.output_dir = Path(output_dir) if output_dir else None
        self.workspace_base = Path(workspace_base) if workspace_base else None
        self.resume = resume
        self.mode = mode
        self.budget_mode = budget_mode
        self.quality_threshold = quality_threshold
        self.pattern = pattern
        self.recursive = recursive
        self.start_index = start_index
        self.end_index = end_index
        self.start_file = start_file
        self.end_file = end_file
        self.continue_on_error = continue_on_error

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.workspace_base:
            self.workspace_base.mkdir(parents=True, exist_ok=True)

    def _list_items(self) -> list[BatchItem]:
        files = discover_txt_files(
            self.input_dir,
            pattern=self.pattern,
            recursive=self.recursive,
        )
        return slice_batch_files(
            files,
            start_index=self.start_index,
            end_index=self.end_index,
            start_file=self.start_file,
            end_file=self.end_file,
        )

    def _workspace_for(self, txt: Path) -> Path | None:
        if self.workspace_base is None:
            return None
        return self.workspace_base / txt.stem

    def _run_one(self, txt: Path) -> Path:
        ws = self._workspace_for(txt)
        if self.mode == "agent":
            from src.agent_pipeline import AgentPipeline

            pipe = AgentPipeline(
                input_file=txt,
                config_path=self.config_path,
                output_dir=self.output_dir,
                workspace=ws,
                exact_workspace=ws is not None,
                resume=self.resume,
                budget_mode=self.budget_mode,
                quality_threshold=self.quality_threshold,
            )
        else:
            from src.pipeline import Pipeline

            pipe = Pipeline(
                input_file=txt,
                config_path=self.config_path,
                output_dir=self.output_dir,
                workspace=ws,
                resume=self.resume,
            )
        return pipe.run()

    def run(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> BatchResult:
        items = self._list_items()
        total = len(items)
        result = BatchResult()

        log.info(
            "批量处理: %s (%d 个 txt, mode=%s)",
            self.input_dir,
            total,
            self.mode,
        )

        for seq, item in enumerate(items, start=1):
            label = item.file.name
            log.info("批量 [%d/%d] %s", seq, total, label)
            if progress_callback:
                progress_callback(seq, total, label)

            try:
                video = self._run_one(item.file)
                result.succeeded.append(video)
                log.info("批量完成 [%d/%d]: %s", seq, total, video)
            except Exception as exc:
                msg = str(exc)
                result.failed.append((item.file, msg))
                log.error("批量失败 [%d/%d] %s: %s", seq, total, label, msg)
                if not self.continue_on_error:
                    raise

        return result
