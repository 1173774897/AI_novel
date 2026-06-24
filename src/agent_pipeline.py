"""Agent 模式流水线入口"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.agent_state_repair import dedupe_completed_nodes, repair_agent_state_data
from src.agents.state import AgentState
from src.agents.utils import save_decisions_to_file
from src.config_manager import load_config, resolve_pipeline_config
from src.logger import log


class AgentPipeline:
    """LangGraph Agent 模式流水线"""

    def __init__(
        self,
        input_file: Path,
        config_path: Path | None = None,
        output_dir: Path | None = None,
        workspace: Path | None = None,
        resume: bool = False,
        budget_mode: bool = False,
        quality_threshold: float | None = None,
        config: dict | None = None,
        *,
        series_registry_path: Path | str | None = None,
        episode_id: str | int | None = None,
        pov_narrator: str | None = None,
        output_video: Path | str | None = None,
        exact_workspace: bool = False,
        era_override: str | None = None,
    ):
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {self.input_file}")

        # 加载配置
        base_cfg = load_config(config_path)
        if config:
            from src.pipeline import _deep_merge

            base_cfg = _deep_merge(base_cfg, config)
        self.cfg = resolve_pipeline_config(base_cfg, "agent")

        # 覆盖质量阈值
        if quality_threshold is not None:
            self.cfg.setdefault("agent", {}).setdefault("quality_check", {})[
                "threshold"
            ] = quality_threshold

        # 工作目录
        proj_name = self.input_file.stem
        base = Path(
            self.cfg.get("project", {}).get("default_workspace", "workspace")
        )
        if workspace is not None and exact_workspace:
            self.workspace = Path(workspace)
        else:
            self.workspace = (workspace or base) / proj_name
        self.workspace.mkdir(parents=True, exist_ok=True)

        # 子目录
        for sub in ["segments", "images", "audio", "subtitles", "videos"]:
            (self.workspace / sub).mkdir(parents=True, exist_ok=True)

        self.output_dir = (
            Path(output_dir)
            if output_dir
            else Path(self.cfg.get("project", {}).get("default_output", "output"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.resume = resume
        self.budget_mode = budget_mode

        self.series_registry_path = (
            str(series_registry_path) if series_registry_path else None
        )
        self.episode_id = episode_id
        self.pov_narrator = pov_narrator
        self.output_video = str(output_video) if output_video else None
        cfg_era = self.cfg.get("promptgen", {}).get("era")
        self.era_override = era_override or (
            cfg_era if cfg_era and str(cfg_era).lower() != "auto" else None
        )

        # State 持久化路径
        self.state_file = self.workspace / "agent_state.json"
        self.decisions_file = self.workspace / "agent_decisions.json"

    def _init_state(self) -> AgentState:
        """初始化 AgentState"""
        text = self.input_file.read_text(encoding="utf-8")
        return AgentState(
            input_file=str(self.input_file),
            config=self.cfg,
            workspace=str(self.workspace),
            mode="agent",
            budget_mode=self.budget_mode,
            resume=self.resume,
            full_text=text,
            genre=None,
            era=None,
            characters=None,
            suggested_style=None,
            segments=[],
            prompts=[],
            images=[],
            video_clips=None,
            audio_files=[],
            srt_files=[],
            final_video=None,
            quality_scores=[],
            retry_counts={},
            decisions=[],
            errors=[],
            completed_nodes=[],
            pipeline_plan=None,
            series_registry_path=self.series_registry_path,
            episode_id=self.episode_id,
            pov_narrator=self.pov_narrator,
            output_video=self.output_video,
            era_override=self.era_override,
        )

    def _load_state(self) -> AgentState | None:
        """加载已保存的 State"""
        if not self.state_file.exists():
            return None
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            # 恢复配置（不从文件加载，用当前配置）
            data["config"] = self.cfg

            old_workspace = data.get("workspace")
            new_workspace = str(self.workspace)
            data["workspace"] = new_workspace
            if self.series_registry_path:
                data["series_registry_path"] = self.series_registry_path
            if self.episode_id is not None:
                data["episode_id"] = self.episode_id
            if self.pov_narrator:
                data["pov_narrator"] = self.pov_narrator
            if self.output_video:
                data["output_video"] = self.output_video
            if self.era_override:
                data["era_override"] = self.era_override

            if old_workspace and Path(old_workspace).resolve() != Path(new_workspace).resolve():
                log.warning(
                    "workspace 路径已变更 (%s → %s)，丢弃旧产物路径",
                    old_workspace,
                    new_workspace,
                )
                data["images"] = []
                data["audio_files"] = []
                data["srt_files"] = []
                data["quality_scores"] = []
                data["retry_counts"] = {}
                completed = data.get("completed_nodes") or []
                data["completed_nodes"] = [
                    n for n in completed
                    if n in ("director", "content_analyzer")
                ]

            data["completed_nodes"] = dedupe_completed_nodes(
                data.get("completed_nodes")
            )
            data = repair_agent_state_data(data, self.cfg, new_workspace)

            return AgentState(**data)
        except Exception as e:
            log.warning("加载 Agent State 失败: %s", e)
            return None

    def _save_state(self, state: dict) -> None:
        """保存 State 到文件（原子写入）"""
        # 不保存 config（太大且含不可序列化对象）
        save_data = {k: v for k, v in state.items() if k != "config"}
        save_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(save_data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(self.state_file)

    def run(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        """运行 Agent 流水线"""
        log.info(
            "启动 Agent 模式: %s (budget=%s)",
            self.input_file.name,
            self.budget_mode,
        )

        # 初始化或恢复 State
        if self.resume:
            state = self._load_state()
            if state:
                log.info("[断点续传] 从已保存状态恢复")
            else:
                state = self._init_state()
        else:
            state = self._init_state()

        # 构建并执行图
        from src.agents.graph import create_agent_graph

        graph = create_agent_graph(self.cfg)

        total_stages = 5

        def _notify(stage: int, desc: str):
            if progress_callback:
                progress_callback(stage, total_stages, desc)

        _notify(1, "Agent 分析中...")

        try:
            result = graph.invoke(
                dict(state),
                config={"configurable": {"pipeline": self}},
            )
        except Exception as e:
            log.error("Agent 流水线失败: %s", e)
            raise

        # 保存最终状态和决策日志
        self._save_state(result)
        save_decisions_to_file(result, self.decisions_file)

        final_video = result.get("final_video")
        if final_video:
            log.info("Agent 模式完成: %s", final_video)
            return Path(final_video)
        else:
            raise RuntimeError("Agent 流水线未生成视频")
