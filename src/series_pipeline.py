"""分集系列视频流水线 — 跨集共享角色注册表与 POV 叙述者。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from src.agent_pipeline import AgentPipeline
from src.logger import log
from src.promptgen.character_registry import CharacterRegistry


@dataclass
class SeriesEpisode:
    id: str
    file: Path
    pov_narrator: str | None = None
    title: str | None = None


@dataclass
class SeriesConfig:
    title: str
    workspace: Path
    output_dir: Path
    registry_path: Path
    episodes: list[SeriesEpisode]
    era: str | None = None

    @classmethod
    def load(cls, path: Path, *, base_dir: Path | None = None) -> SeriesConfig:
        path = Path(path)
        root = base_dir or path.parent
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"无效的 series 配置: {path}")

        title = str(data.get("title") or path.parent.name)
        workspace = Path(data.get("workspace") or f"workspace/{title}")
        if not workspace.is_absolute():
            workspace = Path.cwd() / workspace

        output_dir = Path(data.get("output_dir") or f"output/{title}")
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        registry_rel = data.get("registry") or "character_registry.json"
        registry_path = workspace / registry_rel

        episodes: list[SeriesEpisode] = []
        for raw in data.get("episodes") or []:
            if not isinstance(raw, dict):
                continue
            ep_id = str(raw.get("id") or raw.get("episode") or len(episodes) + 1)
            file_val = raw.get("file")
            if not file_val:
                raise ValueError(f"分集 {ep_id} 缺少 file 字段")
            ep_file = Path(file_val)
            if not ep_file.is_absolute():
                ep_file = Path.cwd() / ep_file
            pov = raw.get("pov_narrator") or raw.get("pov")
            pov_str = str(pov).strip() if pov else None
            episodes.append(
                SeriesEpisode(
                    id=ep_id,
                    file=ep_file,
                    pov_narrator=pov_str,
                    title=str(raw.get("title")).strip() if raw.get("title") else None,
                )
            )

        if not episodes:
            raise ValueError(f"series 配置未定义 episodes: {path}")

        return cls(
            title=title,
            workspace=workspace,
            output_dir=output_dir,
            registry_path=registry_path,
            episodes=episodes,
            era=str(data.get("era")).strip() if data.get("era") else None,
        )


class SeriesPipeline:
    """按顺序跑多集 Agent 流水线，共享 character_registry.json。"""

    def __init__(
        self,
        series_config: Path,
        *,
        config_path: Path | None = None,
        resume: bool = False,
        budget_mode: bool = False,
        quality_threshold: float | None = None,
        start_episode: str | int | None = None,
        end_episode: str | int | None = None,
        config: dict | None = None,
    ):
        self.series = SeriesConfig.load(series_config)
        self.config_path = config_path
        self.resume = resume
        self.budget_mode = budget_mode
        self.quality_threshold = quality_threshold
        self.extra_config = config
        self.start_episode = start_episode
        self.end_episode = end_episode

        self.series.workspace.mkdir(parents=True, exist_ok=True)
        self.series.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.series.registry_path.exists():
            CharacterRegistry(self.series.registry_path).save()

    def _episode_slice(self) -> list[SeriesEpisode]:
        eps = self.series.episodes
        if self.start_episode is None and self.end_episode is None:
            return eps

        def _key(ep: SeriesEpisode) -> str:
            return ep.id

        start = str(self.start_episode) if self.start_episode is not None else _key(eps[0])
        end = str(self.end_episode) if self.end_episode is not None else _key(eps[-1])
        ids = [_key(e) for e in eps]
        if start not in ids or end not in ids:
            raise ValueError(f"起止分集不在配置内: {start} ~ {end}")
        i0, i1 = ids.index(start), ids.index(end)
        if i0 > i1:
            i0, i1 = i1, i0
        return eps[i0 : i1 + 1]

    def run(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[Path]:
        results: list[Path] = []
        episodes = self._episode_slice()
        total = len(episodes)

        for idx, ep in enumerate(episodes, start=1):
            if not ep.file.exists():
                raise FileNotFoundError(f"分集文本不存在: {ep.file}")

            ep_workspace = self.series.workspace / ep.id
            output_video = self.series.output_dir / f"{ep.id}.mp4"
            label = ep.title or ep.file.stem

            log.info(
                "系列 [%s] 分集 %s (%d/%d) POV=%s",
                self.series.title,
                ep.id,
                idx,
                total,
                ep.pov_narrator or "auto",
            )

            if progress_callback:
                progress_callback(idx, total, f"{ep.id} {label}")

            pipe = AgentPipeline(
                input_file=ep.file,
                config_path=self.config_path,
                output_dir=self.series.output_dir,
                workspace=ep_workspace,
                exact_workspace=True,
                resume=self.resume,
                budget_mode=self.budget_mode,
                quality_threshold=self.quality_threshold,
                config=self.extra_config,
                series_registry_path=self.series.registry_path,
                episode_id=ep.id,
                pov_narrator=ep.pov_narrator,
                output_video=output_video,
                era_override=self.series.era,
            )
            results.append(pipe.run())

        return results
