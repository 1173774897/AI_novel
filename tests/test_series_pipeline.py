"""SeriesPipeline 配置与分集切片测试。"""
from __future__ import annotations

import pytest

from src.series_pipeline import SeriesConfig, SeriesPipeline

pytestmark = pytest.mark.signature


@pytest.fixture
def series_yaml(tmp_path):
    ep_dir = tmp_path / "input" / "test_series"
    ep_dir.mkdir(parents=True)
    for i in range(1, 4):
        (ep_dir / f"0{i}.txt").write_text(f"episode {i}", encoding="utf-8")

    yaml_text = f"""
title: 测试系列
workspace: {tmp_path / "ws"}
output_dir: {tmp_path / "out"}
registry: character_registry.json
episodes:
  - id: ep01
    file: {ep_dir / "01.txt"}
    pov_narrator: 甲
  - id: ep02
    file: {ep_dir / "02.txt"}
    pov_narrator: 甲
  - id: ep03
    file: {ep_dir / "03.txt"}
    pov_narrator: 乙
"""
    path = tmp_path / "series.yaml"
    path.write_text(yaml_text.strip(), encoding="utf-8")
    return path


def test_series_config_loads_episodes(series_yaml):
    cfg = SeriesConfig.load(series_yaml)
    assert cfg.title == "测试系列"
    assert len(cfg.episodes) == 3
    assert cfg.episodes[0].pov_narrator == "甲"
    assert cfg.episodes[2].pov_narrator == "乙"


def test_series_pipeline_episode_slice(series_yaml):
    pipe = SeriesPipeline(series_yaml, start_episode="ep02", end_episode="ep03")
    eps = pipe._episode_slice()
    assert [e.id for e in eps] == ["ep02", "ep03"]


def test_series_config_workspace_from_project_root(tmp_path, monkeypatch):
    """series.yaml 在 input/<title>/ 下时，workspace 应相对项目根目录。"""
    monkeypatch.chdir(tmp_path)
    series_dir = tmp_path / "input" / "东城暮雪"
    series_dir.mkdir(parents=True)
    (series_dir / "01.txt").write_text("ep1", encoding="utf-8")
    yaml_path = series_dir / "series.yaml"
    yaml_path.write_text(
        """
title: 东城暮雪
workspace: workspace/东城暮雪
output_dir: output/东城暮雪
registry: character_registry.json
episodes:
  - id: ep01
    file: input/东城暮雪/01.txt
""".strip(),
        encoding="utf-8",
    )
    cfg = SeriesConfig.load(yaml_path)
    assert cfg.workspace == tmp_path / "workspace" / "东城暮雪"
    assert cfg.output_dir == tmp_path / "output" / "东城暮雪"
    assert cfg.registry_path == tmp_path / "workspace" / "东城暮雪" / "character_registry.json"
