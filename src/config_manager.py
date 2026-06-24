"""配置管理 - 加载和验证 YAML 配置"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml

from src.logger import log
from src.tts.voices import apply_tts_voice

PipelineMode = Literal["agent", "director"]

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

# imagegen.ratio -> video.resolution（1080p 基准）
_RATIO_TO_VIDEO_RESOLUTION: dict[str, list[int]] = {
    "9:16": [1080, 1920],
    "16:9": [1920, 1080],
    "1:1": [1080, 1080],
    "4:3": [1440, 1080],
    "3:4": [1080, 1440],
    "3:2": [1620, 1080],
    "2:3": [1080, 1620],
    "21:9": [2520, 1080],
}


def resolution_for_image_ratio(ratio: str) -> list[int] | None:
    """根据 imagegen.ratio 推导推荐成片分辨率。"""
    key = str(ratio).strip()
    return _RATIO_TO_VIDEO_RESOLUTION.get(key)


def _is_portrait_resolution(resolution: list[int]) -> bool:
    return len(resolution) == 2 and resolution[0] < resolution[1]


def _apply_video_resolution_from_imagegen(cfg: dict) -> None:
    """按 imagegen.ratio 同步或校验 video.resolution。"""
    imagegen = cfg.get("imagegen") or {}
    video = cfg.get("video") or {}
    ratio = imagegen.get("ratio")
    if not ratio:
        return

    mapped = resolution_for_image_ratio(str(ratio))
    if mapped is None:
        return

    current = video.get("resolution")
    auto = bool(video.get("auto_resolution", False))

    if auto:
        if current != mapped:
            log.info(
                "video.auto_resolution=true: 按 imagegen.ratio=%s 设置 video.resolution=%s",
                ratio,
                mapped,
            )
        video["resolution"] = mapped
        return

    if not isinstance(current, list) or len(current) != 2:
        return

    mapped_portrait = _is_portrait_resolution(mapped)
    current_portrait = _is_portrait_resolution(current)
    if mapped_portrait != current_portrait:
        log.warning(
            "imagegen.ratio=%s 与 video.resolution=%s 方向不一致，"
            "横图会被裁切进竖屏画幅；可改 video.resolution=%s 或设 video.auto_resolution: true",
            ratio,
            current,
            mapped,
        )


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base 副本。"""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_mode_videogen(cfg: dict[str, Any], mode: PipelineMode) -> dict[str, Any]:
    """按流水线模式读取 videogen 配置（agent 与 director 互不共用）。

    - ``director``：仅 ``director.videogen``
    - ``agent``：``agent.videogen``；若无则兼容顶层 ``videogen``（弃用警告）
    """
    if mode == "director":
        section = cfg.get("director") or {}
        vg = section.get("videogen")
        return dict(vg) if isinstance(vg, dict) else {}

    agent_section = cfg.get("agent") or {}
    agent_vg = agent_section.get("videogen")
    if isinstance(agent_vg, dict) and agent_vg.get("backend"):
        return dict(agent_vg)

    root_vg = cfg.get("videogen")
    if isinstance(root_vg, dict) and root_vg.get("backend"):
        log.warning(
            "顶层 videogen 仅对 run/agent 生效且已弃用，请迁移到 agent.videogen；"
            "create-video 请使用 director.videogen"
        )
        return dict(root_vg)

    return {}


def resolve_pipeline_config(
    cfg: dict[str, Any], mode: PipelineMode
) -> dict[str, Any]:
    """解析某条流水线可用的运行时配置（隔离 videogen / 可选模块覆盖）。

    共用：llm、segmenter、promptgen、tts 等顶层字段。
    隔离：``videogen`` 按模式写入；``director.imagegen`` / ``director.video`` 可覆盖顶层。
    """
    resolved = copy.deepcopy(cfg)
    section_key = "director" if mode == "director" else "agent"
    section = cfg.get(section_key) or {}
    if not isinstance(section, dict):
        section = {}

    resolved["videogen"] = get_mode_videogen(cfg, mode)

    if isinstance(section.get("imagegen"), dict):
        resolved["imagegen"] = _deep_merge(
            resolved.get("imagegen") or {}, section["imagegen"]
        )
    if isinstance(section.get("video"), dict):
        resolved["video"] = _deep_merge(
            resolved.get("video") or {}, section["video"]
        )
    if isinstance(section.get("tts"), dict):
        resolved["tts"] = _deep_merge(resolved.get("tts") or {}, section["tts"])

    _apply_video_resolution_from_imagegen(resolved)
    apply_tts_voice(resolved)
    return resolved


def resolve_v2v_replace_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """v2v 角色替换：共用 director 的 imagegen/videogen，并读取 v2v_replace 段。"""
    resolved = resolve_pipeline_config(cfg, "director")
    v2v = cfg.get("v2v_replace")
    if isinstance(v2v, dict):
        resolved["v2v_replace"] = dict(v2v)
        if isinstance(v2v.get("videogen"), dict):
            resolved["videogen"] = _deep_merge(
                resolved.get("videogen") or {}, v2v["videogen"]
            )
    else:
        resolved["v2v_replace"] = {}
    return resolved


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"配置文件内容无效（期望字典，得到 {type(cfg).__name__}）: {path}"
        )
    _validate(cfg)
    _apply_video_resolution_from_imagegen(cfg)
    apply_tts_voice(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    required_sections = ["segmenter", "promptgen", "imagegen", "tts", "video"]
    for sec in required_sections:
        if sec not in cfg:
            raise ValueError(f"配置缺少必要字段: {sec}")

    res = cfg["video"].get("resolution")
    if not (isinstance(res, list) and len(res) == 2):
        raise ValueError("video.resolution 必须是 [width, height]")

    # agent 配置（可选）
    agent_cfg = cfg.get("agent")
    if agent_cfg is not None:
        if not isinstance(agent_cfg, dict):
            raise ValueError("agent 配置必须是字典")
        _validate_agent(agent_cfg)


def _validate_agent(agent_cfg: dict) -> None:
    """验证 agent 配置子字段。"""
    qc = agent_cfg.get("quality_check")
    if qc is not None:
        if not isinstance(qc, dict):
            raise ValueError("agent.quality_check 必须是字典")

        threshold = qc.get("threshold")
        if threshold is not None:
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 10):
                raise ValueError("agent.quality_check.threshold 必须在 0-10 之间")

        max_retries = qc.get("max_retries")
        if max_retries is not None:
            if not isinstance(max_retries, int) or not (0 <= max_retries <= 10):
                raise ValueError("agent.quality_check.max_retries 必须是 0-10 的整数")

        vision_provider = qc.get("vision_provider")
        if vision_provider is not None and vision_provider not in ("openai", "gemini"):
            raise ValueError("agent.quality_check.vision_provider 必须是 openai 或 gemini")

    decisions = agent_cfg.get("decisions")
    if decisions is not None:
        if not isinstance(decisions, dict):
            raise ValueError("agent.decisions 必须是字典")

    budget = agent_cfg.get("budget_mode")
    if budget is not None:
        if not isinstance(budget, dict):
            raise ValueError("agent.budget_mode 必须是字典")

        _budget_bool_fields = [
            "disable_quality_check",
            "use_cheap_llm",
            "simple_emotion_analysis",
        ]
        for field in _budget_bool_fields:
            val = budget.get(field)
            if val is not None and not isinstance(val, bool):
                raise ValueError(
                    f"agent.budget_mode.{field} 必须是布尔值"
                )
