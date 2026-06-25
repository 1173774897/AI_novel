"""角色外观分段规划 — 规则兜底 + 可选 LLM 补充 + 审核员定稿。"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from src.agents.utils import extract_json_obj
from src.promptgen.visual_state import (
    attach_visual_states,
    normalize_visual_states,
    plan_visual_states_by_rules,
    prune_static_visual_states,
    segments_summary_for_prompt,
)
from src.promptgen.visual_state_reviewer import (
    run_visual_state_review_discussion,
    visual_state_review_enabled,
)

log = logging.getLogger("novel")

_VISUAL_STATE_PROMPT = """\
你是短视频分镜角色外观规划专家。根据全文分段，为需要「前后外观变化」的角色规划 visual_states。

规则：
- 仅当角色 desc 含明显阶段差异（如前期怀孕/后期清瘦、受伤前后）才输出 visual_states
- 每个 state 含 id、desc（完整中文外观 80-150 字）、effective_from_segment（含）、deprecated_at_segment（不含，可选）
- 切换点须与剧情一致（如人流/手术/事故等锚点段）
- desc 须互斥：前期状态不可含后期特征，反之亦然；共享基础信息（年龄性别发型）各写一次即可，勿整段复制
- 无阶段差异的角色不要输出

角色列表（JSON）：
{characters_json}

分段摘要（index + 前 120 字）：
{segments_summary}

输出 JSON 对象：{{"visual_states": {{"角色名": [{{"id":"...", "desc":"...", "effective_from_segment":0, "deprecated_at_segment":12, "trigger":"..."}}]}}}}
无需要规划的角色时输出 {{"visual_states": {{}}}}"""


class _ChatLLM(Protocol):
    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = ...) -> Any: ...


def _segments_summary(segments: list[dict[str, Any]], *, max_chars: int = 120) -> str:
    return segments_summary_for_prompt(segments, max_chars=max_chars)


def _plan_with_llm(
    llm: _ChatLLM,
    characters: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    import json

    try:
        prompt = _VISUAL_STATE_PROMPT.format(
            characters_json=json.dumps(characters, ensure_ascii=False)[:6000],
            segments_summary=_segments_summary(segments),
        )
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            json_mode=True,
        )
        data = extract_json_obj(response.content)
        if not data or not isinstance(data.get("visual_states"), dict):
            return {}
        out: dict[str, list[dict[str, Any]]] = {}
        for name, raw_states in data["visual_states"].items():
            name = str(name).strip()
            if not name or not isinstance(raw_states, list):
                continue
            normalized = normalize_visual_states(raw_states)
            if normalized:
                out[name] = normalized
        return out
    except Exception as exc:
        log.warning("[VisualStatePlanner] LLM 规划失败 (%s)，仅用规则结果", exc)
        return {}


def _review_visual_states(
    text: str,
    characters: list[dict[str, Any]],
    draft: dict[str, list[dict[str, Any]]],
    segments: list[dict[str, Any]],
    *,
    config: dict,
    primary_llm: _ChatLLM,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    from src.agents.character_reviewer import create_reviewer_llm

    try:
        reviewer_llm, reviewer_provider, same_source = create_reviewer_llm(config)
    except Exception as exc:
        log.warning("[VisualStatePlanner] 审核 LLM 不可用 (%s)，跳过讨论", exc)
        return draft, []

    result = run_visual_state_review_discussion(
        text,
        characters,
        draft,
        segments,
        primary_llm=primary_llm,
        reviewer_llm=reviewer_llm,
        reviewer_provider=reviewer_provider,
        same_source=same_source,
    )
    return result.visual_states or draft, result.discussion


def plan_visual_states(
    characters: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    llm: _ChatLLM | None = None,
    budget_mode: bool = False,
    config: dict | None = None,
    full_text: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """规划各角色 visual_states；规则兜底 → 可选 LLM → 审核员 B 定稿。"""
    discussion: list[str] = []
    rule_result = plan_visual_states_by_rules(characters, segments)
    merged: dict[str, list[dict[str, Any]]] = {
        name: normalize_visual_states(states)
        for name, states in rule_result.items()
        if normalize_visual_states(states)
    }

    if not budget_mode and llm is not None:
        llm_result = _plan_with_llm(llm, characters, segments)
        for name, states in llm_result.items():
            if name not in merged and states:
                merged[name] = states

    if not merged:
        return {}, discussion

    if (
        merged
        and config
        and full_text
        and llm is not None
        and visual_state_review_enabled(config, budget_mode)
    ):
        merged, discussion = _review_visual_states(
            full_text,
            characters,
            merged,
            segments,
            config=config,
            primary_llm=llm,
        )

    if merged:
        log.info(
            "[VisualStatePlanner] 已为 %d 个角色规划 visual_states: %s",
            len(merged),
            list(merged.keys()),
        )
    merged = prune_static_visual_states(characters, merged)
    return merged, discussion


def apply_visual_states_to_characters(
    characters: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    llm: _ChatLLM | None = None,
    budget_mode: bool = False,
    config: dict | None = None,
    full_text: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """规划并写入角色条目的 visual_states 字段；返回 (角色列表, 审核讨论日志)。"""
    states_by_name, discussion = plan_visual_states(
        characters,
        segments,
        llm=llm,
        budget_mode=budget_mode,
        config=config,
        full_text=full_text,
    )
    return attach_visual_states(characters, states_by_name), discussion
