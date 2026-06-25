"""visual_states 分段外观 — 双 AI 讨论审核（去重、消歧、互斥定稿）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.agents.character_reviewer import create_reviewer_llm, character_review_enabled
from src.agents.utils import extract_json_obj
from src.logger import log
from src.promptgen.visual_state import (
    _POST_PREGNANCY_ANCHORS,
    format_segment_context,
    normalize_visual_states,
)

_SAMPLE_MAX_LEN = 8000

_SEGMENT_CTX_BLOCK = """
【流水线分段索引 — 切换段必须引用此表 [数字]，勿用小说章节号】
{segment_context}
"""

_REVIEWER_AUDIT_PROMPT = """\
你是 continuity 审核专家（审核员 B）。分析员 A 用规则/初稿拆分了角色的分段外观 visual_states。
请严格审核各 state 的 desc 是否可直接用于 AI 生图。

审核重点：
1. **互斥**：前期/孕相 state 不得含「清瘦/憔悴/术后」等后期特征；流产后/后期 state 不得含「孕肚/微隆/护肚/孕妇装」等孕相
2. **去重**：不得大段重复同一句；共享信息（年龄、性别、发型）只写一次，阶段差异写清楚即可
3. **自洽**：单条 desc 内部不得前后矛盾（如同时写孕肚和清瘦）
4. **完整**：每条 80-150 字，含年龄、性别、身材、发型、服装、神态/道具；可独立于其他 state 理解
5. **切换段**：初稿中的 effective_from_segment / deprecated_at_segment 由规则引擎按下方分段表计算，**你只审核 desc，不要改 segment 数字**（除非发现与锚点表明显矛盾且说明依据）
6. **无阶段变化的角色**（desc 无「前期/后期」且初稿未列入）**不要新增** visual_states

原文摘要（非分段表）：
{text}
{segment_context_block}

角色原 desc（JSON）：
{characters_json}

初稿 visual_states（JSON）：
{draft_json}

请输出 JSON：
{{
  "overall_comment": "总体评价",
  "issues": [
    {{
      "name": "角色名",
      "state_id": "state 的 id",
      "type": "desc_fix|contradiction|duplicate",
      "detail": "问题说明",
      "suggestion": "具体修改建议"
    }}
  ]
}}"""

_PRIMARY_RESPONSE_PROMPT = """\
你是角色视觉设定专家（分析员 A）。审核员 B 对分段外观初稿提出意见，请修订 visual_states。

规则：
- **只改 desc 文案**；effective_from_segment / deprecated_at_segment / id **必须与初稿完全一致**，不得修改
- 各 state 的 desc 必须互斥、无重复堆砌、无内部矛盾
- 每条 80-150 字中文，可直接用于 SD/ComfyUI 角色一致性锁定
- segment index 见下方分段表，不是小说章节「1」「14」等

原文摘要：
{text}
{segment_context_block}

初稿：
{draft_json}

审核员 B 意见：
{review_json}

请输出 JSON：
{{
  "reply": "对审核意见的回应摘要",
  "visual_states": {{
    "角色名": [
      {{
        "id": "early",
        "desc": "该阶段完整外观"
      }}
    ]
  }}
}}"""

_REVIEWER_FINAL_PROMPT = """\
你是 continuity 审核专家（审核员 B）。请综合讨论给出**最终** visual_states。

原则：
- 以原文剧情为准；desc 互斥、简洁、无矛盾、无大段重复
- **禁止修改** effective_from_segment / deprecated_at_segment（已由规则引擎锁定）
- 每条 desc 须可独立用于 AI 生图

原文摘要：
{text}
{segment_context_block}

分析员 A 修订稿：
{revised_json}

分析员 A 回应：
{reply}

请输出 JSON：
{{
  "consensus_note": "定稿结论摘要",
  "visual_states": {{
    "角色名": [
      {{"id": "...", "desc": "..."}}
    ]
  }}
}}"""


@dataclass
class VisualStateReviewResult:
    visual_states: dict[str, list[dict[str, Any]]]
    discussion: list[str] = field(default_factory=list)
    reviewer_provider: str = ""
    same_source: bool = False


def visual_state_review_enabled(config: dict, budget_mode: bool) -> bool:
    """是否启用 visual_states 双 AI 审核（默认与 character_review 同开关）。"""
    if budget_mode:
        return False
    vs_cfg = (config.get("agent") or {}).get("visual_state_review") or {}
    if vs_cfg.get("enabled") is False:
        return False
    if vs_cfg.get("enabled") is True:
        return True
    return character_review_enabled(config, budget_mode)


def _sample_text(text: str, max_len: int = _SAMPLE_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    head = max_len // 2
    return text[:head] + "\n…\n" + text[-(max_len - head) :]


def _chat_json(llm, prompt: str) -> dict | list | None:
    result = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        json_mode=True,
    )
    content = getattr(result, "content", "") or ""
    return extract_json_obj(content)


def _parse_visual_states_payload(
    data: dict | None,
) -> dict[str, list[dict[str, Any]]]:
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


def _preserve_segment_bounds(
    reviewed: dict[str, list[dict[str, Any]]],
    draft: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """LLM 只改 desc；分段边界始终沿用规则初稿。"""
    merged: dict[str, list[dict[str, Any]]] = {}
    for name, draft_states in draft.items():
        rev_states = reviewed.get(name)
        if not rev_states:
            merged[name] = draft_states
            continue
        draft_by_id = {str(s.get("id", "")): s for s in draft_states}
        out_states: list[dict[str, Any]] = []
        for i, rev in enumerate(rev_states):
            rid = str(rev.get("id", "")).strip() or f"state_{i}"
            base = draft_by_id.get(rid) or (draft_states[i] if i < len(draft_states) else {})
            state = dict(rev)
            state["id"] = rid
            for key in ("effective_from_segment", "deprecated_at_segment", "trigger"):
                if key in base:
                    state[key] = base[key]
            out_states.append(state)
        merged[name] = normalize_visual_states(out_states) or draft_states
    return merged


def _segment_context_block(segments: list[dict[str, Any]]) -> str:
    ctx = format_segment_context(segments, _POST_PREGNANCY_ANCHORS)
    return _SEGMENT_CTX_BLOCK.format(segment_context=ctx)


def run_visual_state_review_discussion(
    text: str,
    characters: list[dict[str, Any]],
    draft_states: dict[str, list[dict[str, Any]]],
    segments: list[dict[str, Any]],
    *,
    primary_llm,
    reviewer_llm,
    reviewer_provider: str = "",
    same_source: bool = False,
) -> VisualStateReviewResult:
    """双 AI 讨论定稿 visual_states desc（分段边界锁定为规则初稿）。"""
    discussion: list[str] = []
    if not draft_states:
        return VisualStateReviewResult(
            visual_states={},
            discussion=discussion,
            reviewer_provider=reviewer_provider,
            same_source=same_source,
        )

    sample = _sample_text(text)
    seg_block = _segment_context_block(segments)
    chars_json = json.dumps(
        [
            {"name": c.get("name"), "desc": c.get("desc")}
            for c in characters
            if isinstance(c, dict) and c.get("name")
        ],
        ensure_ascii=False,
        indent=2,
    )
    draft_json = json.dumps(draft_states, ensure_ascii=False, indent=2)

    review_data: dict[str, Any] = {}
    try:
        raw_review = _chat_json(
            reviewer_llm,
            _REVIEWER_AUDIT_PROMPT.format(
                text=sample,
                segment_context_block=seg_block,
                characters_json=chars_json,
                draft_json=draft_json,
            ),
        )
        if isinstance(raw_review, dict):
            review_data = raw_review
    except Exception as exc:
        log.warning("[VisualStateReview] 审核员 B 初审失败 (%s)，保留初稿", exc)
        return VisualStateReviewResult(
            visual_states=draft_states,
            discussion=discussion,
            reviewer_provider=reviewer_provider,
            same_source=same_source,
        )

    overall = str(review_data.get("overall_comment", "")).strip()
    issues = review_data.get("issues") or []
    if overall:
        discussion.append(f"审核员B: {overall}")
        log.info("[VisualStateReview] 审核员 B: %s", overall[:200])
    if issues:
        issue_lines = [
            f"- [{i.get('type', '?')}] {i.get('name', '?')}/{i.get('state_id', '?')}: "
            f"{i.get('detail', '')}"
            for i in issues
            if isinstance(i, dict)
        ]
        if issue_lines:
            discussion.append("审核问题:\n" + "\n".join(issue_lines[:16]))

    revised_states = dict(draft_states)
    reply = ""
    try:
        raw_response = _chat_json(
            primary_llm,
            _PRIMARY_RESPONSE_PROMPT.format(
                text=sample,
                segment_context_block=seg_block,
                draft_json=draft_json,
                review_json=json.dumps(review_data, ensure_ascii=False, indent=2),
            ),
        )
        if isinstance(raw_response, dict):
            reply = str(raw_response.get("reply", "")).strip()
            parsed = _parse_visual_states_payload(raw_response)
            if parsed:
                revised_states = _preserve_segment_bounds(parsed, draft_states)
    except Exception as exc:
        log.warning("[VisualStateReview] 分析员 A 回应失败 (%s)，保留初稿", exc)
        return VisualStateReviewResult(
            visual_states=draft_states,
            discussion=discussion,
            reviewer_provider=reviewer_provider,
            same_source=same_source,
        )

    if reply:
        discussion.append(f"分析员A: {reply}")
        log.info("[VisualStateReview] 分析员 A: %s", reply[:200])

    final_states = revised_states
    try:
        raw_final = _chat_json(
            reviewer_llm,
            _REVIEWER_FINAL_PROMPT.format(
                text=sample,
                segment_context_block=seg_block,
                revised_json=json.dumps(revised_states, ensure_ascii=False, indent=2),
                reply=reply or "（无文字回应）",
            ),
        )
        if isinstance(raw_final, dict):
            note = str(raw_final.get("consensus_note", "")).strip()
            parsed = _parse_visual_states_payload(raw_final)
            if parsed:
                final_states = _preserve_segment_bounds(parsed, draft_states)
            if note:
                discussion.append(f"共识: {note}")
                log.info("[VisualStateReview] 共识: %s", note[:200])
    except Exception as exc:
        log.warning("[VisualStateReview] 终审失败 (%s)，采用分析员 A 修订稿", exc)

    if same_source:
        log.warning("[VisualStateReview] 审核 LLM 与规划 LLM 同源，结果仅供参考")

    return VisualStateReviewResult(
        visual_states=final_states,
        discussion=discussion,
        reviewer_provider=reviewer_provider,
        same_source=same_source,
    )
