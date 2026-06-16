"""ContentAnalyzer 角色列表 — 双 AI 讨论审核。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.agents.utils import extract_json_array, extract_json_obj
from src.logger import log

MAX_CHARACTERS = 8
_SAMPLE_MAX_LEN = 8000

_REVIEWER_AUDIT_PROMPT = """\
你是资深影视选角与 continuity 审核专家（审核员 B）。
另一位分析员 A 已从小说文本中提取了角色视觉设定列表，请你严格审核。

审核重点：
1. 是否遗漏有戏份的具名角色（尤其后段出场者）
2. 是否误把场景词、碎片词、泛称（我、前桌、老师）当人名
3. 同一人物是否重复出现（我/主角 与后文真名）
4. desc 是否与原文矛盾、是否缺少年龄/性别/发型等关键字段
5. 是否超过 {max_chars} 个角色，低优先级角色应建议删除

文本：
{text}

分析员 A 的初稿（JSON）：
{draft_json}

请输出 JSON：
{{
  "overall_comment": "总体评价",
  "issues": [
    {{
      "type": "remove|add|merge|desc_fix|rename",
      "name": "相关姓名",
      "detail": "问题说明",
      "suggestion": "具体修改建议（merge 时 suggestion 写目标真名）"
    }}
  ]
}}"""

_PRIMARY_RESPONSE_PROMPT = """\
你是小说角色视觉设定专家（分析员 A）。
审核员 B 对你提交的角色初稿提出了意见，请结合原文回应并给出修订版角色列表。

规则：
- name 必须用文中真名，禁止泛称占槽
- 接受合理审核意见，若原文支持你的初稿可简要反驳
- 每个 desc 为 80-150 字中文外观描述，含年龄、性别、身材、样貌、发型发色、服装、标志细节
- 最多 {max_chars} 个角色

文本：
{text}

你的初稿：
{draft_json}

审核员 B 的意见：
{review_json}

请输出 JSON：
{{
  "reply": "对审核意见的回应摘要",
  "characters": [{{"name": "姓名", "desc": "完整外观描述"}}]
}}"""

_REVIEWER_FINAL_PROMPT = """\
你是 continuity 审核专家（审核员 B）。
你与分析员 A 已完成一轮讨论，请综合双方观点给出最终角色列表。

原则：
- 以原文为准；双方都无依据时删除该角色
- 合并重复角色，保留信息更完整的 desc
- 最多 {max_chars} 个角色；desc 须可直接用于 AI 生图一致性锁定

文本：
{text}

分析员 A 修订稿：
{revised_json}

分析员 A 的回应摘要：
{reply}

请输出 JSON：
{{
  "consensus_note": "讨论结论摘要",
  "characters": [{{"name": "姓名", "desc": "完整外观描述"}}]
}}"""


@dataclass
class CharacterReviewResult:
    characters: list[dict]
    discussion: list[str] = field(default_factory=list)
    reviewer_provider: str = ""
    same_source: bool = False


def character_review_enabled(config: dict, budget_mode: bool) -> bool:
    """是否启用双 AI 角色讨论（budget 模式默认关闭）。"""
    if budget_mode:
        return False
    review_cfg = (config.get("agent") or {}).get("character_review") or {}
    return review_cfg.get("enabled", True) is not False


def _writer_provider(config: dict) -> str:
    provider = str((config.get("llm") or {}).get("provider") or "auto").strip().lower()
    if provider == "auto":
        from src.llm.llm_client import _detect_provider

        provider, _ = _detect_provider()
    return provider


def create_reviewer_llm(config: dict):
    """创建审核用 LLM（默认异源 reviewer）。"""
    from src.llm.llm_client import create_llm_client
    from src.novel.quality.judge import auto_select_judge

    review_cfg = (config.get("agent") or {}).get("character_review") or {}
    override_provider = review_cfg.get("reviewer_provider")
    override_model = review_cfg.get("reviewer_model")

    if override_provider:
        llm_cfg = {
            "provider": override_provider,
            "model": override_model,
            "temperature": review_cfg.get("temperature", 0.2),
        }
        return create_llm_client(llm_cfg), str(override_provider), False

    judge_cfg = auto_select_judge(_writer_provider(config))
    llm_cfg = {
        "provider": judge_cfg.provider,
        "model": override_model or judge_cfg.model,
        "temperature": review_cfg.get("temperature", 0.2),
    }
    return create_llm_client(llm_cfg), judge_cfg.provider, judge_cfg.same_source


def _sample_text(text: str, max_len: int = _SAMPLE_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    head = max_len // 2
    return text[:head] + "\n…\n" + text[-(max_len - head) :]


def _normalize_characters(data: list | None, *, max_chars: int = MAX_CHARACTERS) -> list[dict]:
    if not data:
        return []
    out: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        desc = str(entry.get("desc", "")).strip()
        if not name:
            continue
        out.append({"name": name, "desc": desc})
    return out[:max_chars]


def _chat_json(llm, prompt: str) -> dict | list | None:
    result = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        json_mode=True,
    )
    content = getattr(result, "content", "") or ""
    obj = extract_json_obj(content)
    if obj:
        return obj
    arr = extract_json_array(content)
    if arr:
        return arr
    return None


def run_character_review_discussion(
    text: str,
    draft_characters: list[dict],
    *,
    primary_llm,
    reviewer_llm,
    reviewer_provider: str = "",
    same_source: bool = False,
    era_addendum: str = "",
    max_chars: int = MAX_CHARACTERS,
) -> CharacterReviewResult:
    """双 AI 三轮讨论，产出最终角色列表。"""
    discussion: list[str] = []
    sample = _sample_text(text)
    draft_json = json.dumps(draft_characters, ensure_ascii=False, indent=2)

    audit_prompt = _REVIEWER_AUDIT_PROMPT.format(
        max_chars=max_chars,
        text=sample,
        draft_json=draft_json,
    )
    if era_addendum:
        audit_prompt += f"\n\n{era_addendum}"

    review_data: dict[str, Any] = {}
    try:
        raw_review = _chat_json(reviewer_llm, audit_prompt)
        if isinstance(raw_review, dict):
            review_data = raw_review
    except Exception as exc:
        log.warning("[CharacterReview] 审核员 B 初审失败 (%s)，保留初稿", exc)
        return CharacterReviewResult(
            characters=draft_characters,
            discussion=discussion,
            reviewer_provider=reviewer_provider,
            same_source=same_source,
        )

    overall = str(review_data.get("overall_comment", "")).strip()
    issues = review_data.get("issues") or []
    if overall:
        discussion.append(f"审核员B: {overall}")
        log.info("[CharacterReview] 审核员 B: %s", overall[:200])
    if issues:
        issue_lines = [
            f"- [{i.get('type', '?')}] {i.get('name', '?')}: {i.get('detail', '')}"
            for i in issues
            if isinstance(i, dict)
        ]
        if issue_lines:
            discussion.append("审核问题:\n" + "\n".join(issue_lines[:12]))

    response_prompt = _PRIMARY_RESPONSE_PROMPT.format(
        max_chars=max_chars,
        text=sample,
        draft_json=draft_json,
        review_json=json.dumps(review_data, ensure_ascii=False, indent=2),
    )
    if era_addendum:
        response_prompt += f"\n\n{era_addendum}"

    revised_characters = list(draft_characters)
    reply = ""
    try:
        raw_response = _chat_json(primary_llm, response_prompt)
        if isinstance(raw_response, dict):
            reply = str(raw_response.get("reply", "")).strip()
            parsed = _normalize_characters(
                raw_response.get("characters"), max_chars=max_chars
            )
            if parsed:
                revised_characters = parsed
    except Exception as exc:
        log.warning("[CharacterReview] 分析员 A 回应失败 (%s)，保留初稿", exc)
        return CharacterReviewResult(
            characters=draft_characters,
            discussion=discussion,
            reviewer_provider=reviewer_provider,
            same_source=same_source,
        )

    if reply:
        discussion.append(f"分析员A: {reply}")
        log.info("[CharacterReview] 分析员 A: %s", reply[:200])

    final_prompt = _REVIEWER_FINAL_PROMPT.format(
        max_chars=max_chars,
        text=sample,
        revised_json=json.dumps(revised_characters, ensure_ascii=False, indent=2),
        reply=reply or "（无文字回应）",
    )
    if era_addendum:
        final_prompt += f"\n\n{era_addendum}"

    final_characters = revised_characters
    try:
        raw_final = _chat_json(reviewer_llm, final_prompt)
        if isinstance(raw_final, dict):
            note = str(raw_final.get("consensus_note", "")).strip()
            parsed = _normalize_characters(
                raw_final.get("characters"), max_chars=max_chars
            )
            if parsed:
                final_characters = parsed
            if note:
                discussion.append(f"共识: {note}")
                log.info("[CharacterReview] 共识: %s", note[:200])
    except Exception as exc:
        log.warning("[CharacterReview] 终审失败 (%s)，采用分析员 A 修订稿", exc)

    if same_source:
        log.warning("[CharacterReview] 审核 LLM 与提取 LLM 同源，讨论结果仅供参考")

    return CharacterReviewResult(
        characters=final_characters,
        discussion=discussion,
        reviewer_provider=reviewer_provider,
        same_source=same_source,
    )
