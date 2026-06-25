"""分段版本化角色外观 — 按 segment_index 选取有效 visual_state。"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger("novel")

# 剧情锚点：命中后视为「孕后/流产后」等状态切换（取最早命中段）
_POST_PREGNANCY_ANCHORS: tuple[str, ...] = (
    "人流手术",
    "做了人流",
    "流产手术",
    "堕胎",
    "人流",
    "手术前一天",
    "约了手术",
    "做完人流",
    "流产的三个月",
    "做完手术",
)

_MIXED_PHASE_RE = re.compile(
    r"前期[^；;。]*?(?=(?:后期|中后期|末尾|结尾))",
    re.DOTALL,
)


def is_effective_at_segment(state: dict[str, Any], segment_index: int) -> bool:
    """判断 visual_state 在 segment_index 是否生效（含起止，deprecated 为开区间）。"""
    if segment_index < 0:
        return False
    effective_from = state.get("effective_from_segment")
    deprecated_at = state.get("deprecated_at_segment")
    if effective_from is not None and segment_index < int(effective_from):
        return False
    if deprecated_at is not None and segment_index >= int(deprecated_at):
        return False
    return True


def pick_visual_state(
    visual_states: list[dict[str, Any]] | None,
    segment_index: int,
) -> dict[str, Any] | None:
    """返回 segment_index 处生效的 visual_state；多版本时取 effective_from 最大者。"""
    if not visual_states:
        return None
    matches = [
        st
        for st in visual_states
        if isinstance(st, dict) and is_effective_at_segment(st, segment_index)
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    def _sort_key(st: dict[str, Any]) -> tuple[int, int]:
        eff = st.get("effective_from_segment")
        ver = st.get("version", 0)
        return (int(eff) if eff is not None else -1, int(ver) if isinstance(ver, int) else 0)

    return max(matches, key=_sort_key)


def resolve_character_desc(
    entry: dict[str, Any] | None,
    segment_index: int,
    *,
    default: str = "",
) -> str:
    """按分段选取角色外观描述；无 visual_states 时回退 entry.desc。"""
    if not entry or not isinstance(entry, dict):
        return default
    states = entry.get("visual_states")
    if isinstance(states, list) and states:
        picked = pick_visual_state(states, segment_index)
        if picked:
            desc = str(picked.get("desc", "")).strip()
            if desc:
                return desc
    return str(entry.get("desc", "")).strip() or default


def normalize_visual_states(raw: list | None) -> list[dict[str, Any]]:
    """清洗 LLM/规则产出的 visual_states 列表。"""
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("desc", "")).strip()
        if not desc:
            continue
        state: dict[str, Any] = {
            "id": str(item.get("id", "")).strip() or f"state_{len(out)}",
            "desc": desc,
        }
        if item.get("effective_from_segment") is not None:
            state["effective_from_segment"] = int(item["effective_from_segment"])
        if item.get("deprecated_at_segment") is not None:
            state["deprecated_at_segment"] = int(item["deprecated_at_segment"])
        if item.get("trigger"):
            state["trigger"] = str(item["trigger"]).strip()
        out.append(state)
    return out


def attach_visual_states(
    characters: list[dict[str, Any]],
    states_by_name: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """将 visual_states 写入角色条目（深拷贝列表项）；未列入者清除旧 visual_states。"""
    merged: list[dict[str, Any]] = []
    for entry in characters:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        new_entry = dict(entry)
        if name and name in states_by_name:
            new_entry["visual_states"] = list(states_by_name[name])
        else:
            new_entry.pop("visual_states", None)
        merged.append(new_entry)
    return merged


def prune_static_visual_states(
    characters: list[dict[str, Any]],
    states_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """去掉无前后外观变化角色的 visual_states；单 state 不应带 deprecated。"""
    if not states_by_name:
        return states_by_name

    char_desc: dict[str, str] = {}
    for entry in characters:
        if isinstance(entry, dict):
            name = str(entry.get("name", "")).strip()
            if name:
                char_desc[name] = str(entry.get("desc", "")).strip()

    pruned: dict[str, list[dict[str, Any]]] = {}
    for name, states in states_by_name.items():
        if not states:
            continue
        desc = char_desc.get(name, "")
        has_phase_in_desc = "前期" in desc and "后期" in desc
        if len(states) == 1 and not has_phase_in_desc:
            log.info("[VisualState] 角色 %s 无外观阶段变化，跳过 visual_states", name)
            continue
        if len(states) == 1:
            only = dict(states[0])
            only.pop("deprecated_at_segment", None)
            pruned[name] = normalize_visual_states([only])
            continue
        pruned[name] = states
    return pruned


def find_anchor_segment(segments: list[dict[str, Any]], anchors: tuple[str, ...]) -> int | None:
    """在分段文本中查找最早命中锚点的 segment index。"""
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text", ""))
        idx = int(seg.get("index", -1))
        if idx < 0:
            continue
        if any(anchor in text for anchor in anchors):
            return idx
    return None


def segments_summary_for_prompt(
    segments: list[dict[str, Any]],
    *,
    max_chars: int = 120,
    max_lines: int | None = None,
) -> str:
    """生成分段索引表，供 visual_states 规划/审核 LLM 使用。"""
    lines: list[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        idx = int(seg.get("index", -1))
        text = str(seg.get("text", "")).replace("\n", " ")[:max_chars]
        if idx >= 0 and text:
            lines.append(f"[{idx}] {text}")
        if max_lines is not None and len(lines) >= max_lines:
            break
    return "\n".join(lines)


def format_segment_context(
    segments: list[dict[str, Any]],
    anchors: tuple[str, ...] | None = None,
    *,
    summary_max_lines: int = 80,
) -> str:
    """分段元信息 + 锚点 + 摘要（审核员须用此表的 index，非小说章节号）。"""
    if not segments:
        return "（无分段数据）"
    indices = [
        int(s.get("index", -1))
        for s in segments
        if isinstance(s, dict) and int(s.get("index", -1)) >= 0
    ]
    total = len(indices)
    idx_min = min(indices) if indices else 0
    idx_max = max(indices) if indices else 0
    header = (
        f"流水线共 {total} 个分段，segment index 范围 {idx_min}–{idx_max}（从 0 起算）。\n"
        "⚠️ effective_from_segment / deprecated_at_segment 必须使用下表 [数字] 索引，"
        "不是小说正文里的章节标记「1」「14」「16」等。"
    )
    anchor_lines: list[str] = []
    if anchors:
        for anchor in anchors:
            hit = find_anchor_segment(segments, (anchor,))
            if hit is not None:
                anchor_lines.append(f"- 「{anchor}」→ segment {hit}")
    anchor_block = ""
    if anchor_lines:
        anchor_block = "\n剧情锚点（规则检测）：\n" + "\n".join(anchor_lines)
    summary = segments_summary_for_prompt(segments, max_lines=summary_max_lines)
    if len(segments) > summary_max_lines:
        summary += f"\n…（仅展示前 {summary_max_lines} 段，全文共 {total} 段）"
    return f"{header}{anchor_block}\n\n分段索引表：\n{summary}"


def split_mixed_phase_desc(desc: str) -> tuple[str, str] | None:
    """从「前期…后期…」混写 desc 拆出两段互斥描述。"""
    if not desc or "前期" not in desc or "后期" not in desc:
        return None
    # 粗拆：前期块 vs 后期块
    early_m = re.search(r"前期[^；;。，,]*", desc)
    late_m = re.search(r"后期[^。]*", desc)
    if not early_m or not late_m:
        return None
    early_part = early_m.group(0).replace("前期", "").strip(" ，,；;")
    late_part = late_m.group(0).replace("后期", "").replace("中后期", "").strip(" ，,；;")
    if not early_part or not late_part:
        return None
    prefix = ""
    head = re.split(r"前期", desc, maxsplit=1)[0].strip(" ，,；;")
    if head and "前期" not in head and "后期" not in head:
        prefix = head + "，"
    early_desc = f"{prefix}{early_part}".strip("，")
    late_desc = f"{prefix}{late_part}".strip("，")
    return early_desc, late_desc


def plan_visual_states_by_rules(
    characters: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """规则兜底：混写 desc 拆分 + 剧情锚点定切换段。"""
    switch_at = find_anchor_segment(segments, _POST_PREGNANCY_ANCHORS)
    result: dict[str, list[dict[str, Any]]] = {}
    for entry in characters:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        desc = str(entry.get("desc", "")).strip()
        if not name or not desc:
            continue
        split = split_mixed_phase_desc(desc)
        if not split and not switch_at:
            continue
        if split:
            early_desc, late_desc = split
            pivot = switch_at if switch_at is not None else max(len(segments) // 2, 1)
            result[name] = [
                {
                    "id": "early",
                    "desc": early_desc,
                    "effective_from_segment": 0,
                    "deprecated_at_segment": pivot,
                    "trigger": "规则拆分前期",
                },
                {
                    "id": "late",
                    "desc": late_desc,
                    "effective_from_segment": pivot,
                    "trigger": "规则拆分后期",
                },
            ]
        elif switch_at is not None and ("孕" in desc or "孕肚" in desc or "微隆" in desc):
            # 有孕相描述 + 锚点：前期保留原 desc 中含孕部分，后期去掉孕相关键词
            late_desc = re.sub(
                r"[^，,。；;]*孕[^，,。；;]*[，,。；;]?", "", desc
            )
            late_desc = re.sub(r"微隆|孕肚|护肚|孕妇", "", late_desc)
            late_desc = re.sub(r"[，,]{2,}", "，", late_desc).strip(" ，,；;")
            if late_desc and late_desc != desc:
                result[name] = [
                    {
                        "id": "before_anchor",
                        "desc": desc,
                        "effective_from_segment": 0,
                        "deprecated_at_segment": switch_at,
                    },
                    {
                        "id": "after_anchor",
                        "desc": late_desc,
                        "effective_from_segment": switch_at,
                        "trigger": "剧情锚点",
                    },
                ]
    return result
