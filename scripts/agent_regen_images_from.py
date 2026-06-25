#!/usr/bin/env python3
"""Agent 工作区工具：注入 visual_states + 从指定段起删除图片以便断点续跑重生图。

不触碰 ContentAnalyzer / segments，孕妇期已生成图片可保留。

用法:
  # 仅注入 visual_states（不切图）
  python scripts/agent_regen_images_from.py workspace/06-身陷泥淖 --patch-only

  # 注入 visual_states，删除段 124 及以后图片，准备 --resume 重生图
  python scripts/agent_regen_images_from.py workspace/06-身陷泥淖 --from-segment 124

  # 指定切换段（覆盖规则锚点）
  python scripts/agent_regen_images_from.py workspace/06-身陷泥淖 --from-segment 124 --pivot 124
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_state_repair import dedupe_completed_nodes
from src.config_manager import load_config
from src.llm.llm_client import create_llm_client
from src.promptgen.visual_state import attach_visual_states, plan_visual_states_by_rules
from src.promptgen.visual_state_planner import plan_visual_states


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _delete_images_from(img_dir: Path, from_segment: int) -> int:
    removed = 0
    for path in img_dir.glob("*.png"):
        stem = path.stem.split("_")[0]
        if not stem.isdigit():
            continue
        if int(stem) >= from_segment:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def patch_visual_states(
    data: dict,
    *,
    pivot: int | None = None,
    budget_mode: bool = False,
    config: dict | None = None,
) -> list[str]:
    characters = data.get("characters") or []
    segments = data.get("segments") or []
    if not characters or not segments:
        return []

    cfg = config or load_config()
    llm = None
    if not budget_mode:
        try:
            llm = create_llm_client(cfg.get("llm", {}))
        except Exception:
            llm = None

    states_by_name, _discussion = plan_visual_states(
        characters,
        segments,
        llm=llm,
        budget_mode=budget_mode,
        config=cfg,
        full_text=data.get("full_text") or "",
    )
    if pivot is not None:
        for name, states in list(states_by_name.items()):
            if len(states) < 2:
                continue
            states[0]["effective_from_segment"] = 0
            states[0]["deprecated_at_segment"] = pivot
            states[1]["effective_from_segment"] = pivot
            if "deprecated_at_segment" in states[1]:
                del states[1]["deprecated_at_segment"]
            states_by_name[name] = states

    if not states_by_name:
        states_by_name = plan_visual_states_by_rules(characters, segments)
        if pivot is not None:
            for name, states in list(states_by_name.items()):
                if len(states) >= 2:
                    states[0]["deprecated_at_segment"] = pivot
                    states[1]["effective_from_segment"] = pivot
                    states_by_name[name] = states

    data["characters"] = attach_visual_states(characters, states_by_name)
    return list(states_by_name.keys())


def prepare_regen(data: dict, from_segment: int) -> None:
    completed = set(data.get("completed_nodes") or [])
    completed.discard("art_director")
    completed.discard("editor")
    data["completed_nodes"] = dedupe_completed_nodes(list(completed))

    ws = Path(data["workspace"])
    img_dir = ws / "images"
    images = []
    for i in range(len(data.get("segments") or [])):
        primary = img_dir / f"{i:04d}.png"
        if i < from_segment and primary.exists() and primary.stat().st_size > 100:
            images.append(str(primary))
        else:
            images.append("")
    data["images"] = images
    scores = data.get("quality_scores") or []
    if len(scores) < len(images):
        scores = scores + [-1.0] * (len(images) - len(scores))
    data["quality_scores"] = scores[: len(images)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 分段外观补丁 + 选择性重生图")
    parser.add_argument("workspace", type=Path, help="工作区目录（含 agent_state.json）")
    parser.add_argument(
        "--from-segment",
        type=int,
        default=None,
        help="从此 segment index 起删除图片并标记重跑 art_director",
    )
    parser.add_argument(
        "--pivot",
        type=int,
        default=None,
        help="visual_states 切换段（默认用规则锚点，如「约了手术」）",
    )
    parser.add_argument(
        "--budget",
        action="store_true",
        help="省钱模式：跳过 LLM 审核，仅用规则拆分",
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="只写入 visual_states，不删图、不改 completed_nodes",
    )
    args = parser.parse_args()

    ws = args.workspace.resolve()
    state_path = ws / "agent_state.json"
    if not state_path.is_file():
        print(f"未找到 {state_path}", file=sys.stderr)
        return 1

    data = _load_state(state_path)
    if not data.get("workspace"):
        data["workspace"] = str(ws)

    cfg = load_config()
    planned = patch_visual_states(
        data,
        pivot=args.pivot,
        budget_mode=args.budget,
        config=cfg,
    )
    if planned:
        print(f"已写入 visual_states: {planned}")
        if args.pivot is not None:
            print(f"  切换段 pivot={args.pivot}")
    else:
        print("未规划出 visual_states（角色 desc 无前后差异或缺少锚点）")

    if args.patch_only:
        _save_state(state_path, data)
        print(f"已保存 {state_path}")
        return 0

    if args.from_segment is None:
        print("未指定 --from-segment；若只补丁请用 --patch-only", file=sys.stderr)
        _save_state(state_path, data)
        return 0

    img_dir = ws / "images"
    removed = _delete_images_from(img_dir, args.from_segment)
    print(f"已删除段 {args.from_segment}+ 图片 {removed} 张")

    prepare_regen(data, args.from_segment)
    _save_state(state_path, data)
    print(
        f"已更新 agent_state：保留段 0–{args.from_segment - 1} 图片，"
        f"移除 art_director 完成标记"
    )
    print(f"下一步: python main.py run <input.txt> --mode agent --resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
