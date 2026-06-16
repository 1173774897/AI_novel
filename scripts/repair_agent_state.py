#!/usr/bin/env python3
"""修复 workspace 下损坏的 agent_state.json。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.agent_state_repair import repair_agent_state_data
from src.config_manager import load_config
from src.logger import log


def main() -> None:
    parser = argparse.ArgumentParser(description="修复 Agent 断点状态")
    parser.add_argument("workspace", type=Path, help="工作目录，如 workspace/01-极致捧杀")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="重新运行 ContentAnalyzer（需 LLM，补全 characters/genre 等）",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    state_file = workspace / "agent_state.json"
    if not state_file.exists():
        raise SystemExit(f"未找到 {state_file}")

    cfg = load_config(args.config)
    data = json.loads(state_file.read_text(encoding="utf-8"))
    backup = state_file.with_suffix(
        f".bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    )
    backup.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("已备份原状态 → %s", backup.name)

    repaired = repair_agent_state_data(data, cfg, workspace)

    if args.reanalyze and repaired.get("full_text"):
        from src.agents.content_analyzer import content_analyzer_node

        log.info("重新运行 ContentAnalyzer …")
        partial = content_analyzer_node(
            {
                **repaired,
                "config": cfg,
                "budget_mode": repaired.get("budget_mode", False),
            }
        )
        for key in (
            "segments",
            "genre",
            "era",
            "characters",
            "suggested_style",
            "intro_variants",
            "pov_narrator",
            "decisions",
        ):
            if key in partial and partial[key] is not None:
                if key == "decisions":
                    repaired["decisions"] = (repaired.get("decisions") or []) + partial[
                        "decisions"
                    ]
                else:
                    repaired[key] = partial[key]
        # 保留 art_director 完成标记（图片已在磁盘）
        completed = set(repaired.get("completed_nodes") or [])
        completed.update(["director", "content_analyzer", "art_director"])
        repaired["completed_nodes"] = list(completed)

    save_data = {k: v for k, v in repaired.items() if k != "config"}
    save_data["timestamp"] = datetime.now(timezone.utc).isoformat()
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(save_data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(state_file)

    log.info(
        "修复完成: segments=%d, images=%d, audio=%d, completed=%s",
        len(repaired.get("segments") or []),
        len(repaired.get("images") or []),
        len(repaired.get("audio_files") or []),
        repaired.get("completed_nodes"),
    )
    print(state_file)


if __name__ == "__main__":
    main()
