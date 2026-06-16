#!/usr/bin/env python3
"""重生成指定分镜图片（使用当前 prompt 配置）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重生成单个分镜图片")
    parser.add_argument("workspace", type=Path, help="workspace/项目名")
    parser.add_argument("index", type=int, help="分镜序号（0-based，如 55 对应 0055.png）")
    parser.add_argument("--config", "-c", type=Path, default=None)
    parser.add_argument("--style", default=None, help="覆盖风格预设，如 anime")
    args = parser.parse_args(argv)

    workspace = args.workspace
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace
    if not workspace.exists():
        print(f"workspace 不存在: {workspace}", file=sys.stderr)
        return 2

    from src.config_manager import load_config
    from src.agents.art_director import ArtDirectorAgent
    from src.tools.segment_tool import SegmentTool

    cfg = load_config(args.config)
    state_path = workspace / "agent_state.json"
    state: dict = {}
    full_text = None
    characters = None
    suggested_style = args.style
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        full_text = state.get("full_text")
        characters = state.get("characters")
        suggested_style = suggested_style or state.get("suggested_style")

    input_file = state.get("input_file")
    if not full_text and input_file:
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        if input_path.exists():
            full_text = input_path.read_text(encoding="utf-8")

    if not full_text:
        print("无法获取全文，请确保 agent_state.json 或 input 文件存在", file=sys.stderr)
        return 2

    segments = SegmentTool(cfg).run(full_text)
    idx = args.index
    if idx < 0 or idx >= len(segments):
        print(f"index 越界: {idx} (共 {len(segments)} 段)", file=sys.stderr)
        return 2

    agent = ArtDirectorAgent(cfg, budget_mode=True)
    if suggested_style:
        agent.prompt_gen.set_style(suggested_style)
    if characters:
        agent.prompt_gen.seed_characters(characters)

    out_path = workspace / "images" / f"{idx:04d}.png"
    if out_path.exists():
        out_path.unlink()

    prev_text = segments[idx - 1]["text"] if idx > 0 else None
    path, score, retries, decisions = agent.generate_image(
        segments[idx]["text"],
        idx,
        workspace,
        full_text=full_text,
        prev_text=prev_text,
    )
    print(f"完成: {path} (score={score}, moderation_retries={retries})")
    for d in decisions:
        if "moderation" in str(d.get("step", "")):
            print(f"  - {d.get('decision')}: {d.get('reason', '')[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
