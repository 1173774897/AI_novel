#!/usr/bin/env python3
"""用分镜原文修复已有 SRT 文本（补回「」等标点），保留原时间轴。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tts.subtitle_generator import SubtitleGenerator
from src.video.video_assembler import VideoAssembler


def _redistribute_source(source: str, entries: list[dict]) -> list[dict]:
    """按原条目字数比例，把完整原文重新分配到各时间片。"""
    if not entries:
        return []
    if len(entries) == 1:
        return [{**entries[0], "text": source}]

    weights = [max(1, len(e.get("text", ""))) for e in entries]
    total_weight = sum(weights)
    chunks: list[str] = []
    pos = 0
    for idx, weight in enumerate(weights):
        if idx == len(weights) - 1:
            chunks.append(source[pos:])
            break
        target = pos + max(1, int(round(weight / total_weight * len(source))))
        cut = SubtitleGenerator._find_break_index(source[pos:], target - pos)
        if cut <= 0:
            cut = min(len(source) - pos, target - pos)
        chunks.append(source[pos : pos + cut])
        pos += cut

    result: list[dict] = []
    for entry, chunk in zip(entries, chunks):
        text = chunk.strip() or entry.get("text", "")
        result.append({**entry, "text": text})
    return result


def fix_workspace(workspace: Path, dry_run: bool = False) -> int:
    state_path = workspace / "agent_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"未找到 {state_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    segments = state.get("segments") or []
    sub_dir = workspace / "subtitles"
    fixed = 0

    for idx, seg in enumerate(segments):
        source = SubtitleGenerator._sanitize_entry_text(seg.get("text") or "")
        srt_path = sub_dir / f"{idx:04d}.srt"
        if not srt_path.exists() or not source:
            continue

        entries = VideoAssembler._parse_srt_entries(srt_path)
        if not entries:
            continue

        old_joined = "".join(e["text"] for e in entries)
        new_entries = _redistribute_source(source, entries)
        new_joined = "".join(e["text"] for e in new_entries)
        if new_joined == old_joined:
            continue

        if not dry_run:
            srt_path.write_text(
                SubtitleGenerator._render_srt(new_entries),
                encoding="utf-8",
            )
        fixed += 1

    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="项目 workspace 目录")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args()

    count = fix_workspace(args.workspace.resolve(), dry_run=args.dry_run)
    action = "将修复" if args.dry_run else "已修复"
    print(f"{action} {count} 个字幕文件")


if __name__ == "__main__":
    main()
