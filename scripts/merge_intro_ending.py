#!/usr/bin/env python3
"""将片头 intro 与片尾 ending 拼接到 output 正片的开头和结尾。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="拼接片头 + 正片 + 片尾为完整视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s workspace/无尽恶意
  %(prog)s workspace/无尽恶意 -o output/无尽恶意_完整版.mp4
  %(prog)s workspace/无尽恶意 --replace
  %(prog)s workspace/无尽恶意 --no-intro
        """,
    )
    parser.add_argument("workspace", type=Path, help="workspace/项目名")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="输出路径（默认 output/<项目名>_full.mp4）",
    )
    parser.add_argument("--main", type=Path, default=None, help="正片 MP4（默认 output/<stem>.mp4）")
    parser.add_argument("--intro", type=Path, default=None, help="片头 MP4（默认 workspace/intro/intro.mp4）")
    parser.add_argument("--ending", type=Path, default=None, help="片尾 MP4（默认 workspace/intro/ending.mp4）")
    parser.add_argument("--no-intro", action="store_true", help="不拼接片头")
    parser.add_argument("--no-ending", action="store_true", help="不拼接片尾")
    parser.add_argument(
        "--replace", action="store_true",
        help="覆盖默认正片路径（先备份为 *.mp4.bak）",
    )
    parser.add_argument("--config", "-c", type=Path, default=None, help="配置文件路径")
    args = parser.parse_args(argv)

    workspace = args.workspace
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace
    if not workspace.is_dir():
        print(f"workspace 不存在: {workspace}", file=sys.stderr)
        return 2

    from src.config_manager import load_config
    from src.video.full_video_merge import (
        merge_intro_main_ending,
        resolve_default_paths,
        resolve_project_stem,
    )

    cfg = load_config(args.config)
    default_intro, default_main, default_ending = resolve_default_paths(
        workspace, cfg, project_root=PROJECT_ROOT,
    )

    main_path = args.main or default_main
    if not main_path.is_absolute():
        main_path = PROJECT_ROOT / main_path

    intro_path = None if args.no_intro else (args.intro or default_intro)
    ending_path = None if args.no_ending else (args.ending or default_ending)

    if intro_path and not intro_path.is_absolute():
        intro_path = PROJECT_ROOT / intro_path
    if ending_path and not ending_path.is_absolute():
        ending_path = PROJECT_ROOT / ending_path

    if not main_path.exists():
        print(f"正片不存在: {main_path}", file=sys.stderr)
        return 2
    if intro_path and not intro_path.exists():
        print(f"片头不存在: {intro_path}", file=sys.stderr)
        print("提示: 先运行 python main.py intro <workspace> ...", file=sys.stderr)
        return 2
    if ending_path and not ending_path.exists():
        print(f"片尾不存在: {ending_path}", file=sys.stderr)
        print("提示: 先运行 python main.py ending <workspace>", file=sys.stderr)
        return 2

    clips: list[Path] = []
    if intro_path:
        clips.append(intro_path)
    clips.append(main_path)
    if ending_path:
        clips.append(ending_path)

    if args.replace:
        output_path = main_path
        backup = main_path.with_suffix(main_path.suffix + ".bak")
        if backup.exists():
            backup.unlink()
        shutil.copy2(main_path, backup)
        print(f"已备份正片: {backup}")
    elif args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        stem = resolve_project_stem(workspace)
        out_dir = Path(cfg.get("project", {}).get("default_output", "output"))
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        output_path = out_dir / f"{stem}_full.mp4"

    if args.replace and len(clips) == 1:
        print("无片头/片尾可拼接", file=sys.stderr)
        return 2

    tmp_dir = workspace / "intro" / "merge_tmp"
    try:
        result = merge_intro_main_ending(
            clips,
            output_path,
            cfg,
            tmp_dir=tmp_dir,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    parts = []
    if intro_path:
        parts.append(f"片头({intro_path.name})")
    parts.append(f"正片({main_path.name})")
    if ending_path:
        parts.append(f"片尾({ending_path.name})")
    print(f"拼接完成: {' + '.join(parts)}")
    print(f"输出: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
