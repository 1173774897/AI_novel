#!/usr/bin/env python3
"""生成 CRT 屏幕蒙版并在 tv-frame 上输出校准预览。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config_manager import load_config
from src.video.intro_tv_frame import calibrate_intro_tv_screen


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 tv-screen 蒙版并输出校准预览图")
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--mask-output",
        default="media/tv-screen-mask.png",
        help="蒙版 PNG 输出路径（屏幕区域尺寸）",
    )
    parser.add_argument(
        "--debug-output",
        default="media/tv-screen-calibration.png",
        help="校准预览图输出路径",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    mask_path, debug_path = calibrate_intro_tv_screen(
        config,
        mask_output=Path(args.mask_output),
        debug_output=Path(args.debug_output),
    )
    print(f"mask: {mask_path}")
    print(f"debug: {debug_path}")


if __name__ == "__main__":
    main()
