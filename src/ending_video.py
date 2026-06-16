"""片尾视频合成 - 固定短片 + TTS + 电视机框 + 关电视片段。"""

from __future__ import annotations

import shutil
from pathlib import Path


def compose_ending(
    workspace: Path,
    config: dict,
    *,
    output_path: Path | None = None,
    rate: str | None = None,
    volume: str | None = None,
    tv_frame: bool | None = None,
) -> Path:
    """合成片尾：感谢短片 + TTS 对齐 + 电视机框 + tv-shotdown。"""
    from src.tools.tts_tool import TTSTool
    from src.video.intro_content import (
        append_video_clip,
        build_focus_tagline_clip,
        prepend_black_lead,
        resolve_ending_config,
    )
    from src.video.intro_tv_frame import (
        composite_content_in_tv_frame,
        resolve_intro_frame_config,
    )

    workspace = Path(workspace)
    intro_cfg = config.get("intro") or {}
    ending_cfg = config.get("ending") or {}
    video_cfg = config.get("video") or {}

    intro_dir = workspace / "intro"
    intro_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = intro_dir / "ending.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip_path, tagline, shutdown_path = resolve_ending_config(ending_cfg)
    codec = str(ending_cfg.get("codec", intro_cfg.get("codec", video_cfg.get("codec", "libx264"))))
    crf = int(ending_cfg.get("crf", intro_cfg.get("crf", video_cfg.get("crf", 18))))
    fps = int(ending_cfg.get("fps", intro_cfg.get("fps", video_cfg.get("fps", 30))))
    width, height = (int(v) for v in video_cfg.get("resolution", [1920, 1080]))

    tagline_audio = intro_dir / "ending_tagline.mp3"
    tagline_srt = intro_dir / "ending_tagline.srt"
    TTSTool(config).run(
        tagline,
        tagline_audio,
        tagline_srt,
        rate=rate,
        volume=volume,
    )

    content_path = intro_dir / "ending_content.mp4"
    build_focus_tagline_clip(
        clip_path,
        tagline_audio,
        content_path,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        crf=crf,
    )

    frame_cfg = None
    if tv_frame is not False:
        frame_cfg = resolve_intro_frame_config(config)

    if frame_cfg is not None:
        tv_part = intro_dir / "ending_tv_part.mp4"
        composite_content_in_tv_frame(content_path, tv_part, frame_cfg)
        main_part = tv_part
    else:
        main_part = content_path

    lead_duration = float(ending_cfg.get("lead_black_duration", 1.0))
    if lead_duration > 0:
        lead_path = intro_dir / "ending_with_lead.mp4"
        prepend_black_lead(
            main_part,
            lead_path,
            duration=lead_duration,
            codec=codec,
            crf=crf,
            fps=fps,
        )
        main_part = lead_path

    if shutdown_path is not None:
        shutdown_volume = float(ending_cfg.get("shutdown_audio_volume", 0.5))
        append_video_clip(
            main_part,
            shutdown_path,
            output_path,
            codec=codec,
            crf=crf,
            fps=fps,
            second_audio_volume=shutdown_volume,
        )
        return output_path

    if main_part.resolve() != output_path.resolve():
        shutil.copy2(main_part, output_path)
    return output_path
