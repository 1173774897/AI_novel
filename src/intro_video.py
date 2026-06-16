"""片头视频合成 - 文案 TTS + workspace 配图 + VideoAssembler。"""

from __future__ import annotations

import shutil
from pathlib import Path


def split_intro_text(text: str) -> list[str]:
    """按中文句末标点切分片头文案。"""
    text = text.strip()
    if not text:
        raise ValueError("片头文案不能为空")

    segments: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in "。！？!?…":
            seg = "".join(buf).strip()
            if seg:
                segments.append(seg)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return segments or [text]


def collect_intro_images(
    workspace: Path,
    count: int,
    *,
    indices: list[int] | None = None,
) -> list[Path]:
    """为片头各句选取配图（默认 workspace/images 前 count 张）。"""
    if count < 1:
        raise ValueError("count 必须 >= 1")

    img_dir = workspace / "images"
    if indices is not None:
        if len(indices) != count:
            raise ValueError(
                f"image_indices 数量 ({len(indices)}) 与分句数 ({count}) 不匹配"
            )
        chosen = indices
    else:
        chosen = list(range(count))

    images: list[Path] = []
    for idx in chosen:
        img = img_dir / f"{idx:04d}.png"
        if not img.exists() or img.stat().st_size < 100:
            raise FileNotFoundError(f"片头配图不存在或为空: {img.name}")
        images.append(img)
    return images


def generate_intro_assets(
    segments: list[str],
    intro_dir: Path,
    config: dict,
    *,
    rate: str | None = None,
    volume: str | None = None,
) -> list[dict]:
    """为每句片头文案生成配音与字幕。"""
    from src.tools.tts_tool import TTSTool

    intro_dir = Path(intro_dir)
    intro_dir.mkdir(parents=True, exist_ok=True)

    tool = TTSTool(config)
    audio_srt: list[dict] = []
    for i, text in enumerate(segments):
        stem = f"{i:04d}"
        audio_path = intro_dir / f"{stem}.mp3"
        srt_path = intro_dir / f"{stem}.srt"
        tool.run(text, audio_path, srt_path, rate=rate, volume=volume)
        audio_srt.append({"audio": audio_path, "srt": srt_path})
    return audio_srt


def compose_intro(
    workspace: Path,
    text: str,
    config: dict,
    *,
    output_path: Path | None = None,
    image_indices: list[int] | None = None,
    split_sentences: bool = True,
    rate: str | None = None,
    volume: str | None = None,
    tv_frame: bool | None = None,
) -> Path:
    """合成片头：故事 TTS + Ken Burns + 专注引导短片 + 电视机框。"""
    from src.tools.video_assemble_tool import VideoAssembleTool
    from src.tools.tts_tool import TTSTool
    from src.video.intro_content import (
        build_focus_tagline_clip,
        concat_content_clips,
        resolve_focus_clip_config,
    )
    from src.video.intro_tv_frame import (
        composite_content_in_tv_frame,
        resolve_intro_frame_config,
    )

    workspace = Path(workspace)
    intro_cfg = config.get("intro") or {}
    video_cfg = config.get("video") or {}
    segments = split_intro_text(text) if split_sentences else [text.strip()]
    if not segments or not segments[0]:
        raise ValueError("片头文案不能为空")

    images = collect_intro_images(
        workspace, len(segments), indices=image_indices
    )
    intro_dir = workspace / "intro"
    audio_srt = generate_intro_assets(
        segments, intro_dir, config, rate=rate, volume=volume
    )

    if output_path is None:
        output_path = intro_dir / "intro.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_cfg = None
    if tv_frame is not False:
        frame_cfg = resolve_intro_frame_config(config)

    story_path = intro_dir / "story_part.mp4"
    intro_tmp = intro_dir / "tmp_video"
    tool = VideoAssembleTool(config)
    try:
        tool.run(
            images=images,
            audio_srt=audio_srt,
            output_path=story_path,
            workspace=workspace,
            tmp_dir=intro_tmp,
        )
    finally:
        shutil.rmtree(intro_tmp, ignore_errors=True)

    focus_clip, focus_tagline = resolve_focus_clip_config(intro_cfg)
    codec = str(intro_cfg.get("codec", video_cfg.get("codec", "libx264")))
    crf = int(intro_cfg.get("crf", video_cfg.get("crf", 18)))
    fps = int(intro_cfg.get("fps", video_cfg.get("fps", 30)))
    width, height = (int(v) for v in video_cfg.get("resolution", [1920, 1080]))

    if focus_clip is not None and focus_tagline:
        tagline_audio = intro_dir / "focus_tagline.mp3"
        tagline_srt = intro_dir / "focus_tagline.srt"
        TTSTool(config).run(
            focus_tagline,
            tagline_audio,
            tagline_srt,
            rate=rate,
            volume=volume,
        )
        focus_part = intro_dir / "focus_part.mp4"
        build_focus_tagline_clip(
            focus_clip,
            tagline_audio,
            focus_part,
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            crf=crf,
        )
        content_path = intro_dir / "content.mp4" if frame_cfg else output_path
        concat_content_clips(
            story_path,
            focus_part,
            content_path,
            codec=codec,
            crf=crf,
        )
    else:
        if frame_cfg:
            content_path = story_path
        else:
            if story_path.resolve() != output_path.resolve():
                shutil.copy2(story_path, output_path)
            return output_path

    if frame_cfg is None:
        return content_path

    return composite_content_in_tv_frame(content_path, output_path, frame_cfg)
