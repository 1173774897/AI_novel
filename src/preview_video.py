"""预览合成 - 仅组装 workspace 前 N 个分镜。"""

from __future__ import annotations

from pathlib import Path


def collect_segment_assets(workspace: Path, count: int) -> tuple[list[Path], list[dict]]:
    """收集前 count 个分镜的图片、音频、字幕路径。"""
    if count < 1:
        raise ValueError("count 必须 >= 1")

    img_dir = workspace / "images"
    audio_dir = workspace / "audio"
    srt_dir = workspace / "subtitles"

    images: list[Path] = []
    audio_srt: list[dict] = []

    for i in range(count):
        img = img_dir / f"{i:04d}.png"
        audio = audio_dir / f"{i:04d}.mp3"
        srt = srt_dir / f"{i:04d}.srt"
        missing = [
            p.name for p in (img, audio)
            if not p.exists() or p.stat().st_size < 100
        ]
        if missing:
            raise FileNotFoundError(
                f"分镜 {i} 素材不完整，缺少或为空: {', '.join(missing)}"
            )
        images.append(img)
        audio_srt.append({"audio": audio, "srt": srt})

    return images, audio_srt


def preview_workspace(
    workspace: Path,
    config: dict,
    count: int = 2,
    output_path: Path | None = None,
) -> Path:
    """仅合成 workspace 前 count 个分镜，用于快速预览字幕与特效。"""
    from src.tools.video_assemble_tool import VideoAssembleTool

    workspace = Path(workspace)
    images, audio_srt = collect_segment_assets(workspace, count)

    if output_path is None:
        out_dir = Path(config.get("project", {}).get("default_output", "output"))
        output_path = out_dir / f"{workspace.name}_preview_{count}.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tool = VideoAssembleTool(config)
    return tool.run(
        images=images,
        audio_srt=audio_srt,
        output_path=output_path,
        workspace=workspace,
    )
