"""将片头、正片、片尾拼接为完整成片。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from src.logger import log
from src.utils.ffmpeg_helper import ensure_ffmpeg, get_ffmpeg_path, get_ffprobe_path
from src.video.intro_content import probe_media_duration, probe_video_stream_size


def resolve_project_stem(workspace: Path) -> str:
    """从 agent_state 或目录名推断正片文件名 stem。"""
    state_path = workspace / "agent_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            input_file = state.get("input_file")
            if input_file:
                return Path(input_file).stem
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("读取 agent_state 失败，回退目录名: %s", exc)
    return workspace.name


def resolve_default_paths(
    workspace: Path,
    config: dict,
    *,
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """返回默认 (intro, main, ending) 路径。"""
    stem = resolve_project_stem(workspace)
    out_dir = Path(config.get("project", {}).get("default_output", "output"))
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    main = out_dir / f"{stem}.mp4"
    intro = workspace / "intro" / "intro.mp4"
    ending = workspace / "intro" / "ending.mp4"
    return intro, main, ending


def _resolve_encode_params(config: dict) -> tuple[str, int, int]:
    video_cfg = config.get("video") or {}
    intro_cfg = config.get("intro") or {}
    codec = str(intro_cfg.get("codec", video_cfg.get("codec", "libx264")))
    crf = int(intro_cfg.get("crf", video_cfg.get("crf", 18)))
    fps = int(intro_cfg.get("fps", video_cfg.get("fps", 30)))
    return codec, crf, fps


def _probe_audio_spec(path: Path) -> tuple[int, int]:
    """读取首条音频流的采样率与声道数。"""
    ensure_ffmpeg()
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        return 48000, 2
    return int(streams[0].get("sample_rate", 48000)), int(
        streams[0].get("channels", 2)
    )


def _find_main_clip_index(clips: list[Path]) -> int:
    """以时长最长的片段作为正片（分辨率/音频基准）。"""
    durations = [probe_media_duration(p, stream="v:0") for p in clips]
    return max(range(len(clips)), key=lambda i: durations[i])


def _clip_matches_target(
    path: Path,
    *,
    width: int,
    height: int,
    sample_rate: int,
    channels: int,
) -> bool:
    w, h = probe_video_stream_size(path)
    sr, ch = _probe_audio_spec(path)
    return w == width and h == height and sr == sample_rate and ch == channels


def _normalize_clip(
    src: Path,
    dst: Path,
    *,
    width: int,
    height: int,
    fps: int,
    sample_rate: int,
    channels: int,
    codec: str,
    crf: int,
) -> Path:
    """将短视频缩放/转码到与正片一致的参数（仅用于片头/片尾）。"""
    ensure_ffmpeg()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    layout = "mono" if channels == 1 else "stereo"
    filter_complex = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps},"
        f"format=yuv420p[v];"
        f"[0:a]aformat=sample_rates={sample_rate}:channel_layouts={layout}[a]"
    )
    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", str(src),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", codec,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(dst),
    ]
    if codec == "libx265":
        cmd.extend(["-tag:v", "hvc1"])
    log.info("归一化短片: %s -> %dx%d", src.name, width, height)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return dst


def _concat_copy(clips: list[Path], output_path: Path, tmp_dir: Path) -> Path:
    """无损拼接已对齐参数的 MP4。"""
    ensure_ffmpeg()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    concat_list = tmp_dir / "concat_full.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for clip in clips:
            safe = str(clip.resolve()).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    cmd = [
        get_ffmpeg_path(), "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    log.info("无损拼接 %d 段 -> %s", len(clips), output_path.name)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def merge_intro_main_ending(
    clips: list[Path],
    output_path: Path,
    config: dict,
    *,
    tmp_dir: Path | None = None,
) -> Path:
    """按顺序拼接多段视频。

    以正片（最长段）的分辨率/音频为基准，仅重编码片头/片尾，正片无损拼接。
    """
    if not clips:
        raise ValueError("至少需提供一段视频")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        if not clip.exists() or clip.stat().st_size < 100:
            raise FileNotFoundError(f"视频不存在或为空: {clip}")

    if len(clips) == 1:
        if clips[0].resolve() != output_path.resolve():
            shutil.copy2(clips[0], output_path)
        return output_path

    codec, crf, fps = _resolve_encode_params(config)
    work_tmp = Path(tmp_dir) if tmp_dir else output_path.parent / ".merge_tmp"
    work_tmp.mkdir(parents=True, exist_ok=True)

    main_idx = _find_main_clip_index(clips)
    main_clip = clips[main_idx]
    width, height = probe_video_stream_size(main_clip)
    sample_rate, channels = _probe_audio_spec(main_clip)

    prepared: list[Path] = []
    temps: list[Path] = []
    for idx, clip in enumerate(clips):
        if _clip_matches_target(
            clip,
            width=width,
            height=height,
            sample_rate=sample_rate,
            channels=channels,
        ):
            prepared.append(clip)
            continue
        norm_path = work_tmp / f"_norm_{idx:02d}.mp4"
        _normalize_clip(
            clip,
            norm_path,
            width=width,
            height=height,
            fps=fps,
            sample_rate=sample_rate,
            channels=channels,
            codec=codec,
            crf=crf,
        )
        prepared.append(norm_path)
        temps.append(norm_path)

    try:
        _concat_copy(prepared, output_path, work_tmp)
    finally:
        for t in temps:
            t.unlink(missing_ok=True)
        (work_tmp / "concat_full.txt").unlink(missing_ok=True)

    log.info("完整成片已生成: %s (%d 段)", output_path, len(clips))
    return output_path
