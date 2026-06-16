"""片头内容拼接 - 故事段 Ken Burns + 专注引导短片。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from src.logger import log
from src.utils.ffmpeg_helper import ensure_ffmpeg, get_ffmpeg_path, get_ffprobe_path


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_media_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = resolve_project_root() / path
    return path


def resolve_focus_clip_config(intro_cfg: dict) -> tuple[Path | None, str]:
    """解析专注引导短片与口播文案。"""
    raw_clip = intro_cfg.get("focus_clip", "media/关掉杂念故事开始咯.mp4")
    if raw_clip is False or raw_clip == 0 or raw_clip == "":
        return None, ""
    clip_path = _resolve_media_path(str(raw_clip))
    tagline = str(
        intro_cfg.get("focus_tagline", "关掉杂念，故事开始咯")
    ).strip()
    if not clip_path.exists():
        log.warning("专注引导短片不存在，跳过: %s", clip_path)
        return None, tagline
    return clip_path, tagline


def resolve_ending_config(ending_cfg: dict) -> tuple[Path, str, Path | None]:
    """解析片尾短片、口播文案与关电视片段。"""
    raw_clip = ending_cfg.get("clip", "media/故事讲完了感谢收听我们下次再见.mp4")
    clip_path = _resolve_media_path(str(raw_clip))
    tagline = str(
        ending_cfg.get("tagline", "故事讲完了，感谢收听。我们下次再见")
    ).strip()
    if not clip_path.exists():
        raise FileNotFoundError(f"片尾短片不存在: {clip_path}")
    if not tagline:
        raise ValueError("片尾口播文案不能为空")

    raw_shutdown = ending_cfg.get("shutdown_clip", "media/tv-shotdown.mp4")
    if raw_shutdown is False or raw_shutdown == 0 or raw_shutdown == "":
        return clip_path, tagline, None
    shutdown_path = _resolve_media_path(str(raw_shutdown))
    if not shutdown_path.exists():
        raise FileNotFoundError(f"关电视片段不存在: {shutdown_path}")
    return clip_path, tagline, shutdown_path


def probe_media_duration(path: Path, stream: str = "a:0") -> float:
    ensure_ffmpeg()
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-select_streams", stream,
        "-show_entries", "stream=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if streams and streams[0].get("duration"):
        return float(streams[0]["duration"])

    cmd_fmt = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd_fmt, check=True, capture_output=True, text=True)
    duration = json.loads(result.stdout).get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError(f"无法读取媒体时长: {path}")
    return float(duration)


def _atempo_chain(factor: float) -> str:
    if factor <= 0:
        raise ValueError(f"atempo factor 必须 > 0: {factor}")
    parts: list[str] = []
    remaining = factor
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def fit_audio_to_duration(
    input_path: Path,
    output_path: Path,
    target_duration: float,
) -> Path:
    """将音频加速/减速并截断至与目标视频等长。"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current = probe_media_duration(input_path, "a:0")
    if current <= 0 or target_duration <= 0:
        raise ValueError(f"时长无效: audio={current}, target={target_duration}")

    if abs(current - target_duration) < 0.01:
        if input_path.resolve() != output_path.resolve():
            shutil.copy2(input_path, output_path)
        return output_path

    speed = current / target_duration
    filter_complex = (
        f"{_atempo_chain(speed)},"
        f"atrim=0:{target_duration:.6f},asetpts=PTS-STARTPTS"
    )
    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", str(input_path),
        "-af", filter_complex,
        "-t", f"{target_duration:.6f}",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def build_focus_tagline_clip(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    codec: str,
    crf: int,
) -> Path:
    """用 TTS 替换专注短片原声，视频时长为准。"""
    ensure_ffmpeg()
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = probe_media_duration(video_path, "v:0")
    fitted_audio = output_path.parent / "focus_tagline_audio.m4a"
    fit_audio_to_duration(audio_path, fitted_audio, duration)

    w = width - width % 2
    h = height - height % 2
    filter_complex = (
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps},format=yuv420p[v]"
    )
    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", str(video_path),
        "-i", str(fitted_audio),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "1:a:0",
        "-t", f"{duration:.6f}",
        "-c:v", codec,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    log.info(
        "专注引导短片: %s, 时长 %.2fs, 口播对齐视频",
        video_path.name,
        duration,
    )
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def concat_content_clips(
    first: Path,
    second: Path,
    output_path: Path,
    *,
    codec: str,
    crf: int,
) -> Path:
    """拼接故事段与专注引导段（统一音视频参数）。"""
    ensure_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (
        "[0:a]aformat=sample_rates=48000:channel_layouts=stereo[a0];"
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[a1];"
        "[0:v][a0][1:v][a1]concat=n=2:v=1:a=1[outv][outa]"
    )
    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", str(first),
        "-i", str(second),
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", codec,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    log.info("拼接片头内容: %s + %s", first.name, second.name)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def probe_video_stream_size(path: Path) -> tuple[int, int]:
    """读取视频流宽高（偶数对齐）。"""
    ensure_ffmpeg()
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"无法读取视频尺寸: {path}")
    w = int(streams[0]["width"])
    h = int(streams[0]["height"])
    return w - w % 2, h - h % 2


def append_video_clip(
    first: Path,
    second: Path,
    output_path: Path,
    *,
    codec: str,
    crf: int,
    fps: int,
    second_audio_volume: float = 1.0,
) -> Path:
    """将 second 缩放后拼接到 first 之后（统一分辨率与帧率）。"""
    ensure_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = probe_video_stream_size(first)
    if second_audio_volume <= 0:
        raise ValueError(f"second_audio_volume 必须 > 0: {second_audio_volume}")
    second_audio_chain = (
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo"
    )
    if abs(second_audio_volume - 1.0) >= 0.001:
        second_audio_chain += f",volume={second_audio_volume:.6f}"
    second_audio_chain += "[a1];"
    filter_complex = (
        "[0:a]aformat=sample_rates=48000:channel_layouts=stereo[a0];"
        f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps},format=yuv420p[v1];"
        f"{second_audio_chain}"
        "[0:v][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]"
    )
    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", str(first),
        "-i", str(second),
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", codec,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    log.info("拼接视频: %s + %s", first.name, second.name)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def prepend_black_lead(
    input_path: Path,
    output_path: Path,
    *,
    duration: float,
    codec: str,
    crf: int,
    fps: int,
) -> Path:
    """在视频最前拼接指定时长的黑屏（静音）。"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if duration <= 0:
        if input_path.resolve() != output_path.resolve():
            shutil.copy2(input_path, output_path)
        return output_path

    ensure_ffmpeg()
    w, h = probe_video_stream_size(input_path)
    filter_complex = (
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS[v0];"
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,"
        "asetpts=PTS-STARTPTS[a0];"
        f"[2:v]setsar=1,fps={fps},format=yuv420p,setpts=PTS-STARTPTS[v1];"
        "[2:a]aformat=sample_rates=48000:channel_layouts=stereo,"
        "asetpts=PTS-STARTPTS[a1];"
        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]"
    )
    cmd = [
        get_ffmpeg_path(), "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps}:d={duration:.6f}",
        "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={duration:.6f}",
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", codec,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    log.info("片尾前导黑屏: %.2fs + %s", duration, input_path.name)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path
