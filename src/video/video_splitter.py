"""原视频切分与片段拼接（v2v 角色替换等流程）。"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.utils.ffmpeg_helper import ensure_ffmpeg, get_ffmpeg_path, get_ffprobe_path
from src.video.jimeng_intro_compositor import probe_stream_duration

JIMENG_MIN_SEGMENT_SEC = 4.0
JIMENG_UPLOAD_MAX_SEC = 15.0  # 即梦 API 硬上限 [2, 15]（含等号）
# FFmpeg 切分常多出几帧（如 15.015s），上传前须留余量
JIMENG_SAFE_MAX_CLIP_SEC = 14.92
# 规划切分时的默认上限（与 SAFE 一致）
JIMENG_MAX_SEGMENT_SEC = JIMENG_SAFE_MAX_CLIP_SEC


def clamp_clip_duration(
    duration_sec: float,
    *,
    safe_max: float = JIMENG_SAFE_MAX_CLIP_SEC,
    min_sec: float = JIMENG_MIN_SEGMENT_SEC,
) -> float:
    """将切分/上传时长钳在即梦允许范围内。"""
    d = float(duration_sec)
    return max(min_sec, min(safe_max, d))


@dataclass(frozen=True)
class VideoSegmentSpan:
    """原视频上的一段区间（秒）。"""

    id: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def plan_segment_spans(
    total_duration: float,
    *,
    max_seg: float = JIMENG_MAX_SEGMENT_SEC,
    min_seg: float = JIMENG_MIN_SEGMENT_SEC,
) -> list[VideoSegmentSpan]:
    """将总时长规划为不超过 max_seg 的片段，且每段 >= min_seg（即梦限制）。"""
    total = max(0.0, float(total_duration))
    if total <= 0:
        return []
    if total <= max_seg:
        return [VideoSegmentSpan(id=1, start_sec=0.0, end_sec=total)]

    spans: list[VideoSegmentSpan] = []
    start = 0.0
    seg_id = 1
    while start < total - 0.05:
        remaining = total - start
        if remaining <= max_seg:
            if remaining < min_seg and spans:
                prev = spans[-1]
                spans[-1] = VideoSegmentSpan(
                    id=prev.id, start_sec=prev.start_sec, end_sec=total
                )
            else:
                spans.append(VideoSegmentSpan(id=seg_id, start_sec=start, end_sec=total))
            break

        end = start + max_seg
        tail = total - end
        if 0 < tail < min_seg:
            end = total - min_seg
        spans.append(VideoSegmentSpan(id=seg_id, start_sec=start, end_sec=end))
        start = end
        seg_id += 1

    return spans


def extract_video_clip(
    source: Path,
    output: Path,
    *,
    start_sec: float,
    duration_sec: float,
    strip_audio: bool = False,
) -> Path:
    """从原视频截取一段并重新编码为 mp4。"""
    ensure_ffmpeg()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = clamp_clip_duration(duration_sec)
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration_sec:.6f}",
    ]
    if strip_audio:
        cmd.extend(["-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"])
    else:
        cmd.extend(["-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p"])
    cmd.append(str(output))
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return ensure_clip_uploadable(output)


def ensure_clip_uploadable(
    clip_path: Path,
    *,
    safe_max: float = JIMENG_SAFE_MAX_CLIP_SEC,
) -> Path:
    """若 clip 仍略超 15s（容器/编码余量），硬截断到 safe_max。"""
    clip_path = Path(clip_path)
    try:
        actual = probe_stream_duration(clip_path, "v:0")
    except RuntimeError:
        return clip_path
    if actual <= JIMENG_UPLOAD_MAX_SEC - 0.004:
        return clip_path

    ensure_ffmpeg()
    tmp = clip_path.with_name(f"{clip_path.stem}_trim{clip_path.suffix}")
    trim_to = min(safe_max, JIMENG_UPLOAD_MAX_SEC - 0.01)
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-i",
        str(clip_path),
        "-t",
        f"{trim_to:.3f}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-pix_fmt",
        "yuv420p",
        str(tmp),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    tmp.replace(clip_path)
    return clip_path


def extract_audio_clip(
    source: Path,
    output: Path,
    *,
    start_sec: float,
    duration_sec: float,
    max_duration: float = JIMENG_MAX_SEGMENT_SEC,
) -> Path | None:
    """提取片段音频；若短于 2s 或长于即梦上限则返回 None。"""
    ensure_ffmpeg()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trim = clamp_clip_duration(
        duration_sec, safe_max=min(float(max_duration), JIMENG_SAFE_MAX_CLIP_SEC)
    )
    if trim < 2.0:
        return None
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source),
        "-t",
        f"{trim:.6f}",
        "-vn",
        "-acodec",
        "aac",
        "-b:a",
        "192k",
        str(output),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    try:
        actual = probe_stream_duration(output, "a:0")
    except RuntimeError:
        return None
    if actual < 2.0 or actual > JIMENG_UPLOAD_MAX_SEC - 0.004:
        return None
    return output


def probe_video_duration(path: Path) -> float:
    """读取视频文件时长（秒）。"""
    return probe_stream_duration(path, "v:0")


def extract_last_frame(video_path: Path, output_image: Path) -> Path:
    """提取视频最后一帧为 PNG（用作扩演 image2video 首帧锚点）。"""
    ensure_ffmpeg()
    video_path = Path(video_path)
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)

    duration = probe_video_duration(video_path)
    # 距结尾略留余量；-sseof -0.05 在 4s 短片上常取不到帧
    seek = max(0.0, duration - 0.12)
    attempts = [
        [
            get_ffmpeg_path(),
            "-y",
            "-ss",
            f"{seek:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_image),
        ],
        [
            get_ffmpeg_path(),
            "-y",
            "-sseof",
            "-0.15",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_image),
        ],
    ]
    last_err = ""
    for cmd in attempts:
        if output_image.is_file():
            output_image.unlink()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and output_image.is_file() and output_image.stat().st_size > 0:
            return output_image
        last_err = (proc.stderr or proc.stdout or "").strip()

    raise RuntimeError(
        f"无法提取视频末帧: {video_path}"
        + (f" ({last_err[-200:]})" if last_err else "")
    )


def concat_videos(clips: list[Path], output: Path) -> Path:
    """按顺序无损拼接多个 mp4。"""
    ensure_ffmpeg()
    if not clips:
        raise ValueError("concat_videos: 无输入片段")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as list_file:
        for clip in clips:
            escaped = str(Path(clip).resolve()).replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")
        list_path = list_file.name

    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        str(output),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            cmd_reencode = [
                get_ffmpeg_path(),
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-c:a",
                "aac",
                str(output),
            ]
            subprocess.run(cmd_reencode, check=True, capture_output=True, text=True)
    finally:
        Path(list_path).unlink(missing_ok=True)
    return output
