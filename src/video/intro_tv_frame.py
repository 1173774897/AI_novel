"""片头电视机框合成 - FFmpeg 蒙版嵌入 tv-frame。"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.logger import log
from src.utils.ffmpeg_helper import ensure_ffmpeg, get_ffmpeg_path, get_ffprobe_path
from src.video.tv_screen_mask import ContentFit, EdgeBow, ensure_screen_mask
from src.video.tv_speaker_audio import (
    TvSpeakerAudioConfig,
    build_tv_speaker_audio_filter,
    resolve_tv_speaker_audio_config,
)

DEFAULT_TV_FRAME = "media/tv-frame.png"
DEFAULT_TV_FRAME_REF_SIZE = (1280, 720)
DEFAULT_PATTERN = "media/intro-pattern.mp4"
DEFAULT_TV_SCREEN = {"x": 384, "y": 158, "w": 537, "h": 394}
DEFAULT_TV_SCREEN_CORNER_RADIUS = 20.0
DEFAULT_TV_SCREEN_MASK_FEATHER = 1.5
DEFAULT_CONTENT_FIT: ContentFit = "fill"


@dataclass(frozen=True)
class TvScreenRect:
    x: int
    y: int
    w: int
    h: int

    def scale(self, sx: float, sy: float) -> TvScreenRect:
        return TvScreenRect(
            x=int(round(self.x * sx)),
            y=int(round(self.y * sy)),
            w=int(round(self.w * sx)),
            h=int(round(self.h * sy)),
        )


@dataclass(frozen=True)
class IntroFrameConfig:
    tv_frame_image: Path
    tv_frame_size: tuple[int, int]
    screen: TvScreenRect
    screen_ref: TvScreenRect
    output_size: tuple[int, int]
    fps: int
    codec: str
    crf: int
    pattern_path: Path | None
    pattern_audio_volume: float
    screen_mask_path: Path | None
    screen_mask_corner_radius: float
    screen_mask_feather: float
    screen_edge_bow: EdgeBow
    content_fit: ContentFit
    use_screen_mask: bool
    tv_speaker_audio: TvSpeakerAudioConfig


def _run(cmd: list[str], description: str) -> None:
    ensure_ffmpeg()
    log.debug("%s: %s", description, " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"{description}失败: {stderr or exc}") from exc


def probe_stream_duration(path: Path, stream: str = "a:0") -> float:
    """获取指定流时长（秒）；优先音频流，配音长度为准。"""
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


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_media_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = resolve_project_root() / path
    return path


def _resolve_optional_media(raw: str | None) -> Path | None:
    if not raw:
        return None
    return _resolve_media_path(str(raw))


def _parse_content_fit(raw: str) -> ContentFit:
    fit = str(raw).strip().lower()
    if fit not in ("fill", "contain"):
        raise ValueError(f"intro.content_fit 无效: {raw!r}，应为 fill | contain")
    return fit  # type: ignore[return-value]


def _parse_edge_bow(raw: object) -> EdgeBow:
    from src.video.tv_screen_mask import _parse_edge_bow as parse_bow

    return parse_bow(raw)


def probe_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def resolve_intro_frame_config(config: dict) -> IntroFrameConfig | None:
    """从 config 解析电视机框参数；未启用或模板缺失时返回 None。"""
    intro_cfg = config.get("intro") or {}
    if intro_cfg.get("enabled", True) is False:
        return None

    tv_frame_path = _resolve_media_path(intro_cfg.get("tv_frame", DEFAULT_TV_FRAME))
    if not tv_frame_path.exists():
        log.warning("片头电视框图片不存在，跳过 TV 框合成: %s", tv_frame_path)
        return None

    ref_size = tuple(int(v) for v in intro_cfg.get("tv_frame_ref_size", DEFAULT_TV_FRAME_REF_SIZE))
    if len(ref_size) != 2:
        raise ValueError("intro.tv_frame_ref_size 必须为 [width, height]")

    configured_size = intro_cfg.get("tv_frame_size")
    if configured_size:
        tv_frame_size = tuple(int(v) for v in configured_size)
    else:
        tv_frame_size = probe_image_size(tv_frame_path)

    screen_cfg = intro_cfg.get("tv_screen") or DEFAULT_TV_SCREEN
    ref_screen = TvScreenRect(
        x=int(screen_cfg["x"]),
        y=int(screen_cfg["y"]),
        w=int(screen_cfg["w"]),
        h=int(screen_cfg["h"]),
    )
    sx = tv_frame_size[0] / ref_size[0]
    sy = tv_frame_size[1] / ref_size[1]
    screen = ref_screen.scale(sx, sy)

    video_cfg = config.get("video") or {}
    output_size = tuple(int(v) for v in video_cfg.get("resolution", tv_frame_size))
    if len(output_size) != 2:
        raise ValueError("video.resolution 必须为 [width, height]")

    scale_mode = intro_cfg.get("output_scale", "match_video")
    if scale_mode == "frame":
        output_size = tv_frame_size  # type: ignore[assignment]

    raw_pattern = intro_cfg.get("pattern", DEFAULT_PATTERN)
    pattern_path = _resolve_media_path(raw_pattern)
    if not pattern_path.exists():
        pattern_path = None

    return IntroFrameConfig(
        tv_frame_image=tv_frame_path,
        tv_frame_size=tv_frame_size,  # type: ignore[arg-type]
        screen=screen,
        screen_ref=ref_screen,
        output_size=output_size,  # type: ignore[arg-type]
        fps=int(intro_cfg.get("fps", video_cfg.get("fps", 30))),
        codec=str(intro_cfg.get("codec", video_cfg.get("codec", "libx264"))),
        crf=int(intro_cfg.get("crf", video_cfg.get("crf", 18))),
        pattern_path=pattern_path,
        pattern_audio_volume=float(intro_cfg.get("pattern_audio_volume", 0.0)),
        screen_mask_path=_resolve_optional_media(intro_cfg.get("tv_screen_mask")),
        screen_mask_corner_radius=float(
            intro_cfg.get("tv_screen_corner_radius", DEFAULT_TV_SCREEN_CORNER_RADIUS)
        ),
        screen_mask_feather=float(
            intro_cfg.get("tv_screen_mask_feather", DEFAULT_TV_SCREEN_MASK_FEATHER)
        ),
        screen_edge_bow=_parse_edge_bow(intro_cfg.get("tv_screen_edge_bow")),
        content_fit=_parse_content_fit(intro_cfg.get("content_fit", DEFAULT_CONTENT_FIT)),
        use_screen_mask=bool(intro_cfg.get("use_screen_mask", True)),
        tv_speaker_audio=resolve_tv_speaker_audio_config(intro_cfg),
    )


def _ensure_even_size(width: int, height: int) -> tuple[int, int]:
    """yuv420p 编码要求宽高为偶数。"""
    return width - width % 2, height - height % 2


def _build_content_scale_chain(sw: int, sh: int, *, content_fit: ContentFit) -> str:
    """按 fit 模式缩放内容至屏幕区域，末尾强制对齐到 sw×sh。"""
    exact = f"scale={sw}:{sh}:flags=lanczos"
    if content_fit == "fill":
        return (
            f"scale={sw}:{sh}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={sw}:{sh},{exact}"
        )
    return (
        f"scale={sw}:{sh}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={sw}:{sh}:(ow-iw)/2:(oh-ih)/2:black,{exact}"
    )


def _build_image_tv_frame_filter(
    *,
    narr_duration: float,
    fps: int,
    tw: int,
    th: int,
    screen: TvScreenRect,
    content_fit: ContentFit = "fill",
    use_screen_mask: bool = True,
) -> str:
    """FFmpeg filter：静态 tv-frame + 蒙版屏幕内容。"""
    ox, oy, sw, sh = screen.x, screen.y, screen.w, screen.h
    scale_chain = _build_content_scale_chain(sw, sh, content_fit=content_fit)

    frame_chain = (
        f"[0:v]scale={tw}:{th}:flags=lanczos,setsar=1,fps={fps},"
        f"format=yuv420p[frame]"
    )
    content_chain = (
        f"[1:v]{scale_chain},"
        f"eq=brightness=0.02:saturation=1.08,"
        f"trim=0:{narr_duration:.6f},setpts=PTS-STARTPTS,setsar=1[vid_raw]"
    )

    if use_screen_mask:
        content_chain += (
            f";[2:v]scale={sw}:{sh}:flags=lanczos:force_original_aspect_ratio=disable,"
            f"format=gray[mask_play]"
            f";[vid_raw][mask_play]alphamerge,format=yuva420p[scr]"
        )
    else:
        content_chain += ";[vid_raw]format=yuv420p[scr]"

    play_chain = (
        f";[frame][scr]overlay={ox}:{oy}:format=auto:"
        f"enable='between(t,0,{narr_duration:.6f})',"
        f"format=yuv420p[outv]"
    )
    return f"{frame_chain};{content_chain}{play_chain}"


def resolve_work_screen_mask(
    frame_cfg: IntroFrameConfig,
    work_dir: Path,
) -> Path | None:
    """解析或生成屏幕蒙版；未启用时返回 None。"""
    if not frame_cfg.use_screen_mask:
        return None
    auto_path = work_dir / "screen_mask.png"
    configured = frame_cfg.screen_mask_path
    return ensure_screen_mask(
        configured,
        ref_width=frame_cfg.screen_ref.w,
        ref_height=frame_cfg.screen_ref.h,
        corner_radius=frame_cfg.screen_mask_corner_radius,
        feather=frame_cfg.screen_mask_feather,
        edge_bow=frame_cfg.screen_edge_bow,
        auto_path=auto_path if configured is None or not configured.exists() else None,
    )


def _composite_with_ffmpeg(
    content_path: Path,
    output_path: Path,
    frame_cfg: IntroFrameConfig,
    *,
    narr_duration: float,
) -> Path:
    tw, th = _ensure_even_size(*frame_cfg.output_size)
    ow, oh = frame_cfg.tv_frame_size
    sx, sy = tw / ow, th / oh
    screen = frame_cfg.screen.scale(sx, sy)
    work_dir = content_path.parent
    mask_path = resolve_work_screen_mask(frame_cfg, work_dir)
    use_mask = mask_path is not None

    video_filter = _build_image_tv_frame_filter(
        narr_duration=narr_duration,
        fps=frame_cfg.fps,
        tw=tw,
        th=th,
        screen=screen,
        content_fit=frame_cfg.content_fit,
        use_screen_mask=use_mask,
    )

    cmd = [
        get_ffmpeg_path(), "-y",
        "-loop", "1",
        "-i", str(frame_cfg.tv_frame_image),
        "-i", str(content_path),
    ]
    if use_mask and mask_path is not None:
        cmd.extend(["-loop", "1", "-i", str(mask_path)])

    filter_complex = video_filter
    audio_map = "1:a:0"
    if frame_cfg.tv_speaker_audio.enabled:
        af = build_tv_speaker_audio_filter(frame_cfg.tv_speaker_audio)
        filter_complex = f"{video_filter};[1:a]{af}[aout]"
        audio_map = "[aout]"

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", audio_map,
        "-t", f"{narr_duration:.6f}",
        "-c:v", frame_cfg.codec,
        "-crf", str(frame_cfg.crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ])
    mask_note = f", mask={mask_path.name}" if use_mask else ""
    audio_note = ", tv_speaker" if frame_cfg.tv_speaker_audio.enabled else ""
    log.info("FFmpeg 电视机框合成: fit=%s%s%s", frame_cfg.content_fit, mask_note, audio_note)
    _run(cmd, "FFmpeg 电视机框片头合成")
    return output_path


def composite_content_in_tv_frame(
    content_path: Path,
    output_path: Path,
    frame_cfg: IntroFrameConfig,
) -> Path:
    """FFmpeg 将完整片头内容嵌入 tv-frame 电视屏幕。"""
    content_path = Path(content_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    narr_duration = probe_stream_duration(content_path, "a:0")
    log.info(
        "电视机框合成: frame=%s, 旁白=%.2fs",
        frame_cfg.tv_frame_image.name,
        narr_duration,
    )
    return _composite_with_ffmpeg(
        content_path, output_path, frame_cfg, narr_duration=narr_duration
    )


def calibrate_intro_tv_screen(
    config: dict,
    *,
    mask_output: Path,
    debug_output: Path,
) -> tuple[Path, Path]:
    """生成屏幕蒙版并在 tv-frame 上输出校准预览图。"""
    from src.video.tv_screen_mask import render_calibration_preview

    frame_cfg = resolve_intro_frame_config(config)
    if frame_cfg is None:
        raise RuntimeError("intro 未启用或 tv-frame 资源缺失")

    mask_path = save_screen_mask_from_config(frame_cfg, mask_output)
    preview = render_calibration_preview(
        frame_cfg.tv_frame_image,
        screen_x=frame_cfg.screen.x,
        screen_y=frame_cfg.screen.y,
        mask_path=mask_path,
        output_path=debug_output,
    )
    log.info(
        "屏幕蒙版: %s (%dx%d, r=%.1f)",
        mask_path,
        frame_cfg.screen_ref.w,
        frame_cfg.screen_ref.h,
        frame_cfg.screen_mask_corner_radius,
    )
    log.info("校准预览: %s", preview)
    return mask_path, preview


def save_screen_mask_from_config(frame_cfg: IntroFrameConfig, output_path: Path) -> Path:
    """按配置写入屏幕蒙版文件。"""
    from src.video.tv_screen_mask import save_screen_mask

    return save_screen_mask(
        output_path,
        frame_cfg.screen_ref.w,
        frame_cfg.screen_ref.h,
        corner_radius=frame_cfg.screen_mask_corner_radius,
        feather=frame_cfg.screen_mask_feather,
        edge_bow=frame_cfg.screen_edge_bow,
    )
