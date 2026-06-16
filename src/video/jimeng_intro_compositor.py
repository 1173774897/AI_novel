"""即梦 CLI 片头合成 - multimodal2video 将介绍视频嵌入 tv-frame 电视机屏幕。"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.utils.ffmpeg_helper import ensure_ffmpeg, get_ffmpeg_path, get_ffprobe_path

log = logging.getLogger("novel")

_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}
_POLL_INTERVAL = 2.0

DEFAULT_PROMPT = (
    "参考图片是2010年代中国家庭客厅的CRT电视机固定场景，镜头不动。"
    "参考视频是需要在电视机屏幕内播放的片头介绍视频。"
    "请将参考视频精确嵌入参考图片中CRT电视机屏幕区域内播放，"
    "保持客厅环境、机位与光影不变。"
    "参考视频播放完毕后，电视机屏幕逐渐熄灭变黑，模拟关电视，"
    "客厅其余部分保持静止。"
    "使用提供的旁白音频，音画同步，旁白完整播完。"
)


@dataclass(frozen=True)
class JimengIntroSettings:
    cli_command: str
    model_version: str
    video_resolution: str
    ratio: str
    poll: int
    timeout: float
    prompt: str
    extra_args: tuple[str, ...]


def resolve_jimeng_intro_settings(config: dict) -> JimengIntroSettings:
    intro_cfg = (config.get("intro") or {}).get("jimeng_cli") or {}
    imagegen_cfg = config.get("imagegen") or {}
    return JimengIntroSettings(
        cli_command=str(
            intro_cfg.get("cli_command")
            or imagegen_cfg.get("cli_command")
            or "dreamina"
        ),
        model_version=str(intro_cfg.get("model_version", "seedance2.0fast")),
        video_resolution=str(intro_cfg.get("video_resolution", "720p")),
        ratio=str(intro_cfg.get("ratio", "16:9")),
        poll=int(intro_cfg.get("poll", 600)),
        timeout=float(intro_cfg.get("timeout", 600)),
        prompt=str(intro_cfg.get("prompt", DEFAULT_PROMPT)).strip(),
        extra_args=tuple(intro_cfg.get("extra_args") or ()),
    )


def jimeng_max_duration() -> int:
    """multimodal2video 支持的 duration 上限。"""
    return 15


JIMENG_AUDIO_LIMIT = 15.0
JIMENG_AUDIO_SAFE_MARGIN = 0.05


def jimeng_safe_content_duration(configured_max: float) -> float:
    """即梦上传音频严格 ≤15s，预留编码/容器余量。"""
    cap = min(float(configured_max), JIMENG_AUDIO_LIMIT)
    return max(2.0, cap - JIMENG_AUDIO_SAFE_MARGIN)


def assert_jimeng_uploadable_audio(audio_path: Path) -> Path:
    """上传前校验旁白音频在即梦允许范围内 [2, 15]。"""
    actual = probe_stream_duration(audio_path, "a:0")
    if actual > JIMENG_AUDIO_LIMIT:
        raise RuntimeError(
            f"旁白音频 {actual:.3f}s 超出即梦上传上限 {JIMENG_AUDIO_LIMIT:.0f}s"
        )
    if actual < 2.0:
        raise RuntimeError(
            f"旁白音频 {actual:.3f}s 短于即梦下限 2s"
        )
    return audio_path


def probe_stream_duration(path: Path, stream: str = "a:0") -> float:
    """获取指定流时长（秒）。"""
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


def fits_jimeng_duration(narr_duration: float, tv_off_duration: float) -> bool:
    """旁白与即梦 API 时长限制是否兼容（旁白须严格 ≤15s）。"""
    _eps = 0.001
    return (
        narr_duration <= JIMENG_AUDIO_LIMIT + _eps
        and narr_duration >= 2.0
    )


def compute_jimeng_output_duration(
    narr_duration: float, tv_off_duration: float
) -> int:
    """计算提交给即梦的成片时长（4-15 秒整数）。"""
    total = narr_duration + tv_off_duration
    capped = min(jimeng_max_duration(), total)
    return max(4, min(jimeng_max_duration(), int(round(capped))))


def _run_cli(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    log.debug("即梦 CLI: %s", " ".join(cmd[:12]) + (" ..." if len(cmd) > 12 else ""))
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"即梦 CLI 超时 ({timeout}s)") from exc


def _parse_cli_json(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise RuntimeError("即梦 CLI 无输出")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise RuntimeError(f"即梦 CLI 返回异常结构: {type(data).__name__}")
    return data


def _find_video_paths(obj: object) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value not in seen:
            seen.add(value)
            found.append(value)

    def walk(value: object) -> None:
        if isinstance(value, str):
            lower = value.lower()
            if any(lower.endswith(ext) for ext in _VIDEO_SUFFIXES):
                add(value)
            return
        if isinstance(value, dict):
            for key in ("path", "file", "filePath", "output", "video", "videoPath", "url"):
                if key in value:
                    walk(value[key])
            for item in value.values():
                walk(item)
            return
        if isinstance(value, list):
            for item in value:
                walk(item)

    walk(obj)
    return found


def _newest_video_in_dir(directory: Path) -> Path | None:
    candidates = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _ensure_command_available(command: str) -> None:
    if shutil.which(command):
        return
    raise RuntimeError(
        f"未找到即梦 CLI 命令 {command!r}。"
        "请安装: curl -s https://jimeng.jianying.com/cli | bash && dreamina login"
    )


def extract_narration_audio(
    content_path: Path,
    output_path: Path,
    *,
    max_duration: float,
) -> Path:
    """从内容视频提取旁白音频，硬截断至 max_duration 以内。"""
    ensure_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trim_to = min(max_duration, JIMENG_AUDIO_LIMIT - 0.001)
    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", str(content_path),
        "-vn",
        "-af", f"atrim=0:{trim_to:.6f},asetpts=PTS-STARTPTS",
        "-t", f"{trim_to:.6f}",
        "-acodec", "aac",
        "-b:a", "192k",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return assert_jimeng_uploadable_audio(output_path)


def strip_video_audio(
    content_path: Path,
    output_path: Path,
    *,
    max_duration: float | None = None,
) -> Path:
    """生成无音轨的内容参考视频（可选截断至与旁白等长）。"""
    ensure_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = [
        get_ffmpeg_path(), "-y",
        "-i", str(content_path),
        "-an",
    ]
    if max_duration is not None:
        cmd = [
            *base,
            "-t", f"{max_duration:.6f}",
            "-c:v", "libx264",
            "-crf", "18",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path

    cmd = [*base, "-c:v", "copy", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        cmd = [
            *base,
            "-c:v", "libx264", "-crf", "18",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def _build_multimodal_command(
    settings: JimengIntroSettings,
    *,
    tv_frame_image: Path,
    content_video: Path,
    audio_path: Path,
    duration: int,
) -> list[str]:
    cmd = [
        settings.cli_command,
        "multimodal2video",
        "--image", str(tv_frame_image),
        "--video", str(content_video),
        "--audio", str(audio_path),
        "--prompt", settings.prompt,
        f"--duration={duration}",
        f"--ratio={settings.ratio}",
        f"--model_version={settings.model_version}",
        f"--video_resolution={settings.video_resolution}",
        "--poll=0",
    ]
    cmd.extend(settings.extra_args)
    return cmd


def _poll_result(
    settings: JimengIntroSettings,
    submit_id: str,
    download_dir: Path,
) -> dict:
    deadline = time.monotonic() + settings.timeout
    last_data: dict = {}

    while time.monotonic() < deadline:
        result = _run_cli(
            [
                settings.cli_command,
                "query_result",
                f"--submit_id={submit_id}",
                f"--download_dir={download_dir}",
            ],
            settings.timeout,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[:500]
            raise RuntimeError(
                f"即梦 query_result 失败 (exit={result.returncode}): {detail}"
            )

        data = _parse_cli_json(result.stdout)
        last_data = data
        status = data.get("gen_status")
        if status == "success":
            return data
        if status == "fail":
            reason = data.get("fail_reason") or "未知原因"
            raise RuntimeError(f"即梦片头合成失败: {reason}")

        time.sleep(_POLL_INTERVAL)

    raise RuntimeError(
        f"即梦片头合成轮询超时 ({settings.timeout}s, submit_id={submit_id}, "
        f"last_status={last_data.get('gen_status')})"
    )


def _resolve_output_video(data: dict, download_dir: Path) -> Path:
    videos = (data.get("result_json") or {}).get("videos") or []
    if videos:
        first = videos[0]
        path_value = first.get("path")
        if path_value:
            path = Path(path_value)
            if path.is_file():
                return path

    for candidate in _find_video_paths(data):
        path = Path(candidate)
        if path.is_file():
            return path

    newest = _newest_video_in_dir(download_dir)
    if newest is not None:
        return newest

    raise RuntimeError("即梦 CLI 响应中无可用视频文件")


def composite_via_jimeng_cli(
    *,
    tv_frame_image: Path,
    content_video: Path,
    narration_audio: Path,
    output_path: Path,
    settings: JimengIntroSettings,
    duration: int,
    work_dir: Path,
) -> Path:
    """调用 dreamina multimodal2video：tv-frame 图片 + 介绍视频 + 旁白。"""
    _ensure_command_available(settings.cli_command)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_multimodal_command(
        settings,
        tv_frame_image=tv_frame_image,
        content_video=content_video,
        audio_path=narration_audio,
        duration=duration,
    )
    log.info(
        "即梦片头合成: frame=%s, content=%s, duration=%ds",
        tv_frame_image.name,
        content_video.name,
        duration,
    )

    result = _run_cli(cmd, min(180.0, settings.timeout))
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:800]
        raise RuntimeError(
            f"即梦 multimodal2video 失败 (exit={result.returncode}): {detail}"
        )

    data = _parse_cli_json(result.stdout)
    status = data.get("gen_status")
    if status == "querying":
        submit_id = data.get("submit_id")
        if not submit_id:
            raise RuntimeError("即梦 CLI 返回 querying 但缺少 submit_id")
        data = _poll_result(settings, submit_id, work_dir)
    elif status == "fail":
        reason = data.get("fail_reason") or "未知原因"
        raise RuntimeError(f"即梦片头合成失败: {reason}")
    elif status != "success":
        raise RuntimeError(f"即梦 CLI 返回未知状态: {status!r}")

    video_path = _resolve_output_video(data, work_dir)
    if video_path.resolve() != output_path.resolve():
        shutil.copy2(video_path, output_path)
    return output_path
