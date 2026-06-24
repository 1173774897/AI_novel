"""即梦 CLI 本地视频生成后端。

通过 subprocess 调用 dreamina / jimeng 命令行，复用已登录账号积分，无需火山方舟 API Key。

典型配置 (config.yaml)::

    director:
      videogen:
        backend: jimeng-cli
        cli_flavor: dreamina
        async_submit: true       # 只提交不等待，适合排队数小时
        poll_timeout: 21600      # resume 轮询最长 6 小时
        request_interval: 60     # 段间提交间隔，缓解 ExceedConcurrencyLimit
        retry_on_concurrency: 5
        fallback_to_image: false
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

from src.imagegen.jimeng_cli_backend import _normalize_model_version
from src.videogen.video_generator import VideoGenerator, VideoResult
from src.video.jimeng_intro_compositor import (
    _ensure_command_available,
    _find_video_paths,
    _newest_video_in_dir,
    _parse_cli_json,
    _run_cli,
)

log = logging.getLogger("novel")

CliFlavor = Literal["dreamina", "jimeng"]
_POLL_INTERVAL = 5.0
_DREAMINA_MIN_DURATION = 4
_DREAMINA_MAX_DURATION = 15
_CONCURRENCY_MARKERS = ("ExceedConcurrencyLimit", "ret=1310", "1310")


def _clamp_dreamina_duration(duration: int) -> int:
    return max(_DREAMINA_MIN_DURATION, min(_DREAMINA_MAX_DURATION, duration))


def _ratio_from_config(config: dict) -> str:
    ratio = config.get("aspect_ratio") or config.get("ratio")
    if ratio:
        return str(ratio).strip()
    return "16:9"


def _video_resolution_from_config(config: dict) -> str:
    res = config.get("video_resolution") or config.get("resolution") or "720p"
    res = str(res).strip().lower()
    if res in ("2k", "1080p"):
        return "1080p"
    return "720p"


def is_concurrency_limit_error(message: str) -> bool:
    text = message or ""
    return any(marker in text for marker in _CONCURRENCY_MARKERS)


def merge_jimeng_cli_videogen_config(videogen: dict, imagegen: dict | None) -> dict:
    """未显式设置的 CLI 字段可从 imagegen.jimeng-cli 继承（不含 timeout）。"""
    merged = dict(videogen)
    ig = imagegen or {}
    if merged.get("backend") != "jimeng-cli":
        return merged
    for key in (
        "cli_flavor",
        "cli_command",
        "region",
        "ratio",
        "aspect_ratio",
        "video_resolution",
        "output_dir",
        "extra_args",
        "request_interval",
    ):
        if key not in merged or merged[key] in (None, ""):
            if key in ig and ig[key] not in (None, ""):
                merged[key] = ig[key]
    # 视频轮询超时单独配置，禁止继承 imagegen.timeout=300
    if not merged.get("poll_timeout"):
        merged.setdefault("poll_timeout", 21600)
    return merged


class JimengCliVideoBackend(VideoGenerator):
    """通过本地即梦 CLI 生成视频片段。"""

    def __init__(self, config: dict) -> None:
        flavor = config.get("cli_flavor", "dreamina")
        if flavor not in ("dreamina", "jimeng"):
            raise ValueError(f"未知 cli_flavor: {flavor!r}，可选 dreamina | jimeng")
        self._flavor: CliFlavor = flavor
        default_cmd = "dreamina" if flavor == "dreamina" else "jimeng"
        self._command = config.get("cli_command") or default_cmd
        self._ratio = _ratio_from_config(config)
        self._video_resolution = _video_resolution_from_config(config)
        if flavor == "dreamina":
            default_model = "seedance2.0fast"
            raw_model = config.get("model_version") or config.get("model") or default_model
            self._model = _normalize_model_version(str(raw_model))
        else:
            self._model = str(
                config.get("model")
                or config.get("model_version")
                or "jimeng-video-seedance-2.0-fast"
            )
        self._region = config.get("region", "cn")
        self._default_duration = int(config.get("duration", 5))
        self._use_image_as_first_frame = bool(
            config.get("use_image_as_first_frame", True)
        )
        self._output_dir = config.get("output_dir", "")
        self._extra_args: list[str] = list(config.get("extra_args") or [])
        self._request_interval = float(config.get("request_interval", 30.0))
        self._poll_timeout = float(config.get("poll_timeout", 21600))
        self._async_submit = bool(config.get("async_submit", False))
        self._retry_on_concurrency = int(config.get("retry_on_concurrency", 5))
        self._concurrency_retry_wait = float(
            config.get("concurrency_retry_wait", 120.0)
        )
        if self._async_submit:
            self._poll = 0
        else:
            self._poll = int(config.get("poll", 0))
        self._last_request_at: float | None = None
        log.info(
            "即梦 CLI 视频后端: flavor=%s, command=%s, model=%s, ratio=%s, "
            "resolution=%s, async=%s, poll_timeout=%ss",
            self._flavor,
            self._command,
            self._model,
            self._ratio,
            self._video_resolution,
            self._async_submit,
            int(self._poll_timeout),
        )

    def _throttle(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait = self._request_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def _resolve_output_dir(self, hint: Path | None = None) -> Path:
        if hint is not None:
            hint.mkdir(parents=True, exist_ok=True)
            return hint
        if self._output_dir:
            out = Path(self._output_dir)
            out.mkdir(parents=True, exist_ok=True)
            return out
        return Path(tempfile.mkdtemp(prefix="jimeng_cli_video_"))

    def _subprocess_timeout(self) -> float:
        if self._async_submit or self._poll == 0:
            return min(600.0, self._poll_timeout)
        return self._poll_timeout + 120.0

    def _dreamina_poll_arg(self) -> int:
        if self._async_submit:
            return 0
        if self._poll > 0:
            return self._poll
        return int(min(self._poll_timeout, 86400))

    def poll_submit_id(
        self, submit_id: str, output_dir: Path, *, duration_hint: float = 5.0
    ) -> VideoResult:
        """轮询已有 submit_id 直至出片或超时（resume 用）。"""
        _ensure_command_available(self._command)
        data = self._poll_dreamina_result(submit_id, output_dir)
        video_path = self._resolve_output_video(data, output_dir)
        meta = self._probe_video(video_path, duration_hint)
        return VideoResult(
            video_path=video_path,
            duration=meta["duration"],
            width=meta["width"],
            height=meta["height"],
            submit_id=submit_id,
            pending=False,
        )

    def _poll_dreamina_result(self, submit_id: str, output_dir: Path) -> dict:
        deadline = time.monotonic() + self._poll_timeout
        last_data: dict = {}

        while time.monotonic() < deadline:
            result = _run_cli(
                [
                    self._command,
                    "query_result",
                    f"--submit_id={submit_id}",
                    f"--download_dir={output_dir}",
                ],
                min(600.0, self._poll_timeout),
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()[:500]
                if is_concurrency_limit_error(detail):
                    log.warning(
                        "query_result 并发限制，%ds 后重试 submit_id=%s",
                        int(self._concurrency_retry_wait),
                        submit_id,
                    )
                    time.sleep(self._concurrency_retry_wait)
                    continue
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
                raise RuntimeError(f"即梦视频生成失败: {reason}")

            log.info(
                "视频排队/生成中: submit_id=%s status=%s",
                submit_id,
                status or "unknown",
            )
            time.sleep(_POLL_INTERVAL)

        raise RuntimeError(
            f"即梦视频轮询超时 ({int(self._poll_timeout)}s, submit_id={submit_id}, "
            f"last_status={last_data.get('gen_status')})"
        )

    def _resolve_output_video(self, data: dict, output_dir: Path) -> Path:
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

        newest = _newest_video_in_dir(output_dir)
        if newest is not None:
            return newest

        raise RuntimeError("即梦 CLI 响应中无可用视频文件")

    def _resolve_jimeng_video_path(self, stdout: str, output_dir: Path) -> Path:
        text = stdout.strip()
        if text:
            try:
                data = json.loads(text)
                for candidate in _find_video_paths(data):
                    path = Path(candidate)
                    if path.is_file():
                        return path
            except json.JSONDecodeError:
                pass

        newest = _newest_video_in_dir(output_dir)
        if newest is not None:
            return newest

        raise RuntimeError(f"即梦 CLI 未产出视频文件 (output_dir={output_dir})")

    def _finalize_dreamina(
        self, stdout: str, output_dir: Path, *, duration_hint: float
    ) -> VideoResult:
        data = _parse_cli_json(stdout)
        status = data.get("gen_status")
        submit_id = str(data.get("submit_id") or "")

        if status == "querying":
            if not submit_id:
                raise RuntimeError("即梦 CLI 返回 querying 但缺少 submit_id")
            if self._async_submit:
                log.info("视频已异步提交: submit_id=%s", submit_id)
                return VideoResult(
                    video_path=output_dir / f"pending_{submit_id}.mp4",
                    duration=duration_hint,
                    width=0,
                    height=0,
                    submit_id=submit_id,
                    pending=True,
                )
            data = self._poll_dreamina_result(submit_id, output_dir)
        elif status == "fail":
            reason = data.get("fail_reason") or "未知原因"
            raise RuntimeError(f"即梦视频生成失败: {reason}")
        elif status != "success":
            raise RuntimeError(f"即梦 CLI 返回未知状态: {status!r}")

        video_path = self._resolve_output_video(data, output_dir)
        meta = self._probe_video(video_path, duration_hint)
        return VideoResult(
            video_path=video_path,
            duration=meta["duration"],
            width=meta["width"],
            height=meta["height"],
            submit_id=submit_id,
            pending=False,
        )

    def _build_dreamina_text2video_command(self, prompt: str, duration: int) -> list[str]:
        cmd = [
            self._command,
            "text2video",
            f"--prompt={prompt}",
            f"--duration={duration}",
            f"--ratio={self._ratio}",
            f"--model_version={self._model}",
            f"--video_resolution={self._video_resolution}",
            f"--poll={self._dreamina_poll_arg()}",
        ]
        cmd.extend(self._extra_args)
        return cmd

    def _build_dreamina_image2video_command(
        self, prompt: str, image_path: Path, duration: int
    ) -> list[str]:
        cmd = [
            self._command,
            "image2video",
            f"--image={image_path}",
            f"--prompt={prompt}",
            f"--duration={duration}",
            f"--model_version={self._model}",
            f"--video_resolution={self._video_resolution}",
            f"--poll={self._dreamina_poll_arg()}",
        ]
        cmd.extend(self._extra_args)
        return cmd

    def _build_dreamina_multimodal_command(
        self,
        prompt: str,
        *,
        image_paths: list[Path],
        video_paths: list[Path],
        audio_paths: list[Path] | None,
        duration: int,
    ) -> list[str]:
        if not image_paths and not video_paths:
            raise ValueError("multimodal2video 至少需要 --image 或 --video")
        cmd = [
            self._command,
            "multimodal2video",
            f"--prompt={prompt}",
            f"--duration={duration}",
            f"--ratio={self._ratio}",
            f"--model_version={self._model}",
            f"--video_resolution={self._video_resolution}",
            f"--poll={self._dreamina_poll_arg()}",
        ]
        for path in image_paths:
            cmd.extend(["--image", str(path)])
        for path in video_paths:
            cmd.extend(["--video", str(path)])
        for path in audio_paths or []:
            cmd.extend(["--audio", str(path)])
        cmd.extend(self._extra_args)
        return cmd

    def generate_multimodal(
        self,
        prompt: str,
        *,
        image_paths: list[Path] | None = None,
        video_paths: list[Path] | None = None,
        audio_paths: list[Path] | None = None,
        duration: float = 5.0,
        output_dir: Path | None = None,
    ) -> VideoResult:
        """dreamina multimodal2video：图片 + 视频 (+ 可选音频) 全能参考生视频。"""
        if self._flavor != "dreamina":
            raise RuntimeError(
                "video2video 角色替换需 dreamina CLI（multimodal2video），"
                f"当前 cli_flavor={self._flavor!r}"
            )
        _ensure_command_available(self._command)
        self._throttle()
        out_dir = self._resolve_output_dir(output_dir)
        images = [Path(p) for p in (image_paths or []) if Path(p).is_file()]
        videos = [Path(p) for p in (video_paths or []) if Path(p).is_file()]
        audios = [Path(p) for p in (audio_paths or []) if Path(p).is_file()]
        if not images and not videos:
            raise ValueError("multimodal2video 缺少有效的 image 或 video 输入")
        clip_duration = _clamp_dreamina_duration(int(duration or self._default_duration))
        cmd = self._build_dreamina_multimodal_command(
            prompt,
            image_paths=images,
            video_paths=videos,
            audio_paths=audios or None,
            duration=clip_duration,
        )
        result = self._run_with_concurrency_retry(cmd)
        gen_result = self._finalize_dreamina(
            result.stdout, out_dir, duration_hint=float(clip_duration)
        )
        self._last_request_at = time.monotonic()
        if not gen_result.pending:
            log.info(
                "即梦 multimodal2video 完成: %s (submit_id=%s)",
                gen_result.video_path.name,
                gen_result.submit_id,
            )
        return gen_result

    def _run_with_concurrency_retry(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        last_detail = ""
        for attempt in range(self._retry_on_concurrency + 1):
            result = _run_cli(cmd, self._subprocess_timeout())
            if result.returncode == 0:
                return result
            detail = (result.stderr or result.stdout or "").strip()
            last_detail = detail[:800]
            if is_concurrency_limit_error(detail) and attempt < self._retry_on_concurrency:
                wait = self._concurrency_retry_wait * (attempt + 1)
                log.warning(
                    "即梦并发限制 (1310)，%ds 后重试 (%d/%d)",
                    int(wait),
                    attempt + 1,
                    self._retry_on_concurrency,
                )
                time.sleep(wait)
                continue
            break
        raise RuntimeError(
            f"即梦 CLI 视频生成失败 (exit={result.returncode}): {last_detail}"
        )

    def _generate_dreamina(
        self,
        prompt: str,
        image_path: Path | None,
        duration: int,
        output_dir: Path,
    ) -> VideoResult:
        use_i2v = (
            image_path is not None
            and image_path.is_file()
            and self._use_image_as_first_frame
        )
        if use_i2v:
            cmd = self._build_dreamina_image2video_command(prompt, image_path, duration)
        else:
            cmd = self._build_dreamina_text2video_command(prompt, duration)

        result = self._run_with_concurrency_retry(cmd)
        return self._finalize_dreamina(
            result.stdout, output_dir, duration_hint=float(duration)
        )

    def _build_jimeng_command(
        self,
        prompt: str,
        image_path: Path | None,
        duration: int,
        output_dir: Path,
    ) -> list[str]:
        use_i2v = (
            image_path is not None
            and image_path.is_file()
            and self._use_image_as_first_frame
        )
        mode = "image_to_video" if use_i2v else "text_to_video"
        cmd = [
            self._command,
            "video",
            "generate",
            "--prompt",
            prompt,
            "--mode",
            mode,
            "--model",
            self._model,
            "--ratio",
            self._ratio,
            "--resolution",
            self._video_resolution,
            "--duration",
            str(duration),
            "--region",
            self._region,
            "--output-dir",
            str(output_dir),
            "--json",
        ]
        if self._async_submit:
            cmd.extend(["--no-wait"])
        else:
            cmd.extend(["--wait"])
        if use_i2v and image_path is not None:
            cmd.extend(["--image-file", str(image_path)])
        cmd.extend(self._extra_args)
        return cmd

    def _generate_jimeng(
        self,
        prompt: str,
        image_path: Path | None,
        duration: int,
        output_dir: Path,
    ) -> VideoResult:
        result = self._run_with_concurrency_retry(
            self._build_jimeng_command(prompt, image_path, duration, output_dir)
        )
        if self._async_submit:
            data = _parse_cli_json(result.stdout)
            submit_id = str(data.get("submit_id") or data.get("task_id") or "")
            if submit_id:
                return VideoResult(
                    video_path=output_dir / f"pending_{submit_id}.mp4",
                    duration=float(duration),
                    width=0,
                    height=0,
                    submit_id=submit_id,
                    pending=True,
                )
        video_path = self._resolve_jimeng_video_path(result.stdout, output_dir)
        meta = self._probe_video(video_path, float(duration))
        return VideoResult(
            video_path=video_path,
            duration=meta["duration"],
            width=meta["width"],
            height=meta["height"],
            pending=False,
        )

    def _probe_video(self, video_path: Path, fallback_duration: float) -> dict:
        if not video_path.is_file():
            return {"width": 0, "height": 0, "duration": fallback_duration}
        try:
            from src.utils.ffmpeg_helper import ensure_ffmpeg, get_ffprobe_path

            ensure_ffmpeg()
            cmd = [
                get_ffprobe_path(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                streams = data.get("streams") or [{}]
                fmt = data.get("format") or {}
                width = int(streams[0].get("width") or 0)
                height = int(streams[0].get("height") or 0)
                duration = float(fmt.get("duration") or fallback_duration)
                return {"width": width, "height": height, "duration": duration}
        except Exception as exc:
            log.debug("ffprobe 读取视频元信息失败: %s", exc)
        return {"width": 0, "height": 0, "duration": fallback_duration}

    def generate(
        self,
        prompt: str,
        image_path: Path | None = None,
        duration: float = 5.0,
        *,
        output_dir: Path | None = None,
    ) -> VideoResult:
        _ensure_command_available(self._command)
        self._throttle()
        out_dir = self._resolve_output_dir(output_dir)
        requested = int(duration or self._default_duration)
        if self._flavor == "dreamina":
            clip_duration = _clamp_dreamina_duration(requested)
        else:
            clip_duration = max(2, min(15, requested))

        if self._flavor == "dreamina":
            result = self._generate_dreamina(prompt, image_path, clip_duration, out_dir)
        else:
            result = self._generate_jimeng(prompt, image_path, clip_duration, out_dir)

        self._last_request_at = time.monotonic()
        if not result.pending:
            log.info(
                "即梦 CLI 视频生成完成: %s (%dx%d, %.1fs)",
                result.video_path.name,
                result.width,
                result.height,
                result.duration,
            )
        return result
