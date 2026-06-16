"""即梦 CLI 本地图片生成后端。

通过 subprocess 调用本地即梦命令行工具，复用已登录账号积分生图，无需云端 API Key。

支持两种 CLI：

1. **官方 Dreamina CLI**（推荐）::

       curl -s https://jimeng.jianying.com/cli | bash
       dreamina login

2. **社区 jimeng-cli**（npm）::

       npm install -g jimeng-cli
       jimeng login

典型配置 (config.yaml)::

    imagegen:
      backend: jimeng-cli
      cli_flavor: dreamina      # dreamina | jimeng
      cli_command: dreamina     # 留空则按 cli_flavor 自动选择
      model: 4.5                # dreamina 用 model_version；jimeng 可用 jimeng-4.5
      ratio: "9:16"
      resolution: 2k            # dreamina 映射为 resolution_type
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

from PIL import Image

from src.imagegen.image_generator import ImageGenerator

log = logging.getLogger("novel")

CliFlavor = Literal["dreamina", "jimeng"]
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_POLL_INTERVAL = 2.0


def _ratio_from_size(width: int, height: int) -> str:
    """根据宽高推导 CLI 支持的 ratio 字符串。"""
    if width <= 0 or height <= 0:
        return "9:16"
    if width == height:
        return "1:1"
    return "9:16" if height > width else "16:9"


def _normalize_model_version(model: str) -> str:
    """dreamina 使用 4.5 形式，兼容 jimeng-4.5 配置。"""
    model = model.strip()
    if model.startswith("jimeng-"):
        return model[len("jimeng-") :]
    return model


def _parse_cli_json(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise RuntimeError("即梦 CLI 无输出")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"即梦 CLI 返回非 JSON: {text[:200]}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"即梦 CLI 返回异常结构: {type(data).__name__}")
    return data


def _find_image_paths(obj: object) -> list[str]:
    """从 CLI JSON 输出中递归提取可能的本地图片路径。"""
    found: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value not in seen:
            seen.add(value)
            found.append(value)

    def walk(value: object) -> None:
        if isinstance(value, str):
            lower = value.lower()
            if any(lower.endswith(ext) for ext in _IMAGE_SUFFIXES):
                add(value)
            return
        if isinstance(value, dict):
            for key in ("path", "file", "filePath", "output", "image", "imagePath"):
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


def _newest_image_in_dir(directory: Path) -> Path | None:
    """返回目录中最新的图片文件。"""
    candidates = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _download_image_url(url: str) -> Image.Image:
    import httpx

    resp = httpx.get(url, timeout=120)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class JimengGenerationError(RuntimeError):
    """即梦 CLI 生图失败（含内容审核 / 平台拒稿）。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"即梦生图失败: {reason}")


class JimengCliBackend(ImageGenerator):
    """通过本地即梦 CLI 生成图片。"""

    def __init__(self, config: dict) -> None:
        flavor = config.get("cli_flavor", "dreamina")
        if flavor not in ("dreamina", "jimeng"):
            raise ValueError(f"未知 cli_flavor: {flavor!r}，可选 dreamina | jimeng")
        self._flavor: CliFlavor = flavor
        default_cmd = "dreamina" if flavor == "dreamina" else "jimeng"
        self._command = config.get("cli_command") or default_cmd
        raw_model = config.get("model", "4.5" if flavor == "dreamina" else "jimeng-4.5")
        self._model = str(raw_model).strip()
        width = int(config.get("width", 1024))
        height = int(config.get("height", 1792))
        ratio = config.get("ratio")
        self._ratio = str(ratio).strip() if ratio else _ratio_from_size(width, height)
        self._resolution = config.get("resolution", "2k")
        self._region = config.get("region", "cn")
        self._negative_prompt = config.get("negative_prompt", "")
        self._output_dir = config.get("output_dir", "")
        self._extra_args: list[str] = list(config.get("extra_args") or [])
        self._request_interval = float(config.get("request_interval", 5.0))
        self._timeout = float(config.get("timeout", 300))
        self._prompt_max_chars = int(
            config.get("prompt_max_chars", 0) or 0
        )
        self._last_request_at: float | None = None
        log.info(
            "即梦 CLI 后端: flavor=%s, command=%s, model=%s, ratio=%s, resolution=%s",
            self._flavor,
            self._command,
            self._model,
            self._ratio,
            self._resolution,
        )

    def _throttle(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait = self._request_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def _resolve_output_dir(self) -> Path:
        if self._output_dir:
            out = Path(self._output_dir)
            out.mkdir(parents=True, exist_ok=True)
            return out
        return Path(tempfile.mkdtemp(prefix="jimeng_cli_"))

    def _run_cli(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        log.debug("即梦 CLI: %s", " ".join(cmd[:8]) + (" ..." if len(cmd) > 8 else ""))
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"即梦 CLI 生图超时 ({self._timeout}s)") from exc

    def _ensure_command_available(self) -> None:
        if shutil.which(self._command):
            return
        if self._flavor == "dreamina":
            hint = (
                "请先安装官方 CLI: curl -s https://jimeng.jianying.com/cli | bash，"
                "并执行 dreamina login"
            )
        else:
            hint = "请先安装: npm install -g jimeng-cli，并执行 jimeng login"
        raise RuntimeError(f"未找到即梦 CLI 命令 {self._command!r}。{hint}")

    def _build_jimeng_command(self, prompt: str, output_dir: Path) -> list[str]:
        cmd = [
            self._command,
            "image",
            "generate",
            "--prompt",
            prompt,
            "--model",
            self._model,
            "--ratio",
            self._ratio,
            "--resolution",
            self._resolution,
            "--region",
            self._region,
            "--output-dir",
            str(output_dir),
            "--wait",
            "--json",
        ]
        if self._negative_prompt:
            cmd.extend(["--negative-prompt", self._negative_prompt])
        cmd.extend(self._extra_args)
        return cmd

    def _build_dreamina_command(self, prompt: str) -> list[str]:
        cmd = [
            self._command,
            "text2image",
            f"--prompt={prompt}",
            f"--ratio={self._ratio}",
            f"--resolution_type={self._resolution}",
            f"--model_version={_normalize_model_version(self._model)}",
            f"--poll={int(self._timeout)}",
        ]
        cmd.extend(self._extra_args)
        return cmd

    def _resolve_jimeng_image_path(self, stdout: str, output_dir: Path) -> Path:
        text = stdout.strip()
        if text:
            try:
                data = json.loads(text)
                for candidate in _find_image_paths(data):
                    path = Path(candidate)
                    if path.is_file():
                        return path
            except json.JSONDecodeError:
                pass

        newest = _newest_image_in_dir(output_dir)
        if newest is not None:
            return newest

        raise RuntimeError(f"即梦 CLI 未产出图片文件 (output_dir={output_dir})")

    def _poll_dreamina_result(self, submit_id: str, output_dir: Path) -> dict:
        deadline = time.monotonic() + self._timeout
        last_data: dict = {}

        while time.monotonic() < deadline:
            result = self._run_cli(
                [
                    self._command,
                    "query_result",
                    f"--submit_id={submit_id}",
                    f"--download_dir={output_dir}",
                ]
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
                raise JimengGenerationError(reason)

            time.sleep(_POLL_INTERVAL)

        raise RuntimeError(
            f"即梦生图轮询超时 ({self._timeout}s, submit_id={submit_id}, "
            f"last_status={last_data.get('gen_status')})"
        )

    def _image_from_dreamina_result(self, data: dict, output_dir: Path) -> Image.Image:
        images = (data.get("result_json") or {}).get("images") or []
        if images:
            first = images[0]
            path_value = first.get("path")
            if path_value:
                path = Path(path_value)
                if path.is_file():
                    image = Image.open(path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return image
            url = first.get("image_url")
            if url:
                return _download_image_url(url)

        for candidate in _find_image_paths(data):
            path = Path(candidate)
            if path.is_file():
                image = Image.open(path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return image

        newest = _newest_image_in_dir(output_dir)
        if newest is not None:
            image = Image.open(newest)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image

        raise RuntimeError("即梦 CLI 响应中无可用图片")

    def _generate_dreamina(self, prompt: str, output_dir: Path) -> Image.Image:
        result = self._run_cli(self._build_dreamina_command(prompt))
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[:500]
            raise RuntimeError(
                f"即梦 CLI 生图失败 (exit={result.returncode}): {detail}"
            )

        data = _parse_cli_json(result.stdout)
        status = data.get("gen_status")
        if status == "fail":
            reason = data.get("fail_reason") or "未知原因"
            raise JimengGenerationError(reason)
        if status == "querying":
            submit_id = data.get("submit_id")
            if not submit_id:
                raise RuntimeError("即梦 CLI 返回 querying 但缺少 submit_id")
            data = self._poll_dreamina_result(submit_id, output_dir)
        elif status != "success":
            raise RuntimeError(f"即梦 CLI 返回未知状态: {status!r}")

        return self._image_from_dreamina_result(data, output_dir)

    def _generate_jimeng(self, prompt: str, output_dir: Path) -> Image.Image:
        result = self._run_cli(self._build_jimeng_command(prompt, output_dir))
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[:500]
            raise RuntimeError(
                f"即梦 CLI 生图失败 (exit={result.returncode}): {detail}"
            )

        image_path = self._resolve_jimeng_image_path(result.stdout, output_dir)
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _prepare_prompt(self, prompt: str) -> str:
        from src.imagegen.moderation import (
            JIMENG_PROMPT_MAX_CHARS,
            truncate_image_prompt_for_jimeng,
        )

        limit = self._prompt_max_chars or JIMENG_PROMPT_MAX_CHARS
        prepared = truncate_image_prompt_for_jimeng(prompt, max_chars=limit)
        if len(prepared) < len(prompt):
            log.info(
                "即梦 prompt 已截断: %d -> %d 字符",
                len(prompt),
                len(prepared),
            )
        return prepared

    def generate(self, prompt: str) -> Image.Image:
        self._ensure_command_available()
        self._throttle()
        output_dir = self._resolve_output_dir()
        prompt = self._prepare_prompt(prompt)

        if self._flavor == "dreamina":
            image = self._generate_dreamina(prompt, output_dir)
        else:
            image = self._generate_jimeng(prompt, output_dir)

        self._last_request_at = time.monotonic()
        log.info("即梦 CLI 生图完成 (%dx%d)", image.width, image.height)
        return image
