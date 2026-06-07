"""阿里云百炼 DashScope 图片生成后端（Qwen-Image / 万相）。"""

import io
import logging
import os
import time

from PIL import Image

from src.imagegen.image_generator import ImageGenerator
from src.imagegen.moderation import is_content_moderation_error

log = logging.getLogger("novel")


class ContentModerationError(RuntimeError):
    """云端生图输出被内容安全策略拦截。"""

_MAX_RETRIES = 8
_RETRY_BASE_DELAY = 5  # 429 限流时指数退避基数（秒）


def _retry_delay(attempt: int, resp=None) -> float:
    """计算重试等待时间，优先尊重 Retry-After。"""
    if resp is not None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
    return min(_RETRY_BASE_DELAY * (2 ** attempt), 60.0)


class DashScopeBackend(ImageGenerator):
    """基于阿里云 DashScope API 的图片生成后端。

    支持 Qwen-Image（qwen-image-2.0-pro 等）与万相 wan2.6-t2i。
    """

    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def __init__(self, config: dict) -> None:
        self._client = None
        self._model = config.get("model", "qwen-image-2.0-pro-2026-04-22")
        width = config.get("width", 1024)
        height = config.get("height", 1792)
        # DashScope 格式: "width*height"
        # qwen-image: 单边 512–2048，总量 512*512–2048*2048，9:16 推荐 928*1664
        # wan2.6-t2i: 总量 1280*1280–1440*1440，9:16 推荐 960*1696
        if config.get("size"):
            self._size = config["size"]
        elif self._model.startswith("qwen-image"):
            self._size = "928*1664" if height >= width else "1664*928"
        else:
            self._size = f"{width}*{height}"
        self._api_key = config.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "DashScope 需要 API Key。请设置环境变量 DASHSCOPE_API_KEY。"
            )
        # 连续请求间隔，降低 429 概率（qwen-image 批量生图时建议 >= 3）
        self._request_interval = float(config.get("request_interval", 3.0))
        self._last_request_at: float | None = None
        # prompt_extend 会扩写 prompt，敏感题材下更容易触发审核，默认关闭
        self._prompt_extend = bool(config.get("prompt_extend", False))
        self._negative_prompt = config.get(
            "negative_prompt",
            "nsfw, nude, violence, gore, blood, explicit sexual content, pornographic",
        )

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=120)
        return self._client

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def _throttle(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait = self._request_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def generate(self, prompt: str) -> Image.Image:
        """调用 DashScope API 生成图片，遇到限流/服务器错误/网络错误自动重试。"""
        import httpx

        self._throttle()
        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "input": {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [{"text": prompt}],
                                }
                            ]
                        },
                        "parameters": {
                            "size": self._size,
                            "n": 1,
                            "prompt_extend": self._prompt_extend,
                            "watermark": False,
                            "negative_prompt": self._negative_prompt,
                        },
                    },
                )
            except httpx.RequestError as exc:
                last_exc = exc
                delay = _retry_delay(attempt)
                log.warning(
                    "DashScope 网络错误，%.0fs 后重试 (%d/%d): %s",
                    delay, attempt + 1, _MAX_RETRIES, exc,
                )
                time.sleep(delay)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
                delay = _retry_delay(attempt, resp)
                log.warning(
                    "DashScope HTTP %d，%.0fs 后重试 (%d/%d)",
                    resp.status_code, delay, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(delay)
                continue

            if resp.status_code >= 400:
                detail = resp.text[:500]
                if is_content_moderation_error(detail):
                    log.warning(
                        "DashScope 内容审核拦截 (model=%s): %s",
                        self._model,
                        detail,
                    )
                    raise ContentModerationError(detail)
                log.error(
                    "DashScope HTTP %d (model=%s, size=%s): %s",
                    resp.status_code,
                    self._model,
                    self._size,
                    detail,
                )
                resp.raise_for_status()
            data = resp.json()

            # 从响应中提取图片 URL
            image_url = data["output"]["choices"][0]["message"]["content"][0]["image"]

            # 下载图片（URL 24小时有效）
            img_resp = client.get(image_url)
            img_resp.raise_for_status()
            image = Image.open(io.BytesIO(img_resp.content))

            self._last_request_at = time.monotonic()
            log.debug("DashScope 生成图片: %dx%d", image.width, image.height)
            return image

        raise RuntimeError(
            f"DashScope 图片生成失败 ({_MAX_RETRIES}次重试): {last_exc}"
        ) from last_exc
