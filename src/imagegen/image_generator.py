"""图片生成抽象接口与工厂函数。

定义 ImageGenerator 基类，所有图片生成后端均需实现 generate() 方法。
通过 create_image_generator() 工厂函数根据配置实例化具体后端。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

log = logging.getLogger("novel")

# Web UI / config.yaml 深合并时，backend 可能被覆盖而 model 仍留在其他后端的值。
_BACKEND_DEFAULTS: dict[str, dict] = {
    "siliconflow": {"model": "black-forest-labs/FLUX.1-schnell"},
    "dashscope": {
        "model": "qwen-image-2.0-pro-2026-04-22",
        "size": "928*1664",
    },
    "together": {"model": "black-forest-labs/FLUX.1-schnell-Free"},
    "jimeng-cli": {
        "model": "4.5",
        "cli_flavor": "dreamina",
        "cli_command": "dreamina",
        "ratio": "9:16",
        "resolution": "2k",
    },
    "comfyui": {
        "model": "",
        "base_url": "http://127.0.0.1:8188",
        "workflow": "workflows/comfyui/fluxscale_workflow.json",
    },
}

_BACKEND_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "dashscope": ("wan", "qwen-image"),
    "siliconflow": ("black-forest-labs/", "stabilityai/", "Pro/", "Kwai-Kolors/"),
    "together": ("black-forest-labs/", "stabilityai/"),
    "jimeng-cli": ("jimeng-", "nanobanana", "3.", "4.", "5."),
}


def _resolve_imagegen_config(config: dict) -> dict:
    """按 backend 校正 model/size，避免跨后端配置残留导致 API 400。"""
    backend = config.get("backend", "diffusers")
    if backend not in _BACKEND_DEFAULTS:
        return dict(config)

    resolved = dict(config)
    defaults = _BACKEND_DEFAULTS[backend]

    # YAML 中 model: 4.5 会被解析为 float，统一转为 str
    raw_model = resolved.get("model", "")
    if raw_model is not None and raw_model != "":
        resolved["model"] = str(raw_model).strip()
    else:
        resolved["model"] = ""

    model = resolved["model"]
    prefixes = _BACKEND_MODEL_PREFIXES.get(backend, ())

    if model and prefixes and not any(model.startswith(p) for p in prefixes):
        log.warning(
            "imagegen.model=%r 与 backend=%r 不匹配，改用 %r",
            model,
            backend,
            defaults["model"],
        )
        resolved["model"] = defaults["model"]
    elif not model:
        resolved["model"] = defaults["model"]

    if backend == "dashscope" and "size" not in resolved:
        resolved["size"] = defaults.get("size", "928*1664")
    elif backend == "jimeng-cli":
        if "ratio" not in resolved:
            resolved["ratio"] = defaults.get("ratio", "9:16")
        else:
            resolved["ratio"] = str(resolved["ratio"]).strip()
        if "resolution" not in resolved:
            resolved["resolution"] = defaults.get("resolution", "2k")
        else:
            resolved["resolution"] = str(resolved["resolution"]).strip()
        if "cli_flavor" not in resolved:
            resolved["cli_flavor"] = defaults.get("cli_flavor", "dreamina")
        if "cli_command" not in resolved:
            resolved["cli_command"] = defaults.get("cli_command", "dreamina")
    elif backend == "comfyui":
        if not resolved.get("workflow"):
            resolved["workflow"] = defaults.get("workflow", "")
        if not resolved.get("base_url"):
            resolved["base_url"] = defaults.get("base_url", "http://127.0.0.1:8188")

    return resolved


class ImageGenerator(ABC):
    """图片生成器抽象基类。"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> Image.Image:
        """根据文本提示词生成一张图片。

        Args:
            prompt: 用于图片生成的文本提示词。
            **kwargs: 后端可选参数（如 ComfyUI 的 ``person_count``）。

        Returns:
            生成的 PIL Image 对象。
        """
        ...


def create_image_generator(config: dict) -> ImageGenerator:
    """根据配置创建图片生成器实例。

    Args:
        config: imagegen 配置字典，必须包含 backend 字段。

    Returns:
        对应后端的 ImageGenerator 实例。

    Raises:
        ValueError: 未知的后端名称。
    """
    config = _resolve_imagegen_config(config)
    backend = config.get("backend", "diffusers")
    if backend == "diffusers":
        from src.imagegen.diffusers_backend import DiffusersBackend

        return DiffusersBackend(config)
    elif backend == "together":
        from src.imagegen.together_backend import TogetherBackend

        return TogetherBackend(config)
    elif backend == "siliconflow":
        from src.imagegen.siliconflow_backend import SiliconFlowBackend

        return SiliconFlowBackend(config)
    elif backend == "dashscope":
        from src.imagegen.dashscope_backend import DashScopeBackend

        return DashScopeBackend(config)
    elif backend == "jimeng-cli":
        from src.imagegen.jimeng_cli_backend import JimengCliBackend

        return JimengCliBackend(config)
    elif backend == "comfyui":
        from src.imagegen.comfyui_backend import ComfyUIBackend

        return ComfyUIBackend(config)
    else:
        raise ValueError(f"Unknown image backend: {backend}")
