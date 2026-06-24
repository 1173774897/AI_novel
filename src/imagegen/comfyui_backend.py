"""ComfyUI 本地图片生成后端。

通过 ComfyUI HTTP API（默认 ``http://127.0.0.1:8188``）提交 workflow，
轮询 ``/history/{prompt_id}`` 并下载 ``/view`` 输出图片。

使用前在 ComfyUI 中导出 API 格式 workflow（Settings → Enable Dev mode → Save API Format）。

典型配置 (config.yaml → imagegen，所有流水线模式共用)::

    imagegen:
      backend: comfyui
      base_url: http://127.0.0.1:8188
      workflow: workflows/comfyui/flux_dev_t2i_api.json
      width: 1024
      height: 1792
      steps: 20
      guidance_scale: 3.5
      node_map:
        positive: "6"
        negative: "7"
        latent: "5"
        sampler: "3"
        guidance: "11"   # FluxGuidance 节点（可选）
"""

from __future__ import annotations

import copy
import io
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image

from src.imagegen.image_generator import ImageGenerator

log = logging.getLogger("novel")

_DEFAULT_BASE_URL = "http://127.0.0.1:8188"
_SAMPLER_NODE_TYPES = frozenset({"KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"})
_GUIDANCE_NODE_TYPES = frozenset({"FluxGuidance", "CFGGuider"})
_SCHEDULER_NODE_TYPES = frozenset({"BasicScheduler", "KSamplerScheduler"})


class ComfyUIError(RuntimeError):
    """ComfyUI 调用失败。"""


def estimate_person_count_from_prompt(prompt: str) -> int:
    """从英文/中文 prompt 粗估画面人数（无分段文本时的兜底）。"""
    if not prompt.strip():
        return 0

    text = prompt.lower()
    total = 0

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    for word, count in number_words.items():
        if re.search(
            rf"\b{word}\s+(young\s+)?(women|men|people|persons|characters|girls|boys)\b",
            text,
        ):
            total = max(total, count)

    for match in re.finditer(r"\b(\d+)\s+(people|persons|characters|men|women)\b", text):
        total = max(total, int(match.group(1)))

    individuals = len(
        re.findall(
            r"\b(a|an)\s+(young\s+)?(woman|man|girl|boy|person|student)\b",
            text,
        )
    )
    if individuals:
        total = max(total, individuals)

    if "【本段相关角色" in prompt:
        cast_lines = [
            line.strip()
            for line in prompt.splitlines()
            if line.strip() and "：" in line and not line.strip().startswith("【")
        ]
        if cast_lines:
            total = max(total, len(cast_lines))

    return total


def prepend_lora_trigger(prompt: str, trigger: str) -> str:
    """将 LoRA 唤醒词前缀到 positive prompt（已存在则跳过）。"""
    trigger = trigger.strip().rstrip(",").strip()
    if not trigger:
        return prompt.strip()
    body = prompt.strip()
    if not body:
        return trigger
    first_segment = body.split(",", 1)[0].strip().lower()
    trigger_lower = trigger.lower()
    if first_segment == trigger_lower or first_segment.startswith(trigger_lower):
        return body
    return f"{trigger}, {body}"


def prepend_prompt_prefix(prompt: str, prefix: str) -> str:
    """将画风/风格前缀插入 positive prompt（已存在则跳过）。"""
    prefix = prefix.strip().rstrip(",").strip()
    if not prefix:
        return prompt.strip()
    body = prompt.strip()
    if not body:
        return prefix
    prefix_lower = prefix.lower()
    body_lower = body.lower()
    if body_lower.startswith(prefix_lower):
        return body
    if "," in body:
        first, rest = body.split(",", 1)
        if rest.strip().lower().startswith(prefix_lower):
            return body
    return f"{prefix}, {body}"


def finalize_comfyui_positive_prompt(
    prompt: str,
    *,
    lora_trigger: str = "",
    prompt_prefix: str = "",
) -> str:
    """ComfyUI positive：先画风前缀，再 LoRA 唤醒词。"""
    out = prompt.strip()
    if prompt_prefix:
        out = prepend_prompt_prefix(out, prompt_prefix)
    if lora_trigger:
        out = prepend_lora_trigger(out, lora_trigger)
    return out


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_workflow_path(workflow: str | Path) -> Path:
    """解析 workflow 路径：绝对路径优先，否则 cwd → 项目根。"""
    path = Path(workflow).expanduser()
    if path.is_absolute() and path.exists():
        return path

    candidates = [Path.cwd() / path, _project_root() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"ComfyUI workflow 不存在: {workflow}（已尝试: {tried}）")


def load_workflow(path: str | Path) -> dict[str, Any]:
    resolved = resolve_workflow_path(path)
    with open(resolved, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"ComfyUI workflow 必须是 JSON 对象: {resolved}")
    return data


def render_template(value: Any, context: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    rendered = value
    for key, raw in context.items():
        rendered = rendered.replace("{" + key + "}", str(raw))
    return rendered


def set_workflow_path(workflow: dict[str, Any], dotted_path: str, value: Any) -> None:
    """按 ``node_id.inputs.field`` 写入 workflow。"""
    parts = dotted_path.split(".")
    if len(parts) < 2:
        raise ValueError(f"workflow_inputs 路径无效: {dotted_path!r}")

    node_id, *rest = parts
    if node_id not in workflow:
        raise KeyError(f"workflow 中不存在节点 {node_id!r}")

    cur: Any = workflow[node_id]
    for part in rest[:-1]:
        if part not in cur:
            cur[part] = {}
        cur = cur[part]

    field = rest[-1]
    original = cur.get(field) if isinstance(cur, dict) else None
    if isinstance(original, bool):
        if isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
    elif isinstance(original, int) and isinstance(value, str) and value.isdigit():
        value = int(value)
    elif isinstance(original, float) and isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    cur[field] = value


def infer_node_map(workflow: dict[str, Any]) -> dict[str, str]:
    """从常见节点类型推断 positive/negative/latent/sampler 映射。"""
    clip_nodes = sorted(
        (
            node_id
            for node_id, node in workflow.items()
            if node.get("class_type") == "CLIPTextEncode"
        ),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )
    node_map: dict[str, str] = {}
    if clip_nodes:
        node_map["positive"] = clip_nodes[0]
    if len(clip_nodes) > 1:
        node_map["negative"] = clip_nodes[1]

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if class_type == "EmptyLatentImage":
            node_map.setdefault("latent", node_id)
        elif class_type in _SAMPLER_NODE_TYPES:
            node_map.setdefault("sampler", node_id)
        elif class_type in _GUIDANCE_NODE_TYPES:
            node_map.setdefault("guidance", node_id)
        elif class_type in _SCHEDULER_NODE_TYPES:
            node_map.setdefault("scheduler", node_id)
        elif class_type == "RandomNoise":
            node_map.setdefault("noise", node_id)
    return node_map


def build_generation_context(
    prompt: str,
    *,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1792,
    steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int | None = None,
) -> dict[str, Any]:
    resolved_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "cfg": guidance_scale,
        "seed": resolved_seed,
    }


def apply_node_map(
    workflow: dict[str, Any],
    node_map: dict[str, str],
    context: dict[str, Any],
) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)

    positive_id = node_map.get("positive")
    if positive_id and positive_id in wf:
        wf[positive_id].setdefault("inputs", {})["text"] = context["prompt"]

    negative_id = node_map.get("negative")
    if negative_id and negative_id in wf and context.get("negative_prompt"):
        wf[negative_id].setdefault("inputs", {})["text"] = context["negative_prompt"]

    latent_id = node_map.get("latent")
    if latent_id and latent_id in wf:
        inputs = wf[latent_id].setdefault("inputs", {})
        inputs["width"] = context["width"]
        inputs["height"] = context["height"]

    sampler_id = node_map.get("sampler")
    if sampler_id and sampler_id in wf:
        inputs = wf[sampler_id].setdefault("inputs", {})
        inputs["seed"] = context["seed"]
        if "steps" in inputs or context.get("steps") is not None:
            inputs["steps"] = context["steps"]
        if "cfg" in inputs:
            inputs["cfg"] = context["guidance_scale"]

    guidance_id = node_map.get("guidance")
    if guidance_id and guidance_id in wf:
        inputs = wf[guidance_id].setdefault("inputs", {})
        if "guidance" in inputs:
            inputs["guidance"] = context["guidance_scale"]

    scheduler_id = node_map.get("scheduler")
    if scheduler_id and scheduler_id in wf:
        inputs = wf[scheduler_id].setdefault("inputs", {})
        if "steps" in inputs:
            inputs["steps"] = context["steps"]

    noise_id = node_map.get("noise")
    if noise_id and noise_id in wf:
        inputs = wf[noise_id].setdefault("inputs", {})
        if "noise_seed" in inputs:
            inputs["noise_seed"] = context["seed"]
        elif "seed" in inputs:
            inputs["seed"] = context["seed"]

    return wf


def apply_workflow_inputs(
    workflow: dict[str, Any],
    workflow_inputs: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)
    for path, raw_value in workflow_inputs.items():
        set_workflow_path(wf, str(path), render_template(raw_value, context))
    return wf


def prepare_workflow(
    base_workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1792,
    steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int | None = None,
    node_map: dict[str, str] | None = None,
    workflow_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = build_generation_context(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    if workflow_inputs:
        return apply_workflow_inputs(base_workflow, workflow_inputs, context)

    mapping = dict(node_map or {})
    if not mapping:
        mapping = infer_node_map(base_workflow)
    if "positive" not in mapping:
        raise ComfyUIError(
            "无法推断 positive 节点，请在 imagegen.node_map 或 workflow_inputs 中显式配置"
        )
    return apply_node_map(base_workflow, mapping, context)


def find_output_images(
    history_entry: dict[str, Any],
    *,
    output_node: str | None = None,
) -> list[dict[str, str]]:
    outputs = history_entry.get("outputs") or {}
    if not isinstance(outputs, dict):
        return []

    node_ids: list[str]
    if output_node:
        node_ids = [output_node]
    else:
        node_ids = list(outputs.keys())

    images: list[dict[str, str]] = []
    for node_id in node_ids:
        node_output = outputs.get(node_id) or {}
        for image in node_output.get("images") or []:
            if isinstance(image, dict) and image.get("filename"):
                images.append(
                    {
                        "filename": str(image["filename"]),
                        "subfolder": str(image.get("subfolder") or ""),
                        "type": str(image.get("type") or "output"),
                    }
                )
    return images


class ComfyUIBackend(ImageGenerator):
    """通过本地 ComfyUI HTTP API 生成图片。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self._client: Any = None
        self.base_url = str(config.get("base_url") or _DEFAULT_BASE_URL).rstrip("/")
        workflow_path = config.get("workflow")
        if not workflow_path:
            raise ValueError("ComfyUI 后端需要配置 imagegen.workflow（API 格式 JSON 路径）")

        self.workflow_path = Path(str(workflow_path))
        self._base_workflow = load_workflow(self.workflow_path)
        self.width = int(config.get("width", 1024))
        self.height = int(config.get("height", 1792))
        self.steps = int(config.get("steps", 20))
        self.steps_multi_person = int(config.get("steps_multi_person", 25))
        self.multi_person_threshold = int(config.get("multi_person_threshold", 2))
        self.guidance_scale = float(config.get("guidance_scale", 7.5))
        raw_seed = config.get("seed")
        self.seed: int | None = int(raw_seed) if raw_seed is not None else None
        self.negative_prompt = ""  # FLUX/ComfyUI 不使用负向提示词
        self.timeout = float(config.get("timeout", 600))
        self.poll_interval = float(config.get("poll_interval", 1.0))
        self.output_node = (
            str(config["output_node"]).strip()
            if config.get("output_node")
            else None
        )
        self.node_map = dict(config.get("node_map") or {})
        self.workflow_inputs = dict(config.get("workflow_inputs") or {})
        self.lora_trigger = str(config.get("lora_trigger") or "").strip()
        if "prompt_prefix" in config:
            self.prompt_prefix = str(config.get("prompt_prefix") or "").strip()
        else:
            self.prompt_prefix = "beautiful anime illustration of"

        log.info(
            "ComfyUIBackend 初始化: url=%s workflow=%s size=%dx%d steps=%d lora_trigger=%r",
            self.base_url,
            self.workflow_path,
            self.width,
            self.height,
            self.steps,
            self.lora_trigger or None,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> ComfyUIBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _queue_prompt(self, workflow: dict[str, Any]) -> str:
        client = self._get_client()
        try:
            resp = client.post("/prompt", json={"prompt": workflow})
        except Exception as exc:
            raise ComfyUIError(
                f"无法连接 ComfyUI ({self.base_url})，请确认服务已启动: {exc}"
            ) from exc

        if resp.status_code >= 400:
            raise ComfyUIError(
                f"ComfyUI /prompt 失败 ({resp.status_code}): {resp.text[:500]}"
            )

        data = resp.json()
        node_errors = data.get("node_errors") or {}
        if node_errors:
            raise ComfyUIError(f"ComfyUI workflow 节点错误: {node_errors}")

        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"ComfyUI 未返回 prompt_id: {data}")
        return str(prompt_id)

    def _wait_for_history(self, prompt_id: str) -> dict[str, Any]:
        client = self._get_client()
        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:
            resp = client.get(f"/history/{prompt_id}")
            if resp.status_code == 404:
                time.sleep(self.poll_interval)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if prompt_id in payload:
                entry = payload[prompt_id]
                if entry.get("outputs"):
                    return entry
            time.sleep(self.poll_interval)

        raise ComfyUIError(
            f"ComfyUI 生图超时 ({self.timeout}s)，prompt_id={prompt_id}"
        )

    def _download_image(self, image_info: dict[str, str]) -> Image.Image:
        client = self._get_client()
        resp = client.get(
            "/view",
            params={
                "filename": image_info["filename"],
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            },
        )
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _resolve_steps(
        self, prompt: str, person_count: int | None = None
    ) -> int:
        count = person_count
        if count is None:
            count = estimate_person_count_from_prompt(prompt)
        if count > self.multi_person_threshold:
            log.info(
                "ComfyUI 多人场景 (%d 人 > %d)，steps %d → %d",
                count,
                self.multi_person_threshold,
                self.steps,
                self.steps_multi_person,
            )
            return self.steps_multi_person
        return self.steps

    def generate(self, prompt: str, **kwargs: Any) -> Image.Image:
        person_count = kwargs.get("person_count")
        if person_count is not None:
            person_count = int(person_count)
        steps = self._resolve_steps(prompt, person_count)
        final_prompt = finalize_comfyui_positive_prompt(
            prompt,
            lora_trigger=self.lora_trigger,
            prompt_prefix=self.prompt_prefix,
        )
        workflow = prepare_workflow(
            self._base_workflow,
            prompt=final_prompt,
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            steps=steps,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
            node_map=self.node_map or None,
            workflow_inputs=self.workflow_inputs or None,
        )

        prompt_id = self._queue_prompt(workflow)
        log.info("ComfyUI 已入队 prompt_id=%s", prompt_id)

        history_entry = self._wait_for_history(prompt_id)
        images = find_output_images(history_entry, output_node=self.output_node)
        if not images:
            raise ComfyUIError(
                f"ComfyUI 执行完成但未找到输出图片，prompt_id={prompt_id}"
            )

        image = self._download_image(images[0])
        log.info(
            "ComfyUI 生图完成: %dx%d prompt_id=%s",
            image.width,
            image.height,
            prompt_id,
        )
        return image
