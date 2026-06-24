"""ComfyUI 本地生图后端测试。"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.config_manager import resolve_pipeline_config
from src.imagegen.comfyui_backend import (
    ComfyUIError,
    ComfyUIBackend,
    apply_node_map,
    apply_workflow_inputs,
    build_generation_context,
    estimate_person_count_from_prompt,
    find_output_images,
    infer_node_map,
    load_workflow,
    prepend_lora_trigger,
    prepend_prompt_prefix,
    prepare_workflow,
    finalize_comfyui_positive_prompt,
    render_template,
    resolve_workflow_path,
)
from src.imagegen.image_generator import create_image_generator

_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {"seed": 1, "steps": 20, "cfg": 8},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "placeholder positive", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "placeholder negative", "clip": ["4", 1]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
    },
}


@pytest.mark.signature
class TestComfyUIHelpers:
    def test_render_template(self):
        ctx = {"prompt": "cat", "width": 1024}
        assert render_template("{prompt} on desk", ctx) == "cat on desk"
        assert render_template(42, ctx) == 42

    def test_infer_node_map(self):
        node_map = infer_node_map(_WORKFLOW)
        assert node_map["positive"] == "6"
        assert node_map["negative"] == "7"
        assert node_map["latent"] == "5"
        assert node_map["sampler"] == "3"

    def test_apply_node_map(self):
        ctx = build_generation_context(
            "a tabby cat",
            negative_prompt="blurry",
            width=1024,
            height=1792,
            steps=28,
            guidance_scale=4.5,
            seed=123,
        )
        wf = apply_node_map(_WORKFLOW, infer_node_map(_WORKFLOW), ctx)
        assert wf["6"]["inputs"]["text"] == "a tabby cat"
        assert wf["7"]["inputs"]["text"] == "blurry"
        assert wf["5"]["inputs"]["width"] == 1024
        assert wf["5"]["inputs"]["height"] == 1792
        assert wf["3"]["inputs"]["seed"] == 123
        assert wf["3"]["inputs"]["steps"] == 28
        assert wf["3"]["inputs"]["cfg"] == 4.5

    def test_apply_workflow_inputs(self):
        wf = apply_workflow_inputs(
            _WORKFLOW,
            {
                "6.inputs.text": "{prompt}",
                "5.inputs.width": "{width}",
            },
            build_generation_context("dog", width=768, height=768, seed=9),
        )
        assert wf["6"]["inputs"]["text"] == "dog"
        assert wf["5"]["inputs"]["width"] == 768

    def test_prepare_workflow_requires_positive_mapping(self):
        broken = {"1": {"class_type": "SaveImage", "inputs": {}}}
        with pytest.raises(ComfyUIError, match="positive"):
            prepare_workflow(broken, prompt="x")

    def test_find_output_images(self):
        history = {
            "outputs": {
                "9": {
                    "images": [
                        {
                            "filename": "ComfyUI_00001_.png",
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            }
        }
        images = find_output_images(history)
        assert len(images) == 1
        assert images[0]["filename"] == "ComfyUI_00001_.png"

    def test_find_output_images_with_output_node(self):
        history = {
            "outputs": {
                "9": {"images": [{"filename": "a.png"}]},
                "10": {"images": [{"filename": "b.png"}]},
            }
        }
        images = find_output_images(history, output_node="10")
        assert images[0]["filename"] == "b.png"

    def test_load_example_workflow(self):
        path = resolve_workflow_path("workflows/comfyui/sd15_t2i_api.example.json")
        wf = load_workflow(path)
        assert wf["6"]["class_type"] == "CLIPTextEncode"

    def test_infer_fluxscale_workflow(self):
        wf = load_workflow("workflows/comfyui/fluxscale_workflow.json")
        node_map = infer_node_map(wf)
        assert node_map["positive"] == "6"
        assert node_map["latent"] == "5"
        assert node_map["noise"] == "17"
        assert node_map["scheduler"] == "19"
        assert "negative" not in node_map

    def test_apply_fluxscale_node_map(self):
        wf = load_workflow("workflows/comfyui/fluxscale_workflow.json")
        ctx = build_generation_context(
            "anime girl in bookstore",
            width=1152,
            height=896,
            steps=28,
            seed=999,
        )
        prepared = apply_node_map(wf, infer_node_map(wf), ctx)
        assert prepared["6"]["inputs"]["text"] == "anime girl in bookstore"
        assert prepared["5"]["inputs"]["width"] == 1152
        assert prepared["17"]["inputs"]["noise_seed"] == 999
        assert prepared["19"]["inputs"]["steps"] == 28

    def test_estimate_person_count_from_prompt(self):
        assert estimate_person_count_from_prompt("") == 0
        assert estimate_person_count_from_prompt("sunset over mountains") == 0
        assert estimate_person_count_from_prompt(
            "a young woman and a young man in a cafe"
        ) == 2
        assert estimate_person_count_from_prompt(
            "three people talking in an office"
        ) == 3
        assert estimate_person_count_from_prompt(
            "【本段相关角色，外观保持一致】\n"
            "Alice：blonde\n"
            "Bob：tall\n"
            "Carol：short"
        ) == 3

    def test_prepend_lora_trigger(self):
        assert prepend_lora_trigger("girl at desk", "YuanRun") == (
            "YuanRun, girl at desk"
        )
        assert prepend_lora_trigger("YuanRun, girl at desk", "YuanRun") == (
            "YuanRun, girl at desk"
        )
        assert prepend_lora_trigger("yuanrun, girl at desk", "YuanRun") == (
            "yuanrun, girl at desk"
        )
        assert prepend_lora_trigger("", "YuanRun") == "YuanRun"
        assert prepend_lora_trigger("girl at desk", "") == "girl at desk"

    def test_prepend_prompt_prefix(self):
        assert prepend_prompt_prefix("girl at desk", "beautiful anime illustration of") == (
            "beautiful anime illustration of, girl at desk"
        )
        assert prepend_prompt_prefix(
            "beautiful anime illustration of, girl at desk",
            "beautiful anime illustration of",
        ) == "beautiful anime illustration of, girl at desk"

    def test_finalize_comfyui_positive_prompt(self):
        assert finalize_comfyui_positive_prompt(
            "girl in bookstore",
            lora_trigger="YuanRun",
            prompt_prefix="beautiful anime illustration of",
        ) == "YuanRun, beautiful anime illustration of, girl in bookstore"


@pytest.mark.signature
class TestComfyUIBackend:
    def _make_backend(self, tmp_path: Path, **overrides) -> ComfyUIBackend:
        workflow_file = tmp_path / "workflow.json"
        workflow_file.write_text(json.dumps(_WORKFLOW), encoding="utf-8")
        config = {
            "base_url": "http://127.0.0.1:8188",
            "workflow": str(workflow_file),
            "width": 640,
            "height": 960,
            "steps": 12,
            "guidance_scale": 6.0,
            "seed": 42,
            "negative_prompt": "bad quality",
            "timeout": 5,
            "poll_interval": 0.01,
            "prompt_prefix": "",
        }
        config.update(overrides)
        return ComfyUIBackend(config)

    def test_init_missing_workflow_raises(self):
        with pytest.raises(ValueError, match="workflow"):
            ComfyUIBackend({"backend": "comfyui"})

    def test_init_missing_workflow_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="workflow 不存在"):
            ComfyUIBackend(
                {
                    "workflow": str(tmp_path / "missing.json"),
                }
            )

    def test_resolve_steps_multi_person(self, tmp_path: Path):
        backend = self._make_backend(
            tmp_path, steps=16, steps_multi_person=25, multi_person_threshold=2
        )
        assert backend._resolve_steps("solo scene", person_count=1) == 16
        assert backend._resolve_steps("duo scene", person_count=2) == 16
        assert backend._resolve_steps("group scene", person_count=3) == 25
        assert backend._resolve_steps(
            "three people in a room", person_count=None
        ) == 25

    @patch("src.imagegen.comfyui_backend.time.sleep")
    def test_generate_uses_multi_person_steps(self, _sleep: MagicMock, tmp_path: Path):
        backend = self._make_backend(
            tmp_path, steps=16, steps_multi_person=25, multi_person_threshold=2
        )
        fake_png = io.BytesIO()
        Image.new("RGB", (640, 960), (120, 80, 40)).save(fake_png, format="PNG")

        prompt_resp = MagicMock()
        prompt_resp.status_code = 200
        prompt_resp.json.return_value = {"prompt_id": "pid-mp", "node_errors": {}}

        history_done = MagicMock()
        history_done.status_code = 200
        history_done.json.return_value = {
            "pid-mp": {
                "outputs": {
                    "9": {
                        "images": [
                            {
                                "filename": "ComfyUI_00001_.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                }
            }
        }

        view_resp = MagicMock()
        view_resp.status_code = 200
        view_resp.content = fake_png.getvalue()
        view_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = prompt_resp
        mock_client.get.side_effect = [history_done, view_resp]

        with patch.object(backend, "_get_client", return_value=mock_client):
            backend.generate("group meeting", person_count=3)

        posted = mock_client.post.call_args.kwargs["json"]["prompt"]
        assert posted["6"]["inputs"]["text"] == "group meeting"
        assert posted["3"]["inputs"]["steps"] == 25

    @patch("src.imagegen.comfyui_backend.time.sleep")
    def test_generate_prepends_lora_trigger(self, _sleep: MagicMock, tmp_path: Path):
        backend = self._make_backend(
            tmp_path,
            lora_trigger="YuanRun",
            prompt_prefix="beautiful anime illustration of",
        )
        fake_png = io.BytesIO()
        Image.new("RGB", (640, 960), (120, 80, 40)).save(fake_png, format="PNG")

        prompt_resp = MagicMock()
        prompt_resp.status_code = 200
        prompt_resp.json.return_value = {"prompt_id": "pid-lora", "node_errors": {}}

        history_done = MagicMock()
        history_done.status_code = 200
        history_done.json.return_value = {
            "pid-lora": {
                "outputs": {
                    "9": {
                        "images": [
                            {
                                "filename": "ComfyUI_00001_.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                }
            }
        }

        view_resp = MagicMock()
        view_resp.status_code = 200
        view_resp.content = fake_png.getvalue()
        view_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = prompt_resp
        mock_client.get.side_effect = [history_done, view_resp]

        with patch.object(backend, "_get_client", return_value=mock_client):
            backend.generate("girl in bookstore")

        posted = mock_client.post.call_args.kwargs["json"]["prompt"]
        assert posted["6"]["inputs"]["text"].startswith(
            "YuanRun, beautiful anime illustration of, "
        )

    @patch("src.imagegen.comfyui_backend.time.sleep")
    def test_generate_success(self, _sleep: MagicMock, tmp_path: Path):
        backend = self._make_backend(tmp_path)
        fake_png = io.BytesIO()
        Image.new("RGB", (640, 960), (120, 80, 40)).save(fake_png, format="PNG")

        prompt_resp = MagicMock()
        prompt_resp.status_code = 200
        prompt_resp.json.return_value = {"prompt_id": "pid-1", "node_errors": {}}

        history_pending = MagicMock()
        history_pending.status_code = 200
        history_pending.json.return_value = {}

        history_done = MagicMock()
        history_done.status_code = 200
        history_done.json.return_value = {
            "pid-1": {
                "outputs": {
                    "9": {
                        "images": [
                            {
                                "filename": "ComfyUI_00001_.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                }
            }
        }

        view_resp = MagicMock()
        view_resp.status_code = 200
        view_resp.content = fake_png.getvalue()
        view_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = prompt_resp
        mock_client.get.side_effect = [history_pending, history_done, view_resp]

        with patch.object(backend, "_get_client", return_value=mock_client):
            image = backend.generate("sunset over mountains")

        assert image.size == (640, 960)
        posted = mock_client.post.call_args.kwargs["json"]["prompt"]
        assert posted["6"]["inputs"]["text"] == "sunset over mountains"
        assert posted["7"]["inputs"]["text"] == "placeholder negative"
        assert posted["5"]["inputs"]["width"] == 640
        assert posted["3"]["inputs"]["seed"] == 42

    @patch("src.imagegen.comfyui_backend.time.sleep")
    def test_generate_node_errors(self, _sleep: MagicMock, tmp_path: Path):
        backend = self._make_backend(tmp_path)
        prompt_resp = MagicMock()
        prompt_resp.status_code = 200
        prompt_resp.json.return_value = {
            "prompt_id": "pid-2",
            "node_errors": {"4": {"errors": ["missing model"]}},
        }
        mock_client = MagicMock()
        mock_client.post.return_value = prompt_resp

        with patch.object(backend, "_get_client", return_value=mock_client):
            with pytest.raises(ComfyUIError, match="节点错误"):
                backend.generate("test")

    @patch("src.imagegen.comfyui_backend.time.sleep")
    def test_generate_timeout(self, _sleep: MagicMock, tmp_path: Path):
        backend = self._make_backend(tmp_path, timeout=0.02, poll_interval=0.01)
        prompt_resp = MagicMock()
        prompt_resp.status_code = 200
        prompt_resp.json.return_value = {"prompt_id": "pid-3", "node_errors": {}}

        history_resp = MagicMock()
        history_resp.status_code = 200
        history_resp.json.return_value = {}

        mock_client = MagicMock()
        mock_client.post.return_value = prompt_resp
        mock_client.get.return_value = history_resp

        with patch.object(backend, "_get_client", return_value=mock_client):
            with pytest.raises(ComfyUIError, match="超时"):
                backend.generate("test")

    def test_generate_connection_error(self, tmp_path: Path):
        backend = self._make_backend(tmp_path)
        mock_client = MagicMock()
        mock_client.post.side_effect = OSError("connection refused")

        with patch.object(backend, "_get_client", return_value=mock_client):
            with pytest.raises(ComfyUIError, match="无法连接 ComfyUI"):
                backend.generate("test")


@pytest.mark.signature
class TestComfyUIFactory:
    def test_create_image_generator_comfyui(self, tmp_path: Path):
        workflow_file = tmp_path / "workflow.json"
        workflow_file.write_text(json.dumps(_WORKFLOW), encoding="utf-8")
        gen = create_image_generator(
            {
                "backend": "comfyui",
                "workflow": str(workflow_file),
            }
        )
        assert isinstance(gen, ComfyUIBackend)
        assert gen.base_url == "http://127.0.0.1:8188"

    def test_resolve_pipeline_config_top_level_comfyui_all_modes(self):
        cfg = {
            "segmenter": {},
            "promptgen": {},
            "imagegen": {
                "backend": "comfyui",
                "workflow": "workflows/comfyui/sd15_t2i_api.example.json",
                "base_url": "http://127.0.0.1:8188",
            },
            "tts": {},
            "video": {"resolution": [1920, 1080]},
            "director": {"videogen": {"backend": "seedance"}},
        }
        for mode in ("agent", "director"):
            resolved = resolve_pipeline_config(cfg, mode)
            assert resolved["imagegen"]["backend"] == "comfyui"
            assert resolved["imagegen"]["workflow"].endswith(
                "workflows/comfyui/sd15_t2i_api.example.json"
            )

    def test_resolve_pipeline_config_agent_imagegen_override(self):
        cfg = {
            "segmenter": {},
            "promptgen": {},
            "imagegen": {"backend": "jimeng-cli", "ratio": "16:9"},
            "tts": {},
            "video": {"resolution": [1920, 1080], "auto_resolution": True},
            "agent": {
                "imagegen": {
                    "backend": "comfyui",
                    "workflow": "workflows/comfyui/sd15_t2i_api.example.json",
                    "width": 1024,
                    "height": 1792,
                }
            },
        }
        resolved = resolve_pipeline_config(cfg, "agent")
        assert resolved["imagegen"]["backend"] == "comfyui"
        assert resolved["imagegen"]["workflow"].endswith(
            "workflows/comfyui/sd15_t2i_api.example.json"
        )
        assert resolved["imagegen"]["width"] == 1024
