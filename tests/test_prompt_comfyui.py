"""ComfyUI/FLUX 模式下 prompt 不含负向提示词。"""

from unittest.mock import MagicMock, patch

import pytest

from tests.test_prompt_peaceful_guard import SEGMENT_85, SEGMENT_86


@pytest.mark.signature
class TestPromptComfyUIMode:
    def _make_gen(self, **overrides):
        from src.promptgen.prompt_generator import PromptGenerator

        cfg = {
            "style": "anime",
            "llm": {"provider": "none"},
            "horror_style": "off",
            "tone": "light",
            "imagegen_backend": "comfyui",
        }
        cfg.update(overrides)
        gen = PromptGenerator(cfg)
        gen._use_llm = False
        return gen

    def test_strip_negative_phrases(self):
        from src.promptgen.prompt_generator import PromptGenerator

        raw = (
            "anime girl, NOT photorealistic, no blood, cheerful mood, "
            "not dark atmosphere, warm lighting"
        )
        out = PromptGenerator._strip_negative_phrases(raw).lower()
        assert "not photorealistic" not in out
        assert "no blood" not in out
        assert "not dark" not in out
        assert "cheerful mood" in out
        assert "warm lighting" in out

    def test_positive_only_keywords_from_anime_preset(self):
        from src.promptgen.style_presets import get_preset

        gen = self._make_gen()
        positive = gen._positive_only_keywords(get_preset("anime")["positive"]).lower()
        assert "anime style" in positive
        assert "not photorealistic" not in positive
        assert "not live action" not in positive

    def test_strip_style_boilerplate_removes_user_example_tail(self):
        from src.promptgen.prompt_generator import PromptGenerator

        raw = (
            "two roommates in a dorm, girl holding phone, "
            "anime style, 2D illustration, hand-drawn animation aesthetic, vibrant colors, "
            "studio ghibli inspired, beautiful scenery, bright soft daylight, "
            "cheerful everyday mood, light storytelling mood, "
            "first person POV inside apartment at night, narrator alone in dim room, "
            "uneasy atmosphere, limited perspective, subjective POV"
        )
        out = PromptGenerator._strip_style_boilerplate(raw).lower()
        assert "two roommates in a dorm" in out
        assert "girl holding phone" in out
        assert "anime style" not in out
        assert "cel shading" not in out
        assert "cheerful everyday mood" not in out
        assert "subjective pov" not in out

    def test_apply_peaceful_scene_guard_skips_negative_suffix(self):
        gen = self._make_gen()
        raw = "a teenage girl with glasses holding a phone, warm lighting"
        out = gen._apply_peaceful_scene_guard(raw, SEGMENT_86, SEGMENT_85).lower()
        assert "no blood" not in out
        assert "no gore" not in out

    def test_generate_content_only_no_style_boilerplate(self):
        gen = self._make_gen(prompt_prefix="", lora_trigger="")
        raw = (
            "beautiful anime illustration of, girl at dining table with phone, "
            "anime style, cel shading, cheerful everyday mood, subjective POV"
        )
        prompt = gen._finalize_image_prompt(raw, SEGMENT_86, SEGMENT_85).lower()
        assert "girl at dining table" in prompt
        assert "phone" in prompt
        assert "beautiful anime illustration" not in prompt
        assert "cel shading" not in prompt
        assert "cheerful everyday mood" not in prompt
        assert "subjective pov" not in prompt

    def test_finalize_prepends_style_prefix_and_lora(self):
        gen = self._make_gen(lora_trigger="YuanRun")
        raw = "girl at dining table with phone, warm lighting"
        prompt = gen._finalize_image_prompt(raw, SEGMENT_86, SEGMENT_85)
        assert prompt.startswith("YuanRun, beautiful anime illustration of, ")
        assert "girl at dining table" in prompt

    def test_finalize_prepends_lora_trigger(self):
        gen = self._make_gen(lora_trigger="YuanRun", prompt_prefix="")
        raw = "girl at dining table with phone, warm lighting"
        prompt = gen._finalize_image_prompt(raw, SEGMENT_86, SEGMENT_85)
        assert prompt.startswith("YuanRun, ")
        assert "girl at dining table" in prompt

    def test_finalize_skips_duplicate_lora_trigger(self):
        gen = self._make_gen(lora_trigger="YuanRun", prompt_prefix="")
        raw = "YuanRun, girl at dining table with phone"
        prompt = gen._finalize_image_prompt(raw, SEGMENT_86, SEGMENT_85)
        assert prompt.count("YuanRun") == 1
        assert prompt.startswith("YuanRun, ")

    @patch("src.promptgen.prompt_generator.PromptGenerator._get_llm_client")
    def test_llm_user_msg_uses_comfyui_content_note(self, mock_client_factory):
        from src.promptgen.prompt_generator import PromptGenerator

        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            content="anime scene, girl at dining table, cheerful mood"
        )
        mock_client_factory.return_value = mock_llm

        gen = PromptGenerator(
            {
                "style": "anime",
                "llm": {"provider": "openai"},
                "tone": "light",
                "imagegen_backend": "comfyui",
            }
        )
        gen._use_llm = True
        gen.generate(SEGMENT_86, segment_index=86, prev_text=SEGMENT_85)

        user_msg = mock_llm.chat.call_args.kwargs["messages"][1]["content"]
        assert "ComfyUI 仅画面内容" in user_msg
        assert "画面风格: anime" not in user_msg
        system_msg = mock_llm.chat.call_args.kwargs["messages"][0]["content"]
        assert "禁止画风词" in system_msg
