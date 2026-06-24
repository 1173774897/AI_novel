"""Prompt 短语去重与增量追加测试。"""

import pytest


@pytest.mark.signature
class TestPromptCompact:
    def test_compact_removes_exact_duplicates(self):
        from src.promptgen.prompt_generator import PromptGenerator

        raw = "anime girl, warm lighting, anime girl, cheerful mood"
        out = PromptGenerator._compact_prompt_phrases(raw)
        assert out.lower().count("anime girl") == 1
        assert "warm lighting" in out
        assert "cheerful mood" in out

    def test_compact_drops_subsumed_phrase(self):
        from src.promptgen.prompt_generator import PromptGenerator

        raw = (
            "supermarket uniform, "
            "young woman with black hair in supermarket uniform smiling"
        )
        out = PromptGenerator._compact_prompt_phrases(raw)
        assert "supermarket uniform" not in out or out.count("supermarket uniform") == 1
        assert "young woman with black hair" in out

    def test_append_missing_skips_existing_positive(self):
        from src.promptgen.prompt_generator import PromptGenerator

        base = (
            "beautiful anime illustration of, two girls talking, "
            "anime style, cel shading, vibrant colors"
        )
        extra = "anime style, cel shading, studio ghibli inspired"
        out = PromptGenerator._append_missing_phrases(base, extra)
        assert out.count("anime style") == 1
        assert out.count("cel shading") == 1
        assert "studio ghibli inspired" in out

    def test_apply_style_skips_duplicate_prefix_and_positive(self):
        from src.promptgen.prompt_generator import PromptGenerator

        gen = PromptGenerator({"style": "anime", "llm": {"provider": "none"}})
        raw = (
            "beautiful anime illustration of, two roommates in dorm, "
            "anime style, cel shading, vibrant colors, detailed scenery"
        )
        out = gen._apply_style(raw)
        assert out.lower().count("beautiful anime illustration of") == 1
        assert out.lower().count("anime style") == 1
        assert "NOT photorealistic" in out

    def test_generate_compacts_repeated_tone_suffix(self):
        from src.promptgen.prompt_generator import PromptGenerator

        gen = PromptGenerator(
            {
                "style": "anime",
                "llm": {"provider": "none"},
                "tone": "light",
                "horror_style": "off",
                "imagegen_backend": "together",
            }
        )
        gen._use_llm = False
        prompt = gen.generate("她在明亮的宿舍里整理书桌。", segment_index=0)
        assert prompt.lower().count("cheerful everyday mood") <= 1
        assert prompt.lower().count("bright soft daylight") <= 1
