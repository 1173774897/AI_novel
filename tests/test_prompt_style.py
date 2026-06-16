"""风格预设与 anime 抗写实注入测试。"""

import pytest


@pytest.mark.signature
class TestAnimeStylePrompt:
    def _make_gen(self, **overrides):
        from src.promptgen.prompt_generator import PromptGenerator

        cfg = {
            "style": "anime",
            "llm": {"provider": "none"},
            "horror_style": "subtle",
        }
        cfg.update(overrides)
        return PromptGenerator(cfg)

    def test_apply_style_prepends_anime_prefix(self):
        gen = self._make_gen()
        out = gen._apply_style("a man looking through peephole, tense mood")
        assert out.startswith("beautiful anime illustration of")
        assert "cel shading" in out
        assert "NOT photorealistic" in out

    def test_apply_style_strips_photorealistic_terms(self):
        gen = self._make_gen()
        raw = "photorealistic cinematic photo, a man at door, live action, 8k photo"
        out = gen._apply_style(raw)
        body = out.split("beautiful anime illustration of,", 1)[-1]
        assert "photorealistic cinematic photo" not in body.lower()
        assert ", live action," not in body.lower()
        assert "anime illustration" in out.lower()

    def test_set_style_switches_preset(self):
        gen = self._make_gen()
        gen.set_style("realistic")
        out = gen._apply_style("a street scene")
        assert "photorealistic" in out

    def test_local_mode_includes_anime_prefix(self):
        gen = self._make_gen()
        prompt = gen.generate("他站在门口，紧张地看着猫眼。", segment_index=0)
        assert "beautiful anime illustration" in prompt.lower()
        assert "cel shading" in prompt.lower()

    def test_anime_llm_note_only_for_anime(self):
        gen = self._make_gen()
        msg = gen._append_style_llm_note("base")
        assert "anime illustration" in msg
        gen.set_style("realistic")
        assert gen._append_style_llm_note("base") == "base"
