"""恐怖分镜 prompt 含蓄恐怖约束测试。"""

import pytest


@pytest.mark.signature
class TestSubtleHorrorPrompt:
    def _make_gen(self, **overrides):
        from src.promptgen.prompt_generator import PromptGenerator

        cfg = {
            "style": "anime",
            "llm": {"provider": "none"},
            "horror_style": "subtle",
        }
        cfg.update(overrides)
        return PromptGenerator(cfg)

    def test_detects_horror_segment(self):
        gen = self._make_gen()
        assert gen._is_horror_segment("走廊里传来诡异的脚步声")
        assert not gen._is_horror_segment("他们在咖啡厅里聊天")

    def test_apply_subtle_horror_appends_suffix(self):
        gen = self._make_gen()
        raw = "a dark hallway, cinematic lighting"
        text = "毛骨悚然的走廊尽头有一扇门"
        out = gen._apply_subtle_horror(raw, text)
        assert "no gore" in out
        assert "subtle psychological horror" in out

    def test_apply_subtle_horror_skips_when_off(self):
        gen = self._make_gen(horror_style="off")
        raw = "a dark hallway"
        out = gen._apply_subtle_horror(raw, "恐怖的氛围")
        assert out == raw

    def test_apply_subtle_horror_skips_when_tone_light(self):
        gen = self._make_gen(tone="light")
        raw = "a dark hallway"
        out = gen._apply_subtle_horror(raw, "毛骨悚然的走廊")
        assert out == raw

    def test_tone_light_softens_local_horror_prompt(self):
        gen = self._make_gen(tone="light", horror_style="subtle")
        prompt = gen.generate("深夜走廊里传来诡异回声，他感到毛骨悚然。", segment_index=0)
        lowered = prompt.lower()
        assert "implied dread" not in lowered
        assert (
            "bright soft daylight" in lowered
            or "high key lighting" in lowered
            or "light storytelling mood" in lowered
        )

    def test_visual_source_text_strips_series_header(self):
        from src.promptgen.prompt_generator import PromptGenerator

        raw = (
            "恶之花：暗黑困境中的觉醒和救赎\n"
            "【极致捧杀】\n"
            "1. 极致捧杀\n"
            "高考出分，我稳上北大，她大闹着把家里的东西给砸了。"
        )
        cleaned = PromptGenerator._visual_source_text(raw)
        assert "暗黑" not in cleaned
        assert "高考出分" in cleaned

    def test_segment0_metadata_not_in_local_prompt(self):
        gen = self._make_gen(tone="light")
        seg0 = (
            "恶之花：暗黑困境中的觉醒和救赎\n"
            "【极致捧杀】1. 极致捧杀高考出分，我稳上北大。"
        )
        prompt = gen.generate(seg0, segment_index=0).lower()
        assert "dark" not in prompt or "not dark" in prompt

    def test_local_mode_uses_subtle_horror_keywords(self):
        gen = self._make_gen()
        prompt = gen.generate("深夜走廊里传来诡异回声，他感到毛骨悚然。", segment_index=0)
        assert "no gore" in prompt.lower() or "implied dread" in prompt.lower()

    def test_blood_scene_avoids_graphic_wording(self):
        gen = self._make_gen()
        prompt = gen.generate("墙上留下了血迹，空气中弥漫着腐臭。", segment_index=0)
        lowered = prompt.lower()
        assert "no blood" in lowered or "no gore" in lowered
        assert "bloody" not in lowered
