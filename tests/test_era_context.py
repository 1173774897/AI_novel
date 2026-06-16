"""时代背景约束测试。"""
from __future__ import annotations

import pytest

from src.promptgen.era_context import (
    CLASSICAL,
    normalize_era,
    sanitize_classical_desc,
    default_hairstyle_for_era,
)
from src.promptgen.prompt_generator import PromptGenerator

pytestmark = pytest.mark.signature


class TestEraContext:
    def test_normalize_era_ancient_aliases(self):
        assert normalize_era("古代") == CLASSICAL
        assert normalize_era("auto") is None
        assert normalize_era("现代") == "modern"

    def test_sanitize_classical_desc_fixes_modern_hair(self):
        desc = "黑色短发，傅知舟年十五，穿雨过天青色广袖衫"
        out = sanitize_classical_desc(desc)
        assert "短发" not in out
        assert "玉冠" in out or "束" in out

    def test_default_hairstyle_classical_female(self):
        assert "髻" in default_hairstyle_for_era("年轻女子", CLASSICAL)


class TestPromptGeneratorEra:
    def test_set_era_overrides_detection(self):
        gen = PromptGenerator({"era": "古代", "character_tracking": False})
        gen.set_full_text("他拿出手机在地铁里刷视频")
        assert gen._get_era("") == CLASSICAL

    def test_llm_user_msg_includes_classical_note(self):
        gen = PromptGenerator({"era": "古代", "character_tracking": False})
        msg = gen._append_era_llm_note("base")
        assert "中国古代" in msg
        assert "ancient China" in msg
