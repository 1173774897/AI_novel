"""叙述者未明确时不入画测试。"""

from __future__ import annotations

import pytest

from src.promptgen.narrator import (
    build_omit_narrator_instruction,
    resolve_narrator_visual,
)
from src.promptgen.prompt_generator import PromptGenerator

pytestmark = pytest.mark.signature

_LING_CHARACTERS = [
    {"name": "伶伶", "desc": "17岁女生，清瘦，校服，短发。"},
    {"name": "崔彤", "desc": "17岁女生，栗色长发，名牌休闲装。"},
    {"name": "伯父", "desc": "40岁中年男人，深色Polo衫，方脸。"},
    {"name": "伯母", "desc": "40岁中年妇女，丝绸开衫。"},
]

_SEG0 = (
    "高考出分，我稳上北大，我堂妹上——挖掘机技术哪家强。"
    "她大闹着把家里的东西给砸了，「不是要我顶替她上大学吗！」"
)
_SEG1 = "教室里，身边的同学在奋笔疾书。堂妹在身后拼命踹我的椅子。我没理她。"

_ANSWERER = [
    {
        "name": "答主（我）",
        "desc": "男，18岁，高三学生。戴黑框眼镜，蓝白校服，短发。",
    },
    {"name": "潘博文", "desc": "男，18岁，高三学生。"},
]


class TestResolveNarratorVisual:
    def test_pov_with_desc(self):
        name, desc = resolve_narrator_visual("伶伶", _LING_CHARACTERS)
        assert name == "伶伶"
        assert "17岁" in desc

    def test_no_pov_no_alias_returns_empty(self):
        name, desc = resolve_narrator_visual(None, _LING_CHARACTERS)
        assert name is None
        assert desc == ""

    def test_answerer_alias_without_pov(self):
        name, desc = resolve_narrator_visual(None, _ANSWERER)
        assert name == "答主（我）"
        assert "18岁" in desc


class TestOmitUnidentifiedNarrator:
    def test_no_pov_does_not_bind_uncle_as_narrator(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_LING_CHARACTERS)
        names = gen._narrator_cast_names(_SEG0)
        bible = gen._build_cast_bible(_SEG0)
        assert names == []
        assert "伯父" not in (bible or "")

    def test_first_person_without_identity_omits_narrator_instruction(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_LING_CHARACTERS)
        _, instruction = gen._build_character_context(_SEG0)
        assert "禁止将「我」画成画面中的可见人物" in instruction
        assert "40岁" not in instruction
        assert "male" not in instruction.lower()

    def test_pov_lingling_includes_narrator_in_cast(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.set_pov_narrator("伶伶")
        gen.seed_characters(_LING_CHARACTERS)
        names = gen._narrator_cast_names(_SEG1)
        bible = gen._build_cast_bible(_SEG1)
        assert names == ["伶伶"]
        assert "伶伶" in bible
        assert "伯父" not in bible

    def test_pov_lingling_uses_identity_instruction_not_uncle(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.set_pov_narrator("伶伶")
        gen.seed_characters(_LING_CHARACTERS)
        _, instruction = gen._build_character_context(_SEG1)
        assert "伶伶" in instruction
        assert "禁止将「我」画成" not in instruction

    def test_apply_author_pov_suffix_omits_body_when_unidentified(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_LING_CHARACTERS)
        out = gen._apply_author_pov("anime classroom scene", _SEG1)
        assert "no visible narrator protagonist" in out.lower()
        assert "first person limited perspective" not in out.lower()

    def test_answerer_alias_still_allows_narrator(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_ANSWERER)
        text = "我的认知和记忆里是这样的，讲出来了，就这样。"
        names = gen._narrator_cast_names(text)
        assert names == ["答主（我）"]
        _, instruction = gen._build_character_context(text)
        assert "禁止将「我」画成" not in instruction
        assert "18岁" in instruction

    def test_omit_instruction_constant(self):
        assert "第三人称" in build_omit_narrator_instruction()
