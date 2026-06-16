"""叙述者有限视角 prompt 测试。"""

import pytest

from src.promptgen.author_pov import (
    build_author_pov_instruction,
    classify_author_pov_scene,
    detect_first_person_work,
    narrator_physically_present,
)
from src.promptgen.prompt_generator import PromptGenerator

pytestmark = pytest.mark.signature

_WUJIN_CHAT = (
    "501的业主突然在群里发了消息：【别出去，楼道里有杀人狂！】\n"
    "302：【不是，你大晚上的，这么吓人好吗？】\n"
    "我莫名觉得501不像是开玩笑。"
)

_WUJIN_HEARD = (
    "我正在303，刚确实听得很真切\n"
    "楼上“嘭”的一声，应该是404摔门而出\n"
    "接着，就归于平静了\n"
    "再没有任何的声音传出来"
)

_WUJIN_MEMORY = (
    "他平时是个少言寡语的人,我见过他几次,四十来岁,总是戴着一副眼镜,"
    "一副老学究的样子"
)


class TestAuthorPovDetection:
    def test_detect_first_person_work_for_wujin(self):
        sample = _WUJIN_CHAT + _WUJIN_HEARD + "我觉得不对劲，我不敢出门。"
        assert detect_first_person_work(sample * 5)

    def test_classify_chat_scene(self):
        assert classify_author_pov_scene(_WUJIN_CHAT) == "chat"

    def test_classify_heard_scene(self):
        assert classify_author_pov_scene(_WUJIN_HEARD) == "heard"

    def test_classify_memory_scene(self):
        assert classify_author_pov_scene(_WUJIN_MEMORY) == "memory"

    def test_narrator_not_present_in_chat(self):
        assert not narrator_physically_present(_WUJIN_CHAT)

    def test_instruction_forbids_offscreen_violence(self):
        inst = build_author_pov_instruction(_WUJIN_HEARD)
        assert "不在场" in inst
        assert "听到" in inst or "倾听" in inst


class TestAuthorPovPromptGenerator:
    def _make_gen(self, **overrides):
        cfg = {
            "style": "anime",
            "llm": {"provider": "none"},
            "pov_mode": "author",
            "horror_style": "subtle",
        }
        cfg.update(overrides)
        return PromptGenerator(cfg)

    def test_chat_segment_uses_phone_pov_keywords(self):
        gen = self._make_gen()
        prompt = gen.generate(_WUJIN_CHAT, segment_index=0)
        lowered = prompt.lower()
        assert "smartphone" in lowered or "phone" in lowered
        assert "first person" in lowered

    def test_heard_segment_avoids_hallway_killer(self):
        gen = self._make_gen()
        prompt = gen.generate(_WUJIN_HEARD, segment_index=1)
        lowered = prompt.lower()
        assert "listening" in lowered or "door" in lowered
        assert "no killer" in lowered or "no off-screen" in lowered

    def test_auto_mode_enables_after_set_full_text(self):
        gen = self._make_gen(pov_mode="auto")
        gen.set_full_text(_WUJIN_CHAT * 20 + _WUJIN_HEARD * 10)
        assert gen._uses_author_pov(_WUJIN_CHAT)
