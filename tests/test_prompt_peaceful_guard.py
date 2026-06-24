"""非暴力分镜 prompt 防血腥串戏测试。"""

from unittest.mock import MagicMock, patch

import pytest


SEGMENT_86 = (
    "「那可不嘛，我们家可可今天出成绩的时候也跟我抱怨，说数学题太难了，"
    "全年级也就一个满分，好像也是姓崔，叫崔什么……噢，崔彤！」"
    "「苏总，你这记性，崔彤不就是我们崔总的千金嘛！」"
)
SEGMENT_85 = (
    "「哎，现在她们的数学试卷可难了，听说都跟那个什么，全国奥数竞赛题一样的难度了。」"
    "伯母说。有一个一块儿吃饭的，孩子应该也是在三中读高三，跟着应和。"
)


@pytest.mark.signature
class TestPeacefulSceneGuard:
    def _make_gen(self, **overrides):
        from src.promptgen.prompt_generator import PromptGenerator

        cfg = {
            "style": "anime",
            "llm": {"provider": "none"},
            "horror_style": "off",
            "tone": "light",
        }
        cfg.update(overrides)
        gen = PromptGenerator(cfg)
        gen._use_llm = False
        return gen

    def test_has_violent_content_detects_blood_not_praise_kill(self):
        from src.promptgen.prompt_generator import PromptGenerator

        assert PromptGenerator._has_violent_content("墙上留下了血迹")
        assert PromptGenerator._has_violent_content("她无证驾驶撞飞了一个人")
        assert not PromptGenerator._has_violent_content("极致捧杀堂妹")
        assert not PromptGenerator._has_violent_content(SEGMENT_86)

    def test_is_dining_scene_uses_prev_text(self):
        from src.promptgen.prompt_generator import PromptGenerator

        assert not PromptGenerator._is_dining_scene(SEGMENT_86)
        assert PromptGenerator._is_dining_scene(SEGMENT_86, SEGMENT_85)

    def test_sanitize_gore_from_prompt(self):
        from src.promptgen.prompt_generator import PromptGenerator

        raw = (
            "anime girl in living room, corpse on sofa, pool of blood, "
            "blood splatter on wall, holding smartphone, warm lighting"
        )
        cleaned = PromptGenerator._sanitize_gore_from_prompt(raw)
        lowered = cleaned.lower()
        assert "corpse" not in lowered
        assert "blood" not in lowered
        assert "smartphone" in lowered

    def test_apply_peaceful_scene_guard_strips_and_adds_dining(self):
        gen = self._make_gen()
        raw = (
            "a teenage girl with glasses holding a phone, living room, "
            "dead body on couch, bloody stains everywhere"
        )
        out = gen._apply_peaceful_scene_guard(raw, SEGMENT_86, SEGMENT_85).lower()
        assert "dead body on couch" not in out
        assert "bloody stains" not in out
        assert "dining room" in out or "restaurant" in out
        assert "no blood" in out
        assert "no gore" in out

    def test_generate_segment86_local_no_gore(self):
        gen = self._make_gen()
        prompt = gen.generate(SEGMENT_86, segment_index=86, prev_text=SEGMENT_85).lower()
        assert "no blood" in prompt
        assert "no gore" in prompt
        assert "dining" in prompt or "restaurant" in prompt

    @patch("src.promptgen.prompt_generator.PromptGenerator._get_llm_client")
    def test_llm_user_msg_includes_peaceful_note(self, mock_client_factory):
        from src.promptgen.prompt_generator import PromptGenerator

        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(
            content="anime scene, girl at dining table, cheerful mood"
        )
        mock_client_factory.return_value = mock_llm

        gen = PromptGenerator(
            {"style": "anime", "llm": {"provider": "openai"}, "tone": "light"}
        )
        gen._use_llm = True
        gen.generate(SEGMENT_86, segment_index=86, prev_text=SEGMENT_85)

        call = mock_llm.chat.call_args
        user_msg = call.kwargs["messages"][1]["content"]
        assert "本段无暴力/血腥描写" in user_msg
        assert "饭局" in user_msg or "餐厅" in user_msg

    def test_violent_segment_skips_peaceful_guard(self):
        gen = self._make_gen()
        violent = "无名小道上，她直接将刹车误踩成油门，撞飞了一个人，头上湿润一片。"
        raw = "car crash scene, blood on windshield"
        out = gen._apply_peaceful_scene_guard(raw, violent)
        assert "blood on windshield" in out.lower()
