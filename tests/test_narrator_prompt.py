"""叙述者性别绑定与 prompt 注入测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.promptgen.character_tracker import CharacterTracker
from src.promptgen.narrator import (
    build_dialogue_scene_context,
    build_narrator_character_prompt,
    build_narrator_instruction,
    build_observed_character_context,
    build_quotation_speaker_context,
    build_scene_character_context,
    detect_narrator_voice,
    find_quotation_speaker,
    find_visual_focus_character,
)
from src.promptgen.prompt_generator import PromptGenerator

pytestmark = pytest.mark.signature

_SEEDED = [
    {
        "name": "答主（我）",
        "desc": "男，18岁，高三学生。戴黑框眼镜，蓝白校服，短发。",
    },
    {
        "name": "潘博文",
        "desc": "男，18岁，高三学生，戴金属细框眼镜。",
    },
]

_SEG94 = (
    "如实叙述，没考虑逻辑不逻辑，我的认知和记忆里是这样的，"
    "讲出来了，就这样。认知和记忆有没有错？我本人没法知道啊。"
)
_SEG69 = (
    "本姑娘一个字都听不懂，年龄摆在那儿呢，理解不了。"
    "读不懂就开始在课本上画小人儿，美术课是我唯一能拿优秀的科目。"
)


class TestNarratorDetection:
    def test_detect_female_for_benguniang(self):
        assert detect_narrator_voice(_SEG69) == "female"

    def test_detect_male_for_first_person_qa(self):
        assert detect_narrator_voice(_SEG94) == "male"

    def test_detect_none_for_third_person(self):
        assert detect_narrator_voice("潘博文走进教室，老师看了他一眼。") is None


class TestNarratorPromptBinding:
    def test_seg94_uses_male_protagonist_desc(self):
        desc = build_narrator_character_prompt(_SEG94, _SEEDED)
        assert "男" in desc
        assert "18岁" in desc

    def test_seg69_uses_female_default_when_no_female_seed(self):
        desc = build_narrator_character_prompt(_SEG69, _SEEDED)
        assert "女" in desc

    def test_instruction_for_seg94_requires_male(self):
        voice, instruction = build_narrator_instruction(_SEG94, _SEEDED)
        assert voice == "male"
        assert "male" in instruction
        assert "绝不可换成异性" in instruction
        assert "男" in instruction

    def test_instruction_for_seg69_requires_female(self):
        voice, instruction = build_narrator_instruction(_SEG69, _SEEDED)
        assert voice == "female"
        assert "female" in instruction
        assert "女" in instruction


class TestCharacterTrackerAllowedNames:
    def test_update_skips_unseeded_false_positive_names(self):
        tracker = CharacterTracker()
        tracker.seed_characters(_SEEDED)
        allowed = frozenset({"答主（我）", "潘博文"})
        tracker.update(
            _SEG94,
            "a young woman with long hair, a young man with glasses",
            allowed_names=allowed,
        )
        assert "我的认" not in tracker.known_characters
        assert "讲出" not in tracker.known_characters

    def test_get_character_prompt_filters_to_allowed_names(self):
        tracker = CharacterTracker()
        tracker._characters["答主（我）"] = "male desc"
        tracker._characters["我的认"] = "wrong desc"
        prompt = tracker.get_character_prompt(
            ["答主（我）", "我的认"],
            allowed_names=frozenset({"答主（我）"}),
        )
        assert prompt == "male desc"
        assert "wrong" not in prompt


_ZHOU_LING_SEEDED = [
    {
        "name": "李同学",
        "desc": "约20岁男大学生，深色休闲衬衫，短发。",
    },
    {
        "name": "周玲",
        "desc": "约20岁女生，利落黑色短发，脖子修长，五官清丽俊秀，气质偏中性帅气，穿宿舍常服，激动含泪。",
    },
]

_SEG84 = (
    "周玲显得十分激动，胸口不断起伏，眼睛仇恨地看着我，最后，眼泪终于从她脸上流下来。"
    "她激动地说：「都是你！都是你！"
)
_SEG85 = (
    "本来我和林郁好好的，如果不是你来引诱她，她不会想要跟我分手，没错，墙上的字是我刻的，"
    "我是恨你，火灾发生后，我确实是想陷害你，我巴不得你赶快从这个世界上消失，但那火，不是我放的啊……」"
)
_SEG79 = (
    "林郁和周玲听张得胜说完，脸色同时变得愤怒起来，随之林郁羞愧地低下头去，周玲见她的样子，不禁更加愤怒。"
    "我心里幸灾乐祸地冷笑了一下，暗道，果然是。张得胜转向周玲，直截了当问道：「周玲，你是不是喜欢女生？"
)
_SEG80 = (
    "你喜欢林郁？」周玲的胸口不断起伏，愤怒地与张得胜四目相对，最后口气决然地说：「是的！那又怎样？不行吗？」"
    "张得胜脸色变得缓和下来，看着周玲轻声说：「行的，我没别的意思，希望你能理解。"
)
_SEG60 = (
    "然而当我的视线不经意间转向她的右边，看到望着我的周玲脸上那极度愤怒的表情时，"
    "我不禁吓了一大跳。即使我是个傻子都可以立即看出来，周玲真的在恨我，为什么？"
)
_SEG71 = "刘丽华惊讶地点了点头，说：「是的，我之前并不认识他。」"
_SEG72 = (
    "于是张得胜把脸转向林郁：「林同学，你看到李同学后，表现出了羞愧和不安，"
    "这说明你在此之前认识他，你为什么会羞愧和不安，现在可以说出你和李同学之间的关系了吧。」"
)
_SEG81 = (
    "」随后他神情重新变得严肃，继续说：「这样，就可以解释你的床位墙壁上为什么会出现李同学的名字了，"
    "李同学，墙上的名字所指的人，确实是你。」我不由激动地站了起来，指着周玲大声说：「哦！"
)
_SEG82 = (
    "原来是你，你因为恨我与林郁交往，所以在墙上刻了我的名字，然后纵火烧坏宿舍，想陷害我。」"
    "周玲的脸色刷的一下变白，林郁激动地站了起来，说：「李昌，你别乱说，周玲不是这种人。」"
)
_FULL_CAST = _ZHOU_LING_SEEDED + [
    {"name": "张得胜", "desc": "白衬衫教授"},
    {
        "name": "刘丽华",
        "desc": "相貌平平，戴近视眼镜，沉默内向的化学系女生。",
    },
    {
        "name": "林郁",
        "desc": "长发微卷，神情羞愧尴尬，与林映洁截然不同。",
    },
    {
        "name": "林映洁",
        "desc": "白裙长发温婉女友，与林郁不是同一人。",
    },
]


class TestQuotationSpeakerBinding:
    def test_find_speaker_for_quote_continuation_seg85(self):
        speaker = find_quotation_speaker(_SEG85, _SEG84, _ZHOU_LING_SEEDED)
        assert speaker == "周玲"

    def test_find_speaker_for_seg84_zhou_ling_opens_quote(self):
        speaker = find_quotation_speaker(_SEG84, None, _ZHOU_LING_SEEDED)
        assert speaker == "周玲"

    def test_build_quotation_context_uses_zhou_ling_not_protagonist(self):
        prompt, instruction = build_quotation_speaker_context(
            _SEG85, _SEG84, _ZHOU_LING_SEEDED
        )
        assert "周玲" in instruction
        assert "短发" in prompt
        assert "不是男主李同学" in instruction

    def test_prompt_generator_seg85_binds_zhou_ling(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_ZHOU_LING_SEEDED)
        prompt, instruction = gen._build_character_context(_SEG85, prev_text=_SEG84)
        assert "周玲" in instruction
        assert "短发" in prompt
        assert "第一人称男性叙述者" not in instruction

    def test_prompt_generator_seg84_binds_zhou_ling(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_ZHOU_LING_SEEDED)
        _, instruction = gen._build_character_context(_SEG84, prev_text=None)
        assert "周玲" in instruction
        assert "第一人称男性叙述者" not in instruction
        assert "不是男主李同学" in instruction or "焦点" in instruction

    def test_seg80_dialogue_between_zhou_ling_and_zhangdesheng(self):
        seeded = _ZHOU_LING_SEEDED + [
            {
                "name": "张得胜",
                "desc": "约30岁男性，白衬衫，深蓝领带，鹰隼般锐利眼神。",
            }
        ]
        _, instruction = build_dialogue_scene_context(_SEG80, _SEG79, seeded)
        assert "张得胜" in instruction and "周玲" in instruction
        assert "对话场景" in instruction
        assert "男主李同学" in instruction
        assert "对男主" not in instruction

    def test_prompt_generator_seg80_multi_character_dialogue(self):
        seeded = _ZHOU_LING_SEEDED + [
            {
                "name": "张得胜",
                "desc": "约30岁男性，白衬衫，深蓝领带，鹰隼般锐利眼神。",
            }
        ]
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(seeded)
        _, instruction = gen._build_character_context(_SEG80, prev_text=_SEG79)
        assert "张得胜" in instruction and "周玲" in instruction
        assert "对话场景" in instruction
        assert "第一人称男性叙述者" not in instruction

    def test_seg60_focuses_on_zhou_ling_looking_at_protagonist(self):
        focus = find_visual_focus_character(_SEG60, _ZHOU_LING_SEEDED)
        assert focus == "周玲"
        _, instruction = build_observed_character_context(_SEG60, _ZHOU_LING_SEEDED)
        assert "周玲" in instruction
        assert "画面焦点" in instruction
        assert "不可把男主画成画面主体" in instruction

        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_ZHOU_LING_SEEDED)
        _, instruction = gen._build_character_context(_SEG60, prev_text=None)
        assert "周玲" in instruction
        assert "第一人称男性叙述者" not in instruction

    def test_seg71_binds_liu_lihua_not_protagonist_or_linyingjie(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_FULL_CAST)
        _, instruction = gen._build_character_context(_SEG71, prev_text=None)
        assert "刘丽华" in instruction
        assert "林映洁" not in instruction or "不可画成林映洁" in instruction
        assert "第一人称男性叙述者" not in instruction

    def test_seg72_binds_lin_yu_not_linyingjie(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_FULL_CAST)
        _, instruction = gen._build_character_context(_SEG72, prev_text=None)
        assert "林郁" in instruction
        assert "不可" in instruction and "林映洁" in instruction
        assert "第一人称男性叙述者" not in instruction

    def test_seg81_focuses_on_zhou_ling_not_dual_dialogue(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_FULL_CAST)
        _, instruction = gen._build_character_context(_SEG81, prev_text=_SEG80)
        assert "周玲" in instruction
        assert "短发" in instruction
        assert "指着" in instruction or "焦点" in instruction
        assert "李同学与周玲" not in instruction
        assert "林映洁" in instruction

    def test_seg82_focuses_on_zhou_ling_reaction_not_lin_yu_dialogue(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_FULL_CAST)
        _, instruction = gen._build_character_context(_SEG82, prev_text=_SEG81)
        assert "周玲" in instruction
        assert "短发" in instruction
        assert "脸色" in instruction or "反应" in instruction
        assert "周玲与林郁" not in instruction
        assert "林映洁" in instruction

    def test_seg79_keeps_protagonist_narration_not_zhou_ling_solo(self):
        seeded = _ZHOU_LING_SEEDED + [
            {"name": "张得胜", "desc": "教授"},
        ]
        _, instruction = build_scene_character_context(_SEG79, None, seeded)
        assert instruction == ""


class TestPromptGeneratorNarratorContext:
    def test_build_character_context_injects_male_for_seg94(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_SEEDED)
        prompt, instruction = gen._build_character_context(_SEG94)
        assert "男" in prompt
        assert "male" in instruction

    def test_build_character_context_injects_female_for_seg69(self):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_SEEDED)
        prompt, instruction = gen._build_character_context(_SEG69)
        # 仅有男性答主预填时，绑定答主外观而非默认女性占位
        assert "男" in prompt
        assert "答主（我）" in instruction

    @patch.object(PromptGenerator, "_detect_llm_available", return_value=False)
    def test_local_generate_includes_narrator_desc_for_seg94(self, _mock_llm):
        gen = PromptGenerator({"character_tracking": True, "style": "anime"})
        gen.seed_characters(_SEEDED)
        prompt = gen.generate(_SEG94, segment_index=94)
        assert "男" in prompt
        assert "我的认" not in gen.character_tracker.known_characters

    @patch.object(PromptGenerator, "_detect_llm_available", return_value=True)
    @patch.object(PromptGenerator, "_get_llm_client")
    def test_llm_user_message_includes_narrator_instruction(
        self, mock_client_factory, _mock_llm
    ):
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            content="a young man with glasses, sitting at a desk"
        )
        mock_client_factory.return_value = mock_client

        gen = PromptGenerator(
            {
                "character_tracking": True,
                "style": "anime",
                "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            }
        )
        gen.seed_characters(_SEEDED)
        gen.generate(_SEG94, segment_index=94)

        user_msg = mock_client.chat.call_args.kwargs["messages"][1]["content"]
        assert "叙述者为男性角色「答主（我）」" in user_msg
        assert "male" in user_msg
        assert "男，18岁" in user_msg
