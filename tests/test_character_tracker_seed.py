"""CharacterTracker / PromptGen 角色预填测试。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.promptgen.character_tracker import CharacterTracker
from src.promptgen.prompt_generator import PromptGenerator
from src.tools.prompt_gen_tool import PromptGenTool
from src.agents.art_director import ArtDirectorAgent, art_director_node
from src.agents.content_analyzer import _log_character_descriptions
from src.agents.utils import make_decision

pytestmark = pytest.mark.signature


class TestCharacterTrackerSeed:
    def test_seed_characters_writes_new_entries(self):
        tracker = CharacterTracker()
        seeded = tracker.seed_characters([
            {"name": "张三", "desc": "年轻男子，短发，穿外卖制服"},
            {"name": "李四", "desc": "年轻女子，长发，穿睡衣"},
        ])
        assert seeded == 2
        assert tracker.known_characters["张三"] == "年轻男子，短发，穿外卖制服"
        assert tracker.known_characters["李四"] == "年轻女子，长发，穿睡衣"

    def test_seed_skips_empty_desc_or_name(self):
        tracker = CharacterTracker()
        seeded = tracker.seed_characters([
            {"name": "", "desc": "有描述无名字"},
            {"name": "王五", "desc": ""},
            {"name": "  ", "desc": "   "},
            "not-a-dict",
            None,
        ])
        assert seeded == 0
        assert tracker.known_characters == {}

    def test_seed_does_not_overwrite_existing(self):
        tracker = CharacterTracker()
        tracker._characters["张三"] = "已有描述"
        seeded = tracker.seed_characters([
            {"name": "张三", "desc": "新描述应被忽略"},
            {"name": "李四", "desc": "可写入"},
        ])
        assert seeded == 1
        assert tracker.known_characters["张三"] == "已有描述"
        assert tracker.known_characters["李四"] == "可写入"

    def test_seed_canonical_overwrites(self):
        tracker = CharacterTracker()
        tracker._characters["张三"] = "已有描述"
        seeded = tracker.seed_characters(
            [{"name": "张三", "desc": "canonical 覆盖"}],
            canonical=True,
        )
        assert seeded == 1
        assert tracker.known_characters["张三"] == "canonical 覆盖"

    def test_seeded_desc_used_in_get_character_prompt(self):
        tracker = CharacterTracker()
        tracker.seed_characters([
            {"name": "张三", "desc": "a tall young man in delivery uniform"},
        ])
        prompt = tracker.get_character_prompt(["张三"])
        assert "a tall young man in delivery uniform" in prompt

    def test_resolve_segment_characters_substring_match(self):
        """子串匹配应覆盖「张得胜专注地…」等正则漏识别写法。"""
        tracker = CharacterTracker()
        tracker.seed_characters([
            {"name": "张得胜", "desc": "心理学教授，白衬衫"},
        ])
        text = "说完，张得胜专注地审视着我的脸，继续显示他的权威说"
        names = tracker.resolve_segment_characters(
            text,
            seeded_names=["张得胜"],
            allowed_names=frozenset({"张得胜"}),
        )
        assert names == ["张得胜"]

    def test_resolve_segment_characters_introduction_line(self):
        text = "那人自我介绍说他叫张得胜，是S大心理学系的教授"
        tracker = CharacterTracker()
        tracker.seed_characters([
            {"name": "张得胜", "desc": "心理学教授"},
        ])
        names = tracker.resolve_segment_characters(
            text,
            seeded_names=["张得胜"],
            allowed_names=frozenset({"张得胜"}),
        )
        assert names == ["张得胜"]

    def test_resolve_segment_characters_filters_non_seeded_regex_noise(self):
        tracker = CharacterTracker()
        tracker.seed_characters([
            {"name": "张得胜", "desc": "心理学教授"},
        ])
        text = "我走出审问室，张得胜跟在我身后出来"
        names = tracker.resolve_segment_characters(
            text,
            seeded_names=["张得胜"],
            allowed_names=frozenset({"张得胜"}),
        )
        assert names == ["张得胜"]
        assert "我走" not in names

    def test_update_does_not_overwrite_seeded_desc(self):
        tracker = CharacterTracker()
        tracker.seed_characters([
            {"name": "张三", "desc": "seeded appearance"},
        ])
        tracker.update(
            "张三说道：你好。",
            "a young woman in red dress, smiling",
        )
        assert tracker.known_characters["张三"] == "seeded appearance"

    def test_to_dict_roundtrip_preserves_seeded(self):
        tracker = CharacterTracker()
        tracker.seed_characters([{"name": "张三", "desc": "desc"}])
        data = tracker.to_dict()
        restored = CharacterTracker()
        restored.from_dict(data)
        assert restored.known_characters == {"张三": "desc"}


class TestPromptGeneratorSeed:
    def test_build_character_context_includes_cast_bible(self):
        gen = PromptGenerator({"character_tracking": True})
        gen.seed_characters([
            {"name": "张得胜", "desc": "约30岁男性，白衬衫，鹰眼"},
            {"name": "李同学", "desc": "约20岁男大学生"},
        ])
        context, _ = gen._build_character_context(
            "张得胜穿着白衬衫打着领带，身材健硕。"
        )
        assert "【本段相关角色，外观保持一致】" in context
        assert "张得胜：约30岁男性，白衬衫，鹰眼" in context
        assert "李同学" not in context

    def test_build_character_context_tags_present_cast_in_segment(self):
        gen = PromptGenerator({"character_tracking": True})
        gen.seed_characters([
            {"name": "张得胜", "desc": "约30岁男性，白衬衫，鹰眼"},
        ])
        context, _ = gen._build_character_context(
            "张得胜展开手里的本子，看我，说：「死者邓紫你认识吗？」"
        )
        assert "【本段出场角色】" in context
        assert "约30岁男性，白衬衫，鹰眼" in context

    def test_seed_characters_disabled_when_tracking_off(self):
        gen = PromptGenerator({"character_tracking": False})
        assert gen.seed_characters([{"name": "张三", "desc": "x"}]) == 0

    def test_seed_characters_delegates_to_tracker(self):
        gen = PromptGenerator({"character_tracking": True})
        seeded = gen.seed_characters([
            {"name": "张三", "desc": "短发男子"},
        ])
        assert seeded == 1
        assert gen.character_tracker is not None
        assert gen.character_tracker.known_characters["张三"] == "短发男子"


class TestPromptGenToolSeed:
    def test_seed_characters_via_tool(self):
        tool = PromptGenTool({"promptgen": {"character_tracking": True}})
        seeded = tool.seed_characters([{"name": "李四", "desc": "长发女子"}])
        assert seeded == 1
        assert tool._get_gen().character_tracker.known_characters["李四"] == "长发女子"


class TestCharacterDescriptionLogging:
    def test_log_character_descriptions_skips_empty_desc(self, caplog):
        import logging

        caplog.set_level(logging.INFO)
        _log_character_descriptions([
            {"name": "张三", "desc": "短发外卖员"},
            {"name": "李四", "desc": ""},
            {"name": "", "desc": "无名字"},
        ])
        messages = [r.message for r in caplog.records]
        assert any("角色 张三: 短发外卖员" in m for m in messages)
        assert not any("李四" in m for m in messages)


class TestArtDirectorSeedIntegration:
    @patch("src.agents.art_director.ImageGenTool")
    @patch("src.agents.art_director.PromptGenTool")
    def test_config_style_overrides_suggested_realistic(
        self, mock_prompt_cls, mock_img_cls, tmp_path
    ):
        mock_pg = mock_prompt_cls.return_value
        mock_pg.run.return_value = "prompt"
        mock_pg.seed_characters.return_value = 0
        mock_img_cls.return_value.run.return_value = "img.png"

        state = {
            "config": {
                "promptgen": {"style": "anime"},
                "imagegen": {"backend": "together"},
            },
            "workspace": str(tmp_path),
            "budget_mode": True,
            "segments": [{"text": "我在教室写字。", "index": 0}],
            "characters": [],
            "suggested_style": "realistic",
            "decisions": [],
            "completed_nodes": [],
        }
        with patch.object(
            ArtDirectorAgent,
            "generate_image",
            return_value=(tmp_path / "0000.png", -1.0, 0, [
                make_decision("ArtDirector", "image_seg0", "ok", "ok"),
            ]),
        ):
            art_director_node(state)
        mock_pg.set_style.assert_called_once_with("anime")

    @patch("src.agents.art_director.ImageGenTool")
    @patch("src.agents.art_director.PromptGenTool")
    def test_art_director_node_seeds_characters_from_state(
        self, mock_prompt_cls, mock_img_cls, tmp_path
    ):
        mock_prompt = MagicMock()
        mock_prompt.seed_characters.return_value = 2
        mock_prompt.run.return_value = "prompt"
        mock_prompt_cls.return_value = mock_prompt

        state = {
            "config": {"promptgen": {}, "imagegen": {"backend": "together"}},
            "workspace": str(tmp_path),
            "budget_mode": True,
            "segments": [{"text": "段1", "index": 0}],
            "characters": [
                {"name": "张三", "desc": "短发外卖员"},
                {"name": "李四", "desc": "长发女子"},
            ],
        }

        with patch.object(
            ArtDirectorAgent,
            "generate_image",
            return_value=(tmp_path / "0000.png", -1.0, 0, [
                make_decision("ArtDirector", "image_seg0", "ok", "ok"),
            ]),
        ):
            result = art_director_node(state)

        mock_prompt.seed_characters.assert_called_once_with(state["characters"])
        seed_steps = [
            d for d in result["decisions"]
            if d.get("step") == "seed_characters"
        ]
        assert len(seed_steps) == 1
        assert "预填 2 个角色" in seed_steps[0]["decision"]

    @patch("src.agents.art_director.ImageGenTool")
    @patch("src.agents.art_director.PromptGenTool")
    def test_art_director_node_skips_seed_when_no_desc(
        self, mock_prompt_cls, mock_img_cls, tmp_path
    ):
        mock_prompt = MagicMock()
        mock_prompt.seed_characters.return_value = 0
        mock_prompt_cls.return_value = mock_prompt

        state = {
            "config": {"promptgen": {}, "imagegen": {"backend": "together"}},
            "workspace": str(tmp_path),
            "budget_mode": True,
            "segments": [],
            "characters": [{"name": "张三", "desc": ""}],
        }

        result = art_director_node(state)

        mock_prompt.seed_characters.assert_called_once_with(state["characters"])
        seed_steps = [
            d for d in result["decisions"]
            if d.get("step") == "seed_characters"
        ]
        assert seed_steps == []
