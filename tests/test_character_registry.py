"""CharacterRegistry 跨集角色一致性测试。"""
from __future__ import annotations

import json

import pytest

from src.promptgen.character_registry import CharacterRegistry
from src.promptgen.prompt_generator import PromptGenerator

pytestmark = pytest.mark.signature


class TestCharacterRegistry:
    def test_merge_does_not_overwrite_existing_desc(self, tmp_path):
        reg = CharacterRegistry(tmp_path / "registry.json")
        reg.characters["写颜"] = {"name": "写颜", "desc": "canonical 外观", "episodes": []}
        updated = reg.merge_character_list(
            [{"name": "写颜", "desc": "本集新描述应被忽略"}],
            episode="ep02",
        )
        assert updated == 0
        assert reg.characters["写颜"]["desc"] == "canonical 外观"
        assert "ep02" in reg.characters["写颜"]["episodes"]

    def test_merge_fills_empty_desc(self, tmp_path):
        reg = CharacterRegistry(tmp_path / "registry.json")
        reg.characters["周临"] = {"name": "周临", "desc": "", "episodes": ["ep01"]}
        updated = reg.merge_character_list(
            [{"name": "周临", "desc": "青年男子，玄色长袍"}],
            episode="ep02",
        )
        assert updated == 1
        assert reg.characters["周临"]["desc"] == "青年男子，玄色长袍"

    def test_apply_canonical_overrides_episode_extract(self, tmp_path):
        reg = CharacterRegistry(tmp_path / "registry.json")
        reg.merge_character_list(
            [{"name": "卓纳林", "desc": "北疆王子，高马尾，银甲"}],
            episode="ep01",
        )
        merged = reg.apply_canonical_to([
            {"name": "卓纳林", "desc": "本集 LLM 乱写的描述"},
            {"name": "写颜", "desc": "仅本集出现"},
        ])
        by_name = {c["name"]: c["desc"] for c in merged}
        assert by_name["卓纳林"] == "北疆王子，高马尾，银甲"
        assert by_name["写颜"] == "仅本集出现"

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "character_registry.json"
        reg = CharacterRegistry(path)
        reg.merge_character_list([{"name": "写颜", "desc": "少女侍女，青衣"}])
        reg.save()

        loaded = CharacterRegistry.load(path)
        assert loaded.characters["写颜"]["desc"] == "少女侍女，青衣"
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1

    def test_to_seed_list_sorted(self, tmp_path):
        reg = CharacterRegistry(tmp_path / "r.json")
        reg.merge_character_list([
            {"name": "周临", "desc": "desc b"},
            {"name": "写颜", "desc": "desc a"},
            {"name": "无名", "desc": ""},
        ])
        seeds = reg.to_seed_list()
        assert [s["name"] for s in seeds] == ["写颜", "周临"]


class TestPovNarratorOverride:
    def test_set_pov_narrator_forces_author_pov(self):
        gen = PromptGenerator({"character_tracking": True, "pov_mode": "auto"})
        gen.seed_characters([{"name": "写颜", "desc": "青衣侍女，少女"}])
        gen.set_pov_narrator("写颜")
        assert gen._uses_author_pov("第三人称段落") is True

    def test_set_pov_without_desc_does_not_force_author_pov(self):
        gen = PromptGenerator({"character_tracking": True, "pov_mode": "auto"})
        gen.seed_characters([{"name": "写颜", "desc": ""}])
        gen.set_pov_narrator("写颜")
        assert gen._uses_author_pov("我走进房间。") is False

    def test_narrator_cast_names_uses_override(self):
        gen = PromptGenerator({"character_tracking": True})
        gen.seed_characters([{"name": "写颜", "desc": "青衣侍女"}])
        gen.set_pov_narrator("写颜")
        names = gen._narrator_cast_names("无关文本")
        assert names == ["写颜"]

    def test_canonical_seed_overwrites_tracker(self):
        gen = PromptGenerator({"character_tracking": True})
        gen.seed_characters([{"name": "写颜", "desc": "旧描述"}])
        gen.seed_characters([{"name": "写颜", "desc": "系列 canonical"}], canonical=True)
        assert gen.character_tracker.known_characters["写颜"] == "系列 canonical"
