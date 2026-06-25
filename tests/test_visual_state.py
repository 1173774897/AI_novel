"""分段版本化角色外观测试。"""

import pytest

from src.promptgen.character_tracker import CharacterTracker
from src.promptgen.visual_state import (
    attach_visual_states,
    find_anchor_segment,
    format_segment_context,
    is_effective_at_segment,
    normalize_visual_states,
    pick_visual_state,
    plan_visual_states_by_rules,
    prune_static_visual_states,
    resolve_character_desc,
    split_mixed_phase_desc,
)
from src.promptgen.visual_state_planner import plan_visual_states


@pytest.mark.signature
class TestVisualStateCore:
    def test_is_effective_at_segment_boundaries(self):
        state = {
            "effective_from_segment": 5,
            "deprecated_at_segment": 10,
        }
        assert not is_effective_at_segment(state, 4)
        assert is_effective_at_segment(state, 5)
        assert is_effective_at_segment(state, 9)
        assert not is_effective_at_segment(state, 10)

    def test_pick_visual_state_prefers_latest_effective_from(self):
        states = [
            {"id": "a", "desc": "早期", "effective_from_segment": 0, "deprecated_at_segment": 8},
            {"id": "b", "desc": "后期", "effective_from_segment": 8},
        ]
        assert pick_visual_state(states, 3)["desc"] == "早期"
        assert pick_visual_state(states, 8)["desc"] == "后期"

    def test_resolve_character_desc_fallback_to_entry_desc(self):
        entry = {"name": "温冉", "desc": "默认描述"}
        assert resolve_character_desc(entry, 0) == "默认描述"

    def test_resolve_character_desc_fallback_when_state_expired(self):
        entry = {
            "name": "李斯",
            "desc": "闺蜜固定外观",
            "visual_states": [
                {
                    "id": "default",
                    "desc": "仅前段",
                    "effective_from_segment": 0,
                    "deprecated_at_segment": 124,
                }
            ],
        }
        assert resolve_character_desc(entry, 50) == "仅前段"
        assert resolve_character_desc(entry, 150) == "闺蜜固定外观"
        entry = {
            "name": "温冉",
            "desc": "混写描述",
            "visual_states": [
                {"id": "early", "desc": "孕肚微隆", "effective_from_segment": 0, "deprecated_at_segment": 12},
                {"id": "late", "desc": "身形清瘦", "effective_from_segment": 12},
            ],
        }
        assert "孕肚" in resolve_character_desc(entry, 5)
        assert "清瘦" in resolve_character_desc(entry, 12)

    def test_normalize_visual_states_drops_empty_desc(self):
        assert normalize_visual_states([{"desc": ""}, {"desc": "有效"}]) == [
            {"id": "state_0", "desc": "有效"}
        ]

    def test_split_mixed_phase_desc(self):
        desc = (
            "女，约22岁，黑色长发，前期怀孕约三个月小腹微隆，"
            "后期身形清瘦面色苍白"
        )
        split = split_mixed_phase_desc(desc)
        assert split is not None
        early, late = split
        assert "微隆" in early
        assert "清瘦" in late
        assert "后期" not in early
        assert "前期" not in late

    def test_find_anchor_segment(self):
        segments = [
            {"index": 0, "text": "她告诉男友怀孕了"},
            {"index": 11, "text": "明天就要去做人流手术"},
            {"index": 12, "text": "手术后的她"},
        ]
        assert find_anchor_segment(segments, ("人流",)) == 11

    def test_format_segment_context_includes_total_count(self):
        segments = [{"index": i, "text": f"段{i}"} for i in range(170)]
        ctx = format_segment_context(segments, ("约了手术",), summary_max_lines=5)
        assert "共 170 个分段" in ctx
        assert "不是小说正文里的章节标记" in ctx

    def test_plan_visual_states_by_rules_mixed_desc(self):
        characters = [
            {
                "name": "温冉",
                "desc": "女，22岁，前期怀孕小腹微隆，后期身形清瘦",
            }
        ]
        segments = [
            {"index": i, "text": "日常" if i < 10 else "她约了人流手术"}
            for i in range(20)
        ]
        result = plan_visual_states_by_rules(characters, segments)
        assert "温冉" in result
        states = result["温冉"]
        assert len(states) == 2
        assert states[0]["effective_from_segment"] == 0
        assert states[1]["effective_from_segment"] == 10

    def test_attach_visual_states(self):
        chars = [{"name": "温冉", "desc": "x"}]
        merged = attach_visual_states(chars, {"温冉": [{"id": "a", "desc": "y"}]})
        assert merged[0]["visual_states"][0]["desc"] == "y"

    def test_prune_static_visual_states_drops_unchanged_character(self):
        chars = [
            {"name": "温冉", "desc": "前期怀孕，后期清瘦"},
            {"name": "李斯", "desc": "齐耳短发闺蜜，无变化"},
        ]
        states = {
            "温冉": [
                {"id": "early", "desc": "孕", "effective_from_segment": 0, "deprecated_at_segment": 124},
                {"id": "late", "desc": "瘦", "effective_from_segment": 124},
            ],
            "李斯": [
                {"id": "default", "desc": "闺蜜", "effective_from_segment": 0, "deprecated_at_segment": 124},
            ],
        }
        pruned = prune_static_visual_states(chars, states)
        assert "温冉" in pruned
        assert "李斯" not in pruned

    def test_attach_clears_visual_states_when_pruned(self):
        chars = [{"name": "李斯", "desc": "闺蜜", "visual_states": [{"id": "old", "desc": "x"}]}]
        out = attach_visual_states(chars, {})
        assert "visual_states" not in out[0]


@pytest.mark.signature
class TestCharacterTrackerVisualStates:
    def test_get_desc_switches_by_segment(self):
        tracker = CharacterTracker()
        tracker.seed_characters([
            {
                "name": "温冉",
                "desc": "混写",
                "visual_states": [
                    {"id": "e", "desc": "早期孕相", "effective_from_segment": 0, "deprecated_at_segment": 5},
                    {"id": "l", "desc": "后期清瘦", "effective_from_segment": 5},
                ],
            }
        ])
        assert tracker.get_desc("温冉", 2) == "早期孕相"
        assert tracker.get_desc("温冉", 5) == "后期清瘦"

    def test_get_character_prompt_uses_segment_index(self):
        tracker = CharacterTracker()
        tracker.seed_characters([
            {
                "name": "温冉",
                "desc": "混写",
                "visual_states": [
                    {"id": "e", "desc": "早期孕相", "effective_from_segment": 0, "deprecated_at_segment": 5},
                    {"id": "l", "desc": "后期清瘦", "effective_from_segment": 5},
                ],
            }
        ])
        early = tracker.get_character_prompt(["温冉"], segment_index=1)
        late = tracker.get_character_prompt(["温冉"], segment_index=6)
        assert early == "早期孕相"
        assert late == "后期清瘦"

    def test_to_dict_roundtrip_visual_states(self):
        tracker = CharacterTracker()
        tracker.seed_characters([
            {
                "name": "温冉",
                "desc": "base",
                "visual_states": [
                    {"id": "e", "desc": "早期", "effective_from_segment": 0, "deprecated_at_segment": 3},
                ],
            }
        ])
        restored = CharacterTracker()
        restored.from_dict(tracker.to_dict())
        assert restored.get_desc("温冉", 0) == "早期"


@pytest.mark.signature
class TestPromptGeneratorVisualStates:
    def _make_gen(self):
        from src.promptgen.prompt_generator import PromptGenerator

        gen = PromptGenerator({"style": "anime", "llm": {"provider": "none"}})
        gen.seed_characters([
            {
                "name": "温冉",
                "desc": "混写",
                "visual_states": [
                    {"id": "e", "desc": "早期孕相明显", "effective_from_segment": 0, "deprecated_at_segment": 5},
                    {"id": "l", "desc": "后期清瘦苍白", "effective_from_segment": 5},
                ],
            }
        ])
        return gen

    def test_cast_bible_switches_by_segment(self):
        gen = self._make_gen()
        text = "温冉坐在床边"
        early = gen._build_cast_bible(text, segment_index=1)
        late = gen._build_cast_bible(text, segment_index=6)
        assert "早期孕相" in early
        assert "后期清瘦" in late


@pytest.mark.signature
class TestVisualStatePlanner:
    def test_budget_mode_uses_rules_only(self):
        characters = [
            {"name": "温冉", "desc": "女，前期怀孕微隆，后期清瘦"},
        ]
        segments = [{"index": i, "text": "人流" if i == 8 else "日常"} for i in range(16)]
        result, discussion = plan_visual_states(characters, segments, llm=None, budget_mode=True)
        assert "温冉" in result
        assert discussion == []
