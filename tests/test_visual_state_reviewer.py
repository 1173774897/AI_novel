"""visual_states 双 AI 审核测试。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.promptgen.visual_state_reviewer import (
    run_visual_state_review_discussion,
    visual_state_review_enabled,
    _preserve_segment_bounds,
)


@pytest.mark.signature
class TestVisualStateReviewEnabled:
    def test_follows_character_review_by_default(self):
        cfg = {"agent": {"character_review": {"enabled": True}}}
        assert visual_state_review_enabled(cfg, budget_mode=False) is True

    def test_disabled_in_budget_mode(self):
        cfg = {"agent": {"character_review": {"enabled": True}}}
        assert visual_state_review_enabled(cfg, budget_mode=True) is False

    def test_explicit_disable(self):
        cfg = {
            "agent": {
                "character_review": {"enabled": True},
                "visual_state_review": {"enabled": False},
            }
        }
        assert visual_state_review_enabled(cfg, budget_mode=False) is False


@pytest.mark.signature
class TestPreserveSegmentBounds:
    def test_backfills_missing_segment_fields(self):
        draft = {
            "温冉": [
                {
                    "id": "early",
                    "desc": "旧早期",
                    "effective_from_segment": 0,
                    "deprecated_at_segment": 124,
                },
                {"id": "late", "desc": "旧后期", "effective_from_segment": 124},
            ]
        }
        reviewed = {
            "温冉": [
                {"id": "early", "desc": "新早期"},
                {"id": "late", "desc": "新后期"},
            ]
        }
        merged = _preserve_segment_bounds(reviewed, draft)
        assert merged["温冉"][0]["effective_from_segment"] == 0
        assert merged["温冉"][0]["deprecated_at_segment"] == 124
        assert merged["温冉"][1]["effective_from_segment"] == 124
    def test_overwrites_llm_wrong_segment_index(self):
        draft = {
            "温冉": [
                {
                    "id": "early",
                    "desc": "旧",
                    "effective_from_segment": 0,
                    "deprecated_at_segment": 124,
                },
                {"id": "late", "desc": "旧", "effective_from_segment": 124},
            ]
        }
        reviewed = {
            "温冉": [
                {
                    "id": "early",
                    "desc": "新早期",
                    "effective_from_segment": 0,
                    "deprecated_at_segment": 14,
                },
                {
                    "id": "late",
                    "desc": "新后期",
                    "effective_from_segment": 14,
                    "deprecated_at_segment": 124,
                },
            ]
        }
        merged = _preserve_segment_bounds(reviewed, draft)
        assert merged["温冉"][0]["deprecated_at_segment"] == 124
        assert merged["温冉"][1]["effective_from_segment"] == 124
        assert merged["温冉"][0]["desc"] == "新早期"


@pytest.mark.signature
class TestVisualStateReviewDiscussion:
    def _mock_llm(self, payloads: list[dict]):
        llm = MagicMock()
        responses = [MagicMock(content=json.dumps(p, ensure_ascii=False)) for p in payloads]
        llm.chat.side_effect = responses
        return llm

    def test_reviewer_finalizes_desc_without_changing_pivot(self):
        draft = {
            "温冉": [
                {
                    "id": "early",
                    "desc": "重复重复，怀孕微隆又清瘦",
                    "effective_from_segment": 0,
                    "deprecated_at_segment": 144,
                },
                {
                    "id": "late",
                    "desc": "重复重复，清瘦又怀孕",
                    "effective_from_segment": 144,
                },
            ]
        }
        reviewer = self._mock_llm([
            {
                "overall_comment": "desc 重复且互斥不足",
                "issues": [
                    {
                        "name": "温冉",
                        "state_id": "early",
                        "type": "contradiction",
                        "detail": "早期含清瘦",
                    }
                ],
            },
            {
                "consensus_note": "已去重互斥",
                "visual_states": {
                    "温冉": [
                        {
                            "id": "early",
                            "desc": "女，22岁，黑色长发，孕三月小腹微隆，宽松浅色衣裙",
                            "effective_from_segment": 0,
                            "deprecated_at_segment": 144,
                        },
                        {
                            "id": "late",
                            "desc": "女，22岁，黑色长发，身形清瘦，神态疏离",
                            "effective_from_segment": 144,
                        },
                    ]
                },
            },
        ])
        primary = self._mock_llm([
            {
                "reply": "接受审核意见",
                "visual_states": {
                    "温冉": [
                        {
                            "id": "early",
                            "desc": "女，22岁，黑色长发，孕三月小腹微隆",
                        },
                        {
                            "id": "late",
                            "desc": "女，22岁，黑色长发，身形清瘦",
                        },
                    ]
                },
            }
        ])

        segments = [
            {"index": i, "text": f"段{i}" + ("约了手术" if i == 124 else "")}
            for i in range(170)
        ]

        result = run_visual_state_review_discussion(
            "她约了手术。三个月后做完人流再见。",
            [{"name": "温冉", "desc": "前期怀孕后期清瘦"}],
            draft,
            segments,
            primary_llm=primary,
            reviewer_llm=reviewer,
            reviewer_provider="dashscope",
        )

        assert "清瘦" in result.visual_states["温冉"][1]["desc"]
        assert "孕" in result.visual_states["温冉"][0]["desc"]
        assert result.visual_states["温冉"][0]["deprecated_at_segment"] == 144
        assert any("共识" in line for line in result.discussion)

    def test_fallback_to_draft_on_reviewer_failure(self):
        draft = {
            "温冉": [
                {"id": "early", "desc": "a", "effective_from_segment": 0, "deprecated_at_segment": 5},
            ]
        }
        reviewer = MagicMock()
        reviewer.chat.side_effect = RuntimeError("api down")
        primary = MagicMock()

        result = run_visual_state_review_discussion(
            "text",
            [{"name": "温冉", "desc": "x"}],
            draft,
            [{"index": 0, "text": "x"}],
            primary_llm=primary,
            reviewer_llm=reviewer,
        )
        assert result.visual_states == draft
        assert primary.chat.call_count == 0
