"""双 AI 角色讨论审核测试。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.character_reviewer import (
    character_review_enabled,
    run_character_review_discussion,
)
from src.agents.content_analyzer import ContentAnalyzerAgent

pytestmark = pytest.mark.signature


def test_character_review_enabled_budget_off():
    assert not character_review_enabled({"agent": {"character_review": {"enabled": True}}}, True)


def test_character_review_enabled_default_on():
    assert character_review_enabled({}, False)


def test_character_review_explicit_disabled():
    cfg = {"agent": {"character_review": {"enabled": False}}}
    assert not character_review_enabled(cfg, False)


def test_discussion_three_rounds_final_list():
    primary = MagicMock()
    reviewer = MagicMock()

    draft = [{"name": "张三", "desc": "约30岁男性，黑色短发"}]
    review_json = {
        "overall_comment": "缺少李四",
        "issues": [{"type": "add", "name": "李四", "detail": "有对话戏份"}],
    }
    response_json = {
        "reply": "同意补入李四",
        "characters": [
            {"name": "张三", "desc": "约30岁男性，黑色短发，穿青衫"},
            {"name": "李四", "desc": "约28岁男性，黑色束发，魁梧"},
        ],
    }
    final_json = {
        "consensus_note": "保留张三李四",
        "characters": response_json["characters"],
    }

    reviewer.chat.side_effect = [
        MagicMock(content=__import__("json").dumps(review_json, ensure_ascii=False)),
        MagicMock(content=__import__("json").dumps(final_json, ensure_ascii=False)),
    ]
    primary.chat.return_value = MagicMock(
        content=__import__("json").dumps(response_json, ensure_ascii=False)
    )

    result = run_character_review_discussion(
        "张三说道你好。李四笑道再见。",
        draft,
        primary_llm=primary,
        reviewer_llm=reviewer,
        reviewer_provider="gemini",
    )

    assert len(result.characters) == 2
    assert result.characters[1]["name"] == "李四"
    assert len(result.discussion) >= 2
    assert reviewer.chat.call_count == 2
    assert primary.chat.call_count == 1


def test_discussion_reviewer_fail_keeps_draft():
    primary = MagicMock()
    reviewer = MagicMock()
    reviewer.chat.side_effect = RuntimeError("api down")
    draft = [{"name": "王五", "desc": "短发"}]

    result = run_character_review_discussion(
        "王五说道。",
        draft,
        primary_llm=primary,
        reviewer_llm=reviewer,
    )

    assert result.characters == draft
    assert primary.chat.call_count == 0


@patch("src.agents.content_analyzer.ContentAnalyzerAgent._extract_characters_by_llm")
@patch("src.agents.content_analyzer.ContentAnalyzerAgent._supplement_discovered_characters")
@patch("src.agents.content_analyzer.ContentAnalyzerAgent._review_characters_with_discussion")
def test_extract_characters_calls_review(mock_review, mock_supp, mock_llm, minimal_config):
    cfg = {**minimal_config, "agent": {"character_review": {"enabled": True}}}
    mock_llm.return_value = [{"name": "张三", "desc": "desc"}]
    mock_supp.side_effect = lambda chars, *_a, **_k: chars
    mock_review.side_effect = lambda _text, chars, **_k: chars

    agent = ContentAnalyzerAgent(cfg, budget_mode=False)
    agent.extract_characters("张三说道。")

    mock_review.assert_called_once()


def test_extract_characters_skips_review_in_budget_mode(minimal_config):
    cfg = {**minimal_config, "agent": {"character_review": {"enabled": True}}}
    agent = ContentAnalyzerAgent(cfg, budget_mode=True)
    with patch.object(agent, "_extract_characters_by_rules", return_value=[]):
        with patch(
            "src.agents.character_reviewer.run_character_review_discussion"
        ) as mock_run:
            agent.extract_characters("无对话文本")
            mock_run.assert_not_called()


@pytest.fixture
def minimal_config():
    return {
        "segmenter": {"method": "simple"},
        "promptgen": {},
        "llm": {"provider": "deepseek"},
        "agent": {"character_review": {"enabled": True}},
    }