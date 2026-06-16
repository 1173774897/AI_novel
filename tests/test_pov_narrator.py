"""第一人称叙述者 pov_narrator 解析测试。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.content_analyzer import ContentAnalyzerAgent, content_analyzer_node

pytestmark = pytest.mark.signature


@pytest.fixture
def minimal_config():
    return {
        "segmenter": {"method": "simple"},
        "promptgen": {},
        "llm": {},
        "agent": {"character_review": {"enabled": False}},
    }


def test_resolve_pov_from_wo_jiao_alias(minimal_config):
    text = "我走进考场。" * 30 + "……我叫王璟，这一切才开始。"
    agent = ContentAnalyzerAgent(minimal_config, budget_mode=True)
    chars = [{"name": "王璟", "desc": "男，黑框眼镜"}]
    agent._last_character_alias_map = {"我": "王璟", "答主": "王璟"}
    assert agent.resolve_pov_narrator(text, chars) == "王璟"


def test_resolve_pov_none_for_third_person(minimal_config):
    text = "张三说道：你好。李四笑道：再见。" * 20
    agent = ContentAnalyzerAgent(minimal_config, budget_mode=True)
    chars = [{"name": "张三", "desc": "男"}, {"name": "李四", "desc": "男"}]
    assert agent.resolve_pov_narrator(text, chars) is None


def test_resolve_pov_llm_fallback(minimal_config):
    text = ("我心里一紧，盯着舒然。" * 25) + "整个考场鸦雀无声。"
    agent = ContentAnalyzerAgent(minimal_config, budget_mode=False)
    chars = [
        {"name": "王璟", "desc": "叙述者"},
        {"name": "舒然", "desc": "女生"},
    ]
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(
        content='{"pov_narrator": "王璟", "reason": "全文第一人称旁白"}'
    )
    agent._llm = mock_llm
    assert agent.resolve_pov_narrator(text, chars) == "王璟"
    mock_llm.chat.assert_called_once()


@patch("src.agents.content_analyzer.SegmentTool")
@patch("src.agents.content_analyzer.ContentAnalyzerAgent.extract_characters")
@patch("src.agents.content_analyzer.ContentAnalyzerAgent.classify_genre")
def test_node_sets_pov_narrator(mock_classify, mock_extract, mock_seg_cls, minimal_config, tmp_path):
    mock_seg_cls.return_value.run.return_value = [{"text": "段1", "index": 0}]
    mock_classify.return_value = {"genre": "悬疑", "era": "现代", "confidence": 0.9}
    mock_extract.return_value = [{"name": "王璟", "desc": "男"}]

    text = ("我走进考场。" * 30) + "我叫王璟。"
    state = {
        "input_file": str(tmp_path / "n.txt"),
        "config": minimal_config,
        "workspace": str(tmp_path / "ws"),
        "budget_mode": True,
        "full_text": text,
        "decisions": [],
    }

    with patch.object(
        ContentAnalyzerAgent,
        "_build_character_alias_map",
        return_value={"我": "王璟"},
    ):
        result = content_analyzer_node(state)

    assert result.get("pov_narrator") == "王璟"
    assert any(d.get("step") == "pov_narrator" for d in result["decisions"])


@patch("src.agents.content_analyzer.SegmentTool")
@patch("src.agents.content_analyzer.ContentAnalyzerAgent.extract_characters")
@patch("src.agents.content_analyzer.ContentAnalyzerAgent.classify_genre")
def test_node_preserves_series_pov_override(
    mock_classify, mock_extract, mock_seg_cls, minimal_config, tmp_path
):
    mock_seg_cls.return_value.run.return_value = [{"text": "段1", "index": 0}]
    mock_classify.return_value = {"genre": "言情", "era": "古代", "confidence": 0.9}
    mock_extract.return_value = [{"name": "写颜", "desc": "女"}]

    state = {
        "input_file": str(tmp_path / "n.txt"),
        "config": minimal_config,
        "workspace": str(tmp_path / "ws"),
        "budget_mode": True,
        "full_text": ("我……" * 40) + "我叫李四。",
        "pov_narrator": "写颜",
        "decisions": [],
    }
    result = content_analyzer_node(state)
    assert result.get("pov_narrator") == "写颜"
    assert not any(d.get("step") == "pov_narrator" for d in result["decisions"])
