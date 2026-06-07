"""LLM 配置解析测试。"""

import pytest

from src.llm.llm_client import _resolve_llm_model


@pytest.mark.signature
def test_deepseek_replaces_gpt_model():
    cfg = _resolve_llm_model(
        "deepseek",
        {"provider": "deepseek", "model": "gpt-4o-mini"},
    )
    assert cfg["model"] == "deepseek-chat"


@pytest.mark.signature
def test_openai_keeps_gpt_model():
    cfg = _resolve_llm_model(
        "openai",
        {"provider": "openai", "model": "gpt-4o-mini"},
    )
    assert cfg["model"] == "gpt-4o-mini"
