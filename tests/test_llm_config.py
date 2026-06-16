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
def test_dashscope_keeps_qwen_model():
    cfg = _resolve_llm_model(
        "dashscope",
        {"provider": "dashscope", "model": "qwen3.7-max"},
    )
    assert cfg["model"] == "qwen3.7-max"


@pytest.mark.signature
def test_create_dashscope_client(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    from src.llm.llm_client import create_llm_client

    client = create_llm_client(
        {"provider": "dashscope", "model": "qwen-plus", "api_key": "sk-test"}
    )
    assert client._model == "qwen-plus"
    assert client._base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


@pytest.mark.signature
def test_openai_keeps_gpt_model():
    cfg = _resolve_llm_model(
        "openai",
        {"provider": "openai", "model": "gpt-4o-mini"},
    )
    assert cfg["model"] == "gpt-4o-mini"
