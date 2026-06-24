"""LLM prompt 失败时须中断，不可回退本地规则模式。"""

from unittest.mock import MagicMock, patch

import pytest

from src.promptgen.prompt_generator import PromptGenerationError, PromptGenerator

pytestmark = pytest.mark.signature


class TestPromptLlmFailFast:
    @patch.object(PromptGenerator, "_detect_llm_available", return_value=True)
    @patch.object(PromptGenerator, "_get_llm_client")
    def test_generate_raises_on_llm_api_error(self, mock_client_factory, _mock_llm):
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("rate limit")
        mock_client_factory.return_value = mock_client

        gen = PromptGenerator(
            {
                "character_tracking": False,
                "style": "anime",
                "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            }
        )
        with pytest.raises(PromptGenerationError, match="LLM prompt 生成失败"):
            gen.generate("她在办公室整理文件。", segment_index=1)

    @patch.object(PromptGenerator, "_detect_llm_available", return_value=True)
    @patch.object(PromptGenerator, "_get_llm_client")
    def test_generate_raises_on_empty_llm_response(
        self, mock_client_factory, _mock_llm
    ):
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(content="   ")
        mock_client_factory.return_value = mock_client

        gen = PromptGenerator(
            {
                "character_tracking": False,
                "style": "anime",
                "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            }
        )
        with pytest.raises(PromptGenerationError, match="LLM 返回空 prompt"):
            gen.generate("她在办公室整理文件。", segment_index=1)

    @patch.object(PromptGenerator, "_detect_llm_available", return_value=True)
    @patch.object(PromptGenerator, "_get_llm_client")
    def test_generate_alternate_raises_on_llm_failure(
        self, mock_client_factory, _mock_llm
    ):
        mock_client = MagicMock()
        mock_client.chat.side_effect = TimeoutError("timeout")
        mock_client_factory.return_value = mock_client

        gen = PromptGenerator(
            {
                "character_tracking": False,
                "style": "anime",
                "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            }
        )
        with pytest.raises(PromptGenerationError, match="换角度 LLM prompt"):
            gen.generate_alternate("她在办公室整理文件。", segment_index=1, variant=0)

    @patch.object(PromptGenerator, "_detect_llm_available", return_value=False)
    def test_no_llm_key_still_uses_local_mode(self, _mock_llm):
        gen = PromptGenerator({"character_tracking": False, "style": "anime"})
        prompt = gen.generate("她在办公室整理文件。", segment_index=1)
        assert prompt
        assert "office" in prompt.lower() or "young woman" in prompt.lower()
