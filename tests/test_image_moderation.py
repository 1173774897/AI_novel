"""生图内容审核软化测试。"""

import pytest

from src.imagegen.moderation import is_content_moderation_error, soften_image_prompt


@pytest.mark.signature
def test_detect_dashscope_moderation_error():
    detail = '{"code":"DataInspectionFailed","message":"inappropriate content"}'
    assert is_content_moderation_error(detail)


@pytest.mark.signature
def test_soften_strips_risky_words():
    raw = "a nude woman, bloody scene, cinematic"
    softened = soften_image_prompt(raw, attempt=0)
    assert "nude" not in softened.lower()
    assert "bloody" not in softened.lower()
    assert "PG-13" in softened


@pytest.mark.signature
def test_soften_fallback_generic():
    out = soften_image_prompt("anything", attempt=1)
    assert "establishing shot" in out
