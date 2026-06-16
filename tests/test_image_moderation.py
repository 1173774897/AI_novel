"""生图内容审核软化测试。"""

import pytest

from src.imagegen.moderation import (
    alternate_angle_hint,
    is_content_moderation_error,
    is_jimeng_retryable_error,
    minimal_safe_fallback_prompt,
    retry_image_prompt_after_failure,
    sanitize_image_prompt_text,
    soften_image_prompt,
    soften_image_prompt_for_attempt,
)


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
def test_soften_attempt1_keeps_scene_context():
    raw = "apartment corridor at night, bloody scene, cinematic"
    softened = soften_image_prompt(raw, attempt=1)
    assert "corridor" in softened.lower()
    assert "bloody" not in softened.lower()
    assert "PG-13" in softened


@pytest.mark.signature
def test_minimal_safe_fallback_is_generic_empty():
    out = minimal_safe_fallback_prompt(0)
    assert "empty corridor" in out or "empty room" in out


@pytest.mark.signature
def test_jimeng_generation_failed_is_retryable():
    assert is_jimeng_retryable_error("generation failed: final generation failed")


@pytest.mark.signature
def test_jimeng_network_error_not_retryable():
    assert not is_jimeng_retryable_error("connection reset by peer")


@pytest.mark.signature
def test_sanitize_strips_cn_violent_and_cast_blocks():
    raw = (
        "【本段相关角色】302：手持菜刀\n"
        "bloodstain on hallway, murder scene, 血渍拖行"
    )
    cleaned = sanitize_image_prompt_text(raw)
    assert "302" not in cleaned
    assert "血" not in cleaned
    assert "bloodstain" not in cleaned.lower()
    assert "murder" not in cleaned.lower()


@pytest.mark.signature
def test_retry_generation_failed_escalates_to_safe_prompt():
    horror = "bloody corridor, 血渍, killer with axe, " + "x" * 200
    step0 = retry_image_prompt_after_failure(
        horror, 0, "generation failed: final generation failed"
    )
    step1 = retry_image_prompt_after_failure(
        horror, 1, "generation failed: final generation failed"
    )
    step2 = retry_image_prompt_after_failure(
        horror, 2, "generation failed: final generation failed"
    )
    assert len(step0) <= 1000
    assert "血" not in step0
    assert "corridor" in step1.lower()
    assert "corridor" in step2.lower()
    step3 = retry_image_prompt_after_failure(
        horror, 3, "generation failed: final generation failed"
    )
    assert "empty corridor" in step3 or "empty room" in step3


@pytest.mark.signature
def test_soften_for_attempt_escalates():
    raw = "apartment corridor, killer, 血渍"
    s0 = soften_image_prompt_for_attempt(raw, 0)
    s2 = soften_image_prompt_for_attempt(raw, 2)
    assert "bloody" not in s0.lower()
    assert "corridor" in s2.lower()
    assert "empty corridor" not in s2.lower()


@pytest.mark.signature
def test_alternate_angle_hint_three_variants():
    hints = [alternate_angle_hint(i) for i in range(3)]
    assert all("远景" in h[0] or "过肩" in h[0] or "隐喻" in h[0] for h in hints)
    assert len({h[1] for h in hints}) == 3
