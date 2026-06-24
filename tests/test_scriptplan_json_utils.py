"""scriptplan JSON 解析工具测试。"""

from __future__ import annotations

import json

import pytest

from src.scriptplan.json_utils import (
    extract_complete_array_objects,
    parse_llm_json,
    repair_truncated_json_object,
)


@pytest.mark.signature
def test_parse_llm_json_markdown_block():
    payload = {"title": "测试", "segments": []}
    text = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    assert parse_llm_json(text) == payload


@pytest.mark.signature
def test_repair_truncated_json_object_salvages_segments():
    truncated = """
    {
      "title": "橘猫拆箱",
      "theme": "快递开箱",
      "hook": "神秘箱子",
      "visual_bible": {
        "style_tags": "realistic",
        "negative_prompt": "blurry, dist
    """
    # 模拟 segments 已输出但被截断在 visual_bible 之前没有 segments 的情况
    assert repair_truncated_json_object(truncated) is None

    with_segments = truncated + """,
      "segments": [
        {"id": 1, "purpose": "hook", "voiceover": "门铃响了", "visual": "橘猫竖耳", "duration_sec": 3},
        {"id": 2, "purpose": "setup", "voiceover": "一个大箱子", "visual": "纸箱特写", "duration_sec": 4}
    """
    data = repair_truncated_json_object(with_segments)
    assert data is not None
    assert len(data["segments"]) == 2
    assert data["title"] == "橘猫拆箱"


@pytest.mark.signature
def test_extract_complete_array_objects():
    text = """
    {
      "segments": [
        {"id": 1, "purpose": "hook", "voiceover": "a", "visual": "b"},
        {"id": 2, "purpose": "setup", "voiceover": "c", "visual": "d"
    """
    items = extract_complete_array_objects(text, "segments")
    assert len(items) == 1
    assert items[0]["id"] == 1
