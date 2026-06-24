"""scriptplan 模块共用的 LLM JSON 解析与截断修复。"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("scriptplan")


def parse_llm_json(content: str | None) -> dict | None:
    """从 LLM 输出中解析 JSON 对象（支持 markdown 代码块）。"""
    if not content:
        return None
    text = content.strip()

    for candidate in (text,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    repaired = repair_truncated_json_object(text)
    if repaired is not None:
        return repaired

    return None


def extract_complete_array_objects(text: str, array_key: str) -> list[dict]:
    """从可能截断的 JSON 文本中提取 array_key 数组里已闭合的对象。"""
    pattern = rf'"{re.escape(array_key)}"\s*:\s*\['
    match = re.search(pattern, text)
    if not match:
        return []

    index = match.end()
    objects: list[dict] = []
    length = len(text)

    while index < length:
        while index < length and text[index] in " \t\n\r,":
            index += 1
        if index >= length or text[index] != "{":
            break

        obj_start = index
        depth = 0
        in_string = False
        escaped = False

        while index < length:
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[obj_start : index + 1]
                    try:
                        item = json.loads(chunk)
                    except json.JSONDecodeError:
                        item = None
                    if isinstance(item, dict):
                        objects.append(item)
                    index += 1
                    break
            index += 1
        else:
            break

    return objects


def repair_truncated_json_object(text: str) -> dict | None:
    """尝试从截断 JSON 中恢复顶层对象（至少 salvage segments）。"""
    stripped = text.strip()
    start = stripped.find("{")
    if start < 0:
        return None
    stripped = stripped[start:]

    segments = extract_complete_array_objects(stripped, "segments")
    if not segments:
        return None

    title_match = re.search(r'"title"\s*:\s*"((?:\\.|[^"\\])*)"', stripped)
    theme_match = re.search(r'"theme"\s*:\s*"((?:\\.|[^"\\])*)"', stripped)
    hook_match = re.search(r'"hook"\s*:\s*"((?:\\.|[^"\\])*)"', stripped)
    ending_match = re.search(r'"ending_hook"\s*:\s*"((?:\\.|[^"\\])*)"', stripped)

    def _unescape(value: str) -> str:
        try:
            return json.loads(f'"{value}"')
        except json.JSONDecodeError:
            return value

    data: dict = {"segments": segments}
    if title_match:
        data["title"] = _unescape(title_match.group(1))
    if theme_match:
        data["theme"] = _unescape(theme_match.group(1))
    if hook_match:
        data["hook"] = _unescape(hook_match.group(1))
    if ending_match:
        data["ending_hook"] = _unescape(ending_match.group(1))

    log.warning(
        "LLM JSON 被截断，已 salvage %d 个 segment（title=%s）",
        len(segments),
        data.get("title", "?"),
    )
    return data


def close_truncated_json(text: str) -> dict | None:
    """通过补全括号尝试解析截断 JSON。"""
    stripped = text.strip()
    start = stripped.find("{")
    if start < 0:
        return None
    stripped = stripped[start:]

    suffixes = (
        '"}]}',
        '"]}',
        '"}',
        "}",
        "]}",
        '""}]}',
        '"} ] }',
    )
    for suffix in suffixes:
        try:
            parsed = json.loads(stripped + suffix)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
