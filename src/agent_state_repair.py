"""Agent 断点状态修复 — 从磁盘产物重建 segments / 资源路径。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.logger import log
from src.tools.segment_tool import SegmentTool

_LEGACY_SENTENCE_ENDINGS = re.compile(r"([。！？…]+)")


def dedupe_completed_nodes(nodes: list[str] | None) -> list[str]:
    """去重 completed_nodes，保留首次出现顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    for name in nodes or []:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _legacy_split_to_sentences(text: str) -> list[str]:
    """旧版 simple 分段器的句切分（MVP 提交中的正则逻辑）。"""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = _LEGACY_SENTENCE_ENDINGS.split(para)
        buf = ""
        for part in parts:
            if _LEGACY_SENTENCE_ENDINGS.fullmatch(part):
                buf += part
                if buf.strip():
                    sentences.append(buf.strip())
                buf = ""
            else:
                buf += part
        if buf.strip():
            sentences.append(buf.strip())
    return sentences


def _legacy_merge_sentences(
    sentences: list[str], *, max_chars: int, min_chars: int
) -> list[str]:
    if not sentences:
        return []
    segments: list[str] = []
    buffer = ""
    for sentence in sentences:
        if (
            buffer
            and len(buffer) + len(sentence) > max_chars
            and len(buffer) >= min_chars
        ):
            segments.append(buffer)
            buffer = sentence
        else:
            buffer += sentence
    if buffer:
        if len(buffer) < min_chars and segments:
            segments[-1] += buffer
        else:
            segments.append(buffer)
    return segments


def segment_legacy(text: str, config: dict[str, Any]) -> list[dict]:
    """使用旧版句切分规则分段（与引号感知 split_sentences 之前的行为一致）。"""
    seg_cfg = config.get("segmenter", {})
    max_chars = int(seg_cfg.get("max_chars", 100))
    min_chars = int(seg_cfg.get("min_chars", 20))
    sentences = _legacy_split_to_sentences(text)
    segments = _legacy_merge_sentences(
        sentences, max_chars=max_chars, min_chars=min_chars
    )
    return [{"text": seg, "index": idx} for idx, seg in enumerate(segments)]


def segment_for_image_count(
    text: str, config: dict[str, Any], image_count: int
) -> list[dict]:
    """按磁盘图片数量选择最匹配的分段结果。"""
    if not text or not text.strip():
        return []
    if image_count <= 0:
        return SegmentTool(config).run(text)

    current = SegmentTool(config).run(text)
    if len(current) == image_count:
        return current

    legacy = segment_legacy(text, config)
    if len(legacy) == image_count:
        log.info(
            "分段数 %d 与图片数一致（legacy 规则），采用 legacy 分段",
            image_count,
        )
        return legacy

    # 取更接近图片数的方案
    if abs(len(legacy) - image_count) <= abs(len(current) - image_count):
        chosen, label = legacy, "legacy"
    else:
        chosen, label = current, "current"

    log.warning(
        "分段数 (%s=%d, current=%d, legacy=%d) 与图片数 %d 不完全一致，"
        "采用 %s=%d；若音画不同步请删除 images 后重跑 art_director",
        "current",
        len(current),
        len(current),
        len(legacy),
        image_count,
        label,
        len(chosen),
    )
    return chosen


def _collect_numbered_files(directory: Path, suffix: str) -> list[str]:
    if not directory.is_dir():
        return []
    files = sorted(directory.glob(f"*{suffix}"))
    return [str(p) for p in files if p.stat().st_size > 0]


def repair_agent_state_data(
    data: dict[str, Any],
    config: dict[str, Any],
    workspace: Path | str,
) -> dict[str, Any]:
    """修复 agent_state 中缺失或与磁盘产物不一致的字段。"""
    ws = Path(workspace)
    data = dict(data)
    data["workspace"] = str(ws)
    data["completed_nodes"] = dedupe_completed_nodes(data.get("completed_nodes"))

    images_dir = ws / "images"
    image_paths = sorted(images_dir.glob("*.png"))
    image_count = len(image_paths)

    full_text = data.get("full_text") or ""
    segments = data.get("segments") or []
    # 已有 segments 时不再按图片数重分段，避免「删后半段图只重生图」时破坏段对齐
    needs_resegment = not segments

    if needs_resegment and full_text.strip():
        if image_count:
            segments = segment_for_image_count(full_text, config, image_count)
        else:
            segments = SegmentTool(config).run(full_text)
        data["segments"] = segments
        log.info("已重建 segments: %d 段", len(segments))
    elif segments and image_count and len(segments) != image_count:
        log.warning(
            "segments=%d 与磁盘图片数=%d 不一致；保留已有 segments，"
            "可删除指定段图片后重跑 art_director",
            len(segments),
            image_count,
        )

    n_seg = len(data.get("segments") or [])
    if image_count:
        expected_images = [
            str(images_dir / f"{i:04d}.png") for i in range(image_count)
        ]
        if data.get("images") != expected_images:
            data["images"] = expected_images
            log.info("已重建 images 路径: %d 张", len(expected_images))
    elif n_seg and not data.get("images"):
        data["images"] = [
            str(images_dir / f"{i:04d}.png")
            for i in range(n_seg)
            if (images_dir / f"{i:04d}.png").exists()
        ]

    audio_files = _collect_numbered_files(ws / "audio", ".mp3")
    srt_files = _collect_numbered_files(ws / "subtitles", ".srt")
    if audio_files:
        data["audio_files"] = audio_files
    if srt_files:
        data["srt_files"] = srt_files

    # 下游节点未完成时，去掉可能因异常写入的 completed 标记
    completed_list = dedupe_completed_nodes(data.get("completed_nodes"))
    completed = set(completed_list)
    if "voice_director" in completed and len(audio_files) < n_seg:
        completed.discard("voice_director")
        completed.discard("editor")
        log.info(
            "音频仅 %d/%d，移除 voice_director/editor 完成标记以续跑 TTS",
            len(audio_files),
            n_seg,
        )
    if "art_director" in completed and n_seg and image_count < n_seg:
        completed.discard("art_director")
        log.info(
            "图片仅 %d/%d，移除 art_director 完成标记以续跑生图",
            image_count,
            n_seg,
        )
    if "editor" in completed and not data.get("final_video"):
        completed.discard("editor")

    data["completed_nodes"] = [n for n in completed_list if n in completed]
    return data
