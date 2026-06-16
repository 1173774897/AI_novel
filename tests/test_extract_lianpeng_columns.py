"""Tests for two-column OCR assembly."""

from __future__ import annotations

import pytest

from scripts.lianpeng_column_ocr import OcrBox, assemble_column_text, split_boxes_two_column


@pytest.mark.signature
def test_assemble_column_text_reads_top_to_bottom() -> None:
    boxes = [
        OcrBox(cx=100, y0=200, x0=90, text="第二行"),
        OcrBox(cx=100, y0=50, x0=90, text="第一行"),
        OcrBox(cx=100, y0=350, x0=90, text="第三行"),
    ]
    assert assemble_column_text(boxes) == "第一行\n第二行\n第三行"


@pytest.mark.signature
def test_assemble_column_text_merges_same_line_boxes() -> None:
    boxes = [
        OcrBox(cx=80, y0=40, x0=70, text="你好"),
        OcrBox(cx=140, y0=42, x0=130, text="世界"),
    ]
    assert assemble_column_text(boxes) == "你好世界"


@pytest.mark.signature
def test_split_boxes_two_column_detects_layout() -> None:
    boxes = [
        OcrBox(cx=120, y0=50, x0=110, text="左1"),
        OcrBox(cx=130, y0=100, x0=120, text="左2"),
        OcrBox(cx=150, y0=150, x0=140, text="左3"),
        OcrBox(cx=520, y0=60, x0=510, text="右1"),
        OcrBox(cx=530, y0=110, x0=520, text="右2"),
        OcrBox(cx=540, y0=160, x0=530, text="右3"),
    ]
    left, right, two_column = split_boxes_two_column(boxes, page_width=700)
    assert two_column is True
    assert len(left) == 3
    assert len(right) == 3
    assert assemble_column_text(left) == "左1\n左2\n左3"
    assert assemble_column_text(right) == "右1\n右2\n右3"


@pytest.mark.signature
def test_split_boxes_single_column_title_page() -> None:
    boxes = [
        OcrBox(cx=350, y0=100, x0=300, text="山妖"),
        OcrBox(cx=360, y0=160, x0=310, text="【文】姻合"),
    ]
    left, right, two_column = split_boxes_two_column(boxes, page_width=700)
    assert two_column is False
    assert len(left) == 2
    assert right == []
