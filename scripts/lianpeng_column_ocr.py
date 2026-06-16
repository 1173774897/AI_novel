"""Pure helpers for two-column OCR text assembly."""

from __future__ import annotations

from typing import NamedTuple


class OcrBox(NamedTuple):
    cx: float
    y0: float
    x0: float
    text: str


class PageOcr(NamedTuple):
    left: str
    right: str
    full: str
    two_column: bool


def box_metrics(box: list[list[float]]) -> tuple[float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs) / 4, min(ys), min(xs)


def assemble_column_text(boxes: list[OcrBox], line_merge_px: float = 14.0) -> str:
    """Sort OCR boxes top-to-bottom and merge into lines."""
    if not boxes:
        return ""
    boxes = sorted(boxes, key=lambda b: (round(b.y0 / line_merge_px), b.x0))
    lines: list[str] = []
    current_y: float | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            lines.append("".join(buffer))

    for box in boxes:
        if current_y is None:
            current_y = box.y0
            buffer = [box.text]
            continue
        if abs(box.y0 - current_y) <= line_merge_px:
            buffer.append(box.text)
        else:
            flush()
            buffer = [box.text]
            current_y = box.y0
    flush()
    return "\n".join(lines)


def _find_column_split(boxes: list[OcrBox], page_width: int) -> float:
    """Detect gutter between columns via largest horizontal gap."""
    centers = sorted(b.cx for b in boxes)
    lo = page_width * 0.32
    hi = page_width * 0.68
    mids = [c for c in centers if lo <= c <= hi]
    if len(mids) < 2:
        return page_width / 2

    best_gap = 0.0
    split = page_width / 2
    for i in range(len(mids) - 1):
        gap = mids[i + 1] - mids[i]
        if gap > best_gap:
            best_gap = gap
            split = (mids[i] + mids[i + 1]) / 2
    if best_gap < page_width * 0.025:
        return page_width / 2
    return split


def split_boxes_two_column(
    boxes: list[OcrBox], page_width: int
) -> tuple[list[OcrBox], list[OcrBox], bool]:
    if not boxes:
        return [], [], False

    split = _find_column_split(boxes, page_width)
    left = [b for b in boxes if b.cx < split]
    right = [b for b in boxes if b.cx >= split]

    left_ratio = len(left) / len(boxes)
    right_ratio = len(right) / len(boxes)
    two_column = (
        len(left) >= 3
        and len(right) >= 3
        and left_ratio >= 0.25
        and right_ratio >= 0.25
        and max(left, key=lambda b: b.cx).cx < split * 1.02
        and min(right, key=lambda b: b.cx).cx > split * 0.98
    )
    if not two_column:
        return boxes, [], False
    return left, right, True


def build_page_ocr(
    ocr_result: list[tuple[list[list[float]], str, float]] | None,
    page_width: int,
) -> PageOcr:
    """Build ordered page text from raw RapidOCR output."""
    if not ocr_result:
        return PageOcr("", "", "", False)

    boxes = [
        OcrBox(*box_metrics(box), text.strip())
        for box, text, _score in ocr_result
        if text and text.strip()
    ]
    left_boxes, right_boxes, two_column = split_boxes_two_column(boxes, page_width)

    if not two_column:
        full = assemble_column_text(left_boxes)
        return PageOcr(full, "", full, False)

    left_text = assemble_column_text(left_boxes)
    right_text = assemble_column_text(right_boxes)
    if left_text and right_text:
        full = f"{left_text}\n\n{right_text}"
    else:
        full = left_text or right_text
    return PageOcr(left_text, right_text, full, True)
