#!/usr/bin/env python3
"""Extract scanned 莲蓬鬼话 PDF into per-story txt files via two-column OCR."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import fitz
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from lianpeng_column_ocr import PageOcr, build_page_ocr

PDF_PATH = Path(
    "/Users/lizhe/Downloads/天涯文本/books/"
    "38.《莲蓬鬼话》[莲蓬编著][重庆出版社][978-7-229-03310-1][2010.12][P214].pdf"
)
OUT_DIR = Path("/Users/lizhe/Downloads/天涯文本/books/莲蓬鬼话")
CACHE_PATH = OUT_DIR / "_ocr_cache.json"

SINGLE_RANGES: list[tuple[int, int, str]] = [
    (8, 22, "山妖"),
    (24, 34, "种人得人"),
    (35, 43, "团购死亡"),
    (44, 45, "莲藕开会"),
    (58, 58, "超短群"),
    (84, 84, "超短群"),
    (85, 85, "别对我撒谎"),
    (123, 123, "超短群"),
    (135, 135, "八卦炉"),
    (159, 159, "公告板"),
    (160, 160, "超短群"),
    (169, 171, "棒得情报站"),
    (181, 181, "超短群"),
    (182, 184, "名家逼供"),
    (211, 211, "超短群"),
]

INTERLEAVED_RANGES: list[tuple[int, int, list[str]]] = [
    (46, 57, ["日食", "羊皮日记"]),
    (59, 71, ["吉庆", "羊皮日记"]),
    (72, 83, ["被自己出卖", "羊皮日记"]),
    (86, 96, ["骇人之心不可无", "别对我撒谎"]),
    (97, 107, ["万能钥匙", "骇人之心不可无"]),
    (108, 122, ["鬼市", "骇人之心不可无"]),
    (124, 135, ["阴阳门快刀", "夜半诡话"]),
    (136, 144, ["一支录音笔", "夜半诡话"]),
    (145, 158, ["冥婚", "夜半诡话"]),
    (161, 180, ["速效救心丸", "微博·杀手"]),
    (186, 210, ["暗夜尽头", "深水之下"]),
]

HEADER_PATTERNS: dict[str, re.Pattern[str]] = {
    "山妖": re.compile(r"山妖"),
    "种人得人": re.compile(r"种人得人"),
    "团购死亡": re.compile(r"团购死亡"),
    "日食": re.compile(r"^日食"),
    "羊皮日记": re.compile(r"羊皮日记"),
    "吉庆": re.compile(r"^吉庆"),
    "被自己出卖": re.compile(r"被自己出卖"),
    "骇人之心不可无": re.compile(r"骇人之心|骏人之心"),
    "别对我撒谎": re.compile(r"别对我撒谎"),
    "万能钥匙": re.compile(r"万能钥匙"),
    "鬼市": re.compile(r"^鬼市"),
    "夜半诡话": re.compile(r"夜半诡话|夜半论话"),
    "阴阳门快刀": re.compile(r"阴阳门"),
    "一支录音笔": re.compile(r"一支录音笔"),
    "冥婚": re.compile(r"冥婚|其婚|掌婚"),
    "速效救心丸": re.compile(r"速效救心丸|速数救心丸|陈伯"),
    "微博·杀手": re.compile(r"微博"),
    "暗夜尽头": re.compile(r"^暗夜尽头$|惊险连篇"),
    "深水之下": re.compile(r"深水之下|AYJTSSZX"),
    "之后如何": re.compile(r"之后如何"),
    "超短群": re.compile(r"^超短群"),
    "莲藕开会": re.compile(r"莲藕开会"),
    "八卦炉": re.compile(r"八卦炉"),
    "公告板": re.compile(r"公告板"),
    "棒得情报站": re.compile(r"棒得情报"),
    "名家逼供": re.compile(r"名家逼供"),
}


def sanitize_filename(name: str) -> str:
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()


def ocr_page_columns(img: Image.Image, ocr: RapidOCR) -> PageOcr:
    """OCR a page: read left column line-by-line, then right column."""
    width, _ = img.size
    result, _ = ocr(img)
    return build_page_ocr(result, width)


def render_page(doc: fitz.Document, page_index: int, scale: float = 2.0) -> Image.Image:
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    return Image.open(io.BytesIO(pix.tobytes("png")))


def ocr_document(
    doc: fitz.Document, ocr: RapidOCR, *, force: bool = False
) -> list[PageOcr]:
    if not force and CACHE_PATH.exists():
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            if len(raw) == doc.page_count:
                print(f"Loaded OCR cache ({len(raw)} pages)")
                return [PageOcr(**item) for item in raw]

    pages: list[PageOcr] = []
    total = doc.page_count
    for i in range(total):
        img = render_page(doc, i)
        pages.append(ocr_page_columns(img, ocr))
        if (i + 1) % 10 == 0 or i + 1 == total:
            print(f"OCR {i + 1}/{total}")

    payload = [p._asdict() for p in pages]
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return pages


def single_story_for_page(page_no: int) -> str | None:
    for start, end, title in SINGLE_RANGES:
        if start <= page_no <= end:
            return title
    if page_no == 185:
        return "之后如何"
    return None


def interleaved_candidates(page_no: int) -> list[str] | None:
    for start, end, candidates in INTERLEAVED_RANGES:
        if start <= page_no <= end:
            return candidates
    return None


def detect_from_headers(text: str, candidates: list[str] | None = None) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    search = candidates if candidates else list(HEADER_PATTERNS)
    for line in lines[:6]:
        for name in search:
            pat = HEADER_PATTERNS[name]
            if pat.search(line):
                return name
    return None


def assign_story_chunks(pages: list[PageOcr]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    interleaved_state: dict[tuple[int, int], str] = {}

    for page_no, page in enumerate(pages, start=1):
        fixed = single_story_for_page(page_no)
        candidates = interleaved_candidates(page_no)

        if fixed:
            text = page.full.strip()
            if text:
                grouped[fixed].append(text)
            continue

        if candidates and page.two_column:
            left_story = detect_from_headers(page.left, candidates)
            right_story = detect_from_headers(page.right, candidates)

            key = (candidates[0], candidates[1])
            if left_story is None:
                left_story = interleaved_state.get(key, candidates[0])
            if right_story is None:
                alt = candidates[1] if left_story == candidates[0] else candidates[0]
                right_story = interleaved_state.get((candidates[1], candidates[0]), alt)

            if left_story == right_story:
                text = page.full.strip()
                if text:
                    grouped[left_story].append(text)
                    interleaved_state[key] = left_story
                continue

            if page.left.strip():
                grouped[left_story].append(page.left.strip())
            if page.right.strip():
                grouped[right_story].append(page.right.strip())
            interleaved_state[key] = right_story
            continue

        if candidates:
            key = (candidates[0], candidates[1])
            story = detect_from_headers(page.full, candidates)
            if story is None:
                story = interleaved_state.get(key, candidates[0])
            text = page.full.strip()
            if text:
                grouped[story].append(text)
                interleaved_state[key] = story
            continue

    return grouped


STORY_PAGE_RANGES: dict[str, tuple[int, int]] = {
    "被自己出卖": (72, 83),
}


def reocr_story_pages(
    doc: fitz.Document,
    ocr: RapidOCR,
    pages: list[PageOcr],
    page_start: int,
    page_end: int,
    *,
    scale: float = 3.0,
) -> list[PageOcr]:
    """Re-OCR a page range at higher resolution and patch the cache."""
    updated = list(pages)
    for page_no in range(page_start, page_end + 1):
        idx = page_no - 1
        img = render_page(doc, idx, scale=scale)
        updated[idx] = ocr_page_columns(img, ocr)
        print(f"Re-OCR page {page_no}/{page_end} (scale={scale})")
    return updated


def collect_story_chunks(
    pages: list[PageOcr],
    story: str,
    page_start: int,
    page_end: int,
) -> list[str]:
    """Collect text chunks for one story inside a page range."""
    chunks: list[str] = []
    candidates: list[str] | None = None
    for start, end, cands in INTERLEAVED_RANGES:
        if start <= page_start <= end:
            candidates = cands if story in cands else None
            break

    state: str | None = None
    for page_no in range(page_start, page_end + 1):
        page = pages[page_no - 1]
        if page_no == page_start and not page.two_column:
            text = page.full.strip()
            if text:
                chunks.append(text)
            state = story
            continue

        if candidates and page.two_column:
            left_story = detect_from_headers(page.left, candidates)
            right_story = detect_from_headers(page.right, candidates)

            if left_story is None:
                left_story = state if state == story else (
                    story if candidates[0] == story else candidates[1]
                )
            if right_story is None:
                right_story = (
                    story
                    if left_story != story
                    else (candidates[1] if candidates[0] == story else candidates[0])
                )

            if left_story == right_story == story:
                text = page.full.strip()
                if text:
                    chunks.append(text)
            else:
                if left_story == story and page.left.strip():
                    chunks.append(page.left.strip())
                if right_story == story and page.right.strip():
                    chunks.append(page.right.strip())
            state = story
            continue

        text = page.full.strip()
        if text and (detect_from_headers(text, [story]) == story or state == story):
            chunks.append(text)
            state = story
    return chunks


def save_cache(pages: list[PageOcr]) -> None:
    CACHE_PATH.write_text(
        json.dumps([p._asdict() for p in pages], ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract 莲蓬鬼话 PDF stories")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-run OCR for entire PDF",
    )
    parser.add_argument(
        "--story",
        metavar="NAME",
        help="Re-OCR and rewrite a single story (e.g. 被自己出卖)",
    )
    args = parser.parse_args(argv)

    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(PDF_PATH))
    ocr = RapidOCR()

    if args.story:
        story = args.story
        if story not in STORY_PAGE_RANGES:
            print(f"Unknown story: {story}", file=sys.stderr)
            print(f"Available: {', '.join(STORY_PAGE_RANGES)}", file=sys.stderr)
            return 1
        page_start, page_end = STORY_PAGE_RANGES[story]
        pages = ocr_document(doc, ocr, force=False)
        pages = reocr_story_pages(
            doc, ocr, pages, page_start, page_end, scale=3.0
        )
        save_cache(pages)
        chunks = collect_story_chunks(pages, story, page_start, page_end)
        content = "\n\n".join(chunks).strip() + "\n"
        out_path = OUT_DIR / f"{sanitize_filename(story)}.txt"
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote {out_path.name} ({len(chunks)} chunks, pdf {page_start}-{page_end})")
        return 0

    pages = ocr_document(doc, ocr, force=args.force)
    grouped = assign_story_chunks(pages)

    written: list[str] = []
    for title in sorted(grouped, key=lambda t: grouped[t][0][:1] if grouped[t] else ""):
        chunks = grouped[title]
        content = "\n\n".join(chunks).strip()
        if not content:
            continue
        out_path = OUT_DIR / f"{sanitize_filename(title)}.txt"
        out_path.write_text(content + "\n", encoding="utf-8")
        written.append(f"{title}.txt ({len(chunks)} chunks)")
        print(f"Wrote {out_path.name} ({len(chunks)} chunks)")

    (OUT_DIR / "_index.txt").write_text("\n".join(written) + "\n", encoding="utf-8")
    print(f"\nDone: {len(written)} files -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
