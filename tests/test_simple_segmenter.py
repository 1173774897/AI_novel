"""SimpleSegmenter 引号感知断句测试。"""

import pytest

from src.segmenter.simple_segmenter import SimpleSegmenter

pytestmark = pytest.mark.signature


@pytest.fixture
def seg():
    return SimpleSegmenter({"max_chars": 200, "min_chars": 1})


def test_split_at_corner_quote_close(seg):
    text = "老人说：「你能来，我还是很高兴的。」然后停住了。"
    sentences = seg._split_to_sentences(text)
    assert sentences == [
        "老人说：「你能来，我还是很高兴的。」",
        "然后停住了。",
    ]


def test_split_at_ascii_double_quote(seg):
    text = '记者问道："你去哪了？"他沉默了。'
    sentences = seg._split_to_sentences(text)
    assert sentences == ['记者问道："你去哪了？"', "他沉默了。"]


def test_no_split_on_period_inside_quotes(seg):
    text = "「第一句。第二句。」旁白继续。"
    sentences = seg._split_to_sentences(text)
    assert sentences == ["「第一句。第二句。」", "旁白继续。"]


def test_segment_merges_short_sentences(seg):
    text = "甲说：「你好。」乙说：「再见。」"
    parts = [s["text"] for s in seg.segment(text)]
    assert len(parts) == 1
    assert "「你好。」" in parts[0]
    assert "「再见。」" in parts[0]
