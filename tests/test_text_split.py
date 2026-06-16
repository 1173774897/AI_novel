"""配音/字幕断句规则测试。"""

import pytest

from src.tts.text_split import split_utterances

pytestmark = pytest.mark.signature


class TestSplitUtterances:
    def test_newline_breaks(self):
        assert split_utterances("现在\n我来") == ["现在", "我来"]

    def test_period_and_newline_break_once(self):
        assert split_utterances("第一句。\n第二句") == ["第一句。", "第二句"]

    def test_period_then_multiple_newlines(self):
        assert split_utterances("完。\n\n\n下一句") == ["完。", "下一句"]

    def test_period_without_newline(self):
        assert split_utterances("甲。乙") == ["甲。", "乙"]

    def test_newline_before_period_on_next_line(self):
        assert split_utterances("上半\n句。下半") == ["上半", "句。", "下半"]

    def test_empty_returns_empty(self):
        assert split_utterances("   ") == []

    def test_crlf_normalized(self):
        assert split_utterances("a\r\nb") == ["a", "b"]

    def test_ellipsis_does_not_split_inside_quote(self):
        text = "「我……我什么都没干。」他开始嗫嚅，嘴唇哆嗦着。"
        assert split_utterances(text) == [
            "「我……我什么都没干。」",
            "他开始嗫嚅，嘴唇哆嗦着。",
        ]

    def test_dialogue_with_ellipsis_and_quotes(self):
        text = (
            "「想清楚了啊，要是有什么隐瞒被查出来，这辈子可就搭进去了。」"
            "我直愣愣地盯着他，步步紧逼。"
            "「我……我什么都没干。」他开始嗫嚅，嘴唇哆嗦着。"
            "「不怕告诉你，从你们餐厅消失的这个女人，很可能已经遇害了！"
        )
        parts = split_utterances(text)
        assert all(len(p) > 1 for p in parts)
        assert "……" in parts[2]
        assert parts[-1].endswith("！")

    def test_nested_questions_merge_closing_quote(self):
        text = (
            "而且，那个箱子我也没敢留在手里……」\n"
            "这倒引起了我的兴趣，问他：「为什么？你不就是为了偷她的箱子吗？」\n"
            "「那个箱子有古怪。」小徐看来真是怕了：「我看着害怕，就丢掉了。"
            "当时我以为里面有钱，结果没有。"
        )
        parts = split_utterances(text)
        assert "'」'" not in [repr(p) for p in parts]
        assert all(not _is_orphan_only(p) for p in parts)
        assert any("你不就是为了偷她的箱子吗？」" in p for p in parts)

    def test_short_speech_merges_trailing_quote(self):
        assert split_utterances("他摇头：「不。」") == ["他摇头：「不。」"]

    def test_wechat_bracket_closes_like_quote(self):
        text = "看到消息。【学姐你好，高价求预测卷。】他笑了。"
        assert split_utterances(text) == [
            "看到消息。",
            "【学姐你好，高价求预测卷。】",
            "他笑了。",
        ]

    def test_orphan_closing_wechat_bracket_merges_to_next(self):
        """分段边界残留的单独「】」须并入下一句，否则 edge-tts 会失败。"""
        text = "】\n我勾了勾嘴角，通过认证。"
        parts = split_utterances(text)
        assert parts == ["】我勾了勾嘴角，通过认证。"]
        assert all(not _is_orphan_only(p) for p in parts)

    def test_lone_wechat_brackets_are_unspeakable(self):
        from src.tts.text_split import is_unspeakable_fragment

        assert is_unspeakable_fragment("】")
        assert is_unspeakable_fragment("【")
        assert not is_unspeakable_fragment("【你好】")

    def test_em_dash_line_merges_into_previous_sentence(self):
        """单独成行的破折号 edge-tts 无法合成，须并入前句。"""
        text = "偷来的东西终究不会长久，唯有拼搏才能创造一切可能。——\n（全文完）"
        assert split_utterances(text) == [
            "偷来的东西终究不会长久，唯有拼搏才能创造一切可能。——",
            "（全文完）",
        ]

    def test_lone_em_dash_is_unspeakable(self):
        from src.tts.text_split import is_unspeakable_fragment

        assert is_unspeakable_fragment("——")
        assert is_unspeakable_fragment("—")


def _is_orphan_only(text: str) -> bool:
    from src.tts.text_split import _is_orphan_fragment

    return _is_orphan_fragment(text)
