"""SubtitleGenerator 分条与二次切分测试。"""

import pytest


@pytest.mark.signature
class TestSubtitleGeneratorSplit:
    def test_short_entry_unchanged(self):
        from src.tts.subtitle_generator import SubtitleGenerator

        gen = SubtitleGenerator({"max_entry_chars": 30})
        entries = [{"start": 0.0, "end": 2.0, "text": "短字幕。"}]
        result = gen._split_oversized_entries(entries)
        assert result == entries

    def test_splits_at_comma_before_hard_cut(self):
        from src.tts.subtitle_generator import SubtitleGenerator

        gen = SubtitleGenerator({"max_entry_chars": 30})
        text = (
            "林默推开教室的门，里面空无一人，"
            "黑板上却写着一行字欢迎回来不要回头"
        )
        chunks = gen._split_text_chunks(text, 30)
        assert all(len(c) <= 30 for c in chunks)
        assert chunks[0].endswith("，")
        assert "，" in chunks[0]
        assert len(chunks) >= 2

    def test_hard_cut_when_no_punctuation(self):
        from src.tts.subtitle_generator import SubtitleGenerator

        gen = SubtitleGenerator({"max_entry_chars": 30})
        text = "abcdefghijklmnopqrstuvwxyz1234567890ABCD"
        chunks = gen._split_text_chunks(text, 30)
        assert chunks == [
            "abcdefghijklmnopqrstuvwxyz1234",
            "567890ABCD",
        ]

    def test_time_split_proportional_and_covers_range(self):
        from src.tts.subtitle_generator import SubtitleGenerator

        gen = SubtitleGenerator({"max_entry_chars": 10})
        entry = {
            "start": 1.0,
            "end": 5.0,
            "text": "一二三四五六七八九十十一十二十三十四十五",
        }
        split = gen._split_entry_by_chars(entry)
        assert len(split) >= 2
        assert all(len(e["text"]) <= 10 for e in split)
        assert split[0]["start"] == pytest.approx(1.0)
        assert split[-1]["end"] == pytest.approx(5.0)
        total_chars = sum(len(e["text"]) for e in split)
        assert total_chars == len(entry["text"])
        durations = [e["end"] - e["start"] for e in split]
        assert sum(durations) == pytest.approx(4.0)

    def test_sentence_boundary_entry_gets_secondary_split(self):
        from src.tts.subtitle_generator import SubtitleGenerator

        gen = SubtitleGenerator({"max_entry_chars": 30})
        long_text = (
            "他走进教室，发现讲台上坐着一个陌生的人，"
            "那个人缓缓抬起头，露出一张完全陌生的脸，"
            "林默却觉得在哪里见过。"
        )
        boundaries = [
            {
                "offset": 0.0,
                "duration": 12.0,
                "text": long_text,
            }
        ]
        entries = gen._build_from_boundaries(boundaries)
        split = gen._split_oversized_entries(entries)
        assert len(split) > 1
        assert all(len(e["text"]) <= 30 for e in split)
        assert split[0]["start"] == pytest.approx(0.0)
        assert split[-1]["end"] == pytest.approx(12.0)

    def test_fallback_long_clause_gets_secondary_split(self):
        from src.tts.subtitle_generator import SubtitleGenerator

        gen = SubtitleGenerator({"max_entry_chars": 30})
        text = (
            "这是一段没有任何内部标点的超长旁白文本用来测试回退模式"
            "是否会在二次切分阶段被正确截断成多条字幕"
        )
        entries = gen._build_fallback(text)
        split = gen._split_oversized_entries(entries)
        assert len(split) >= 2
        assert all(len(e["text"]) <= 30 for e in split)

    def test_generate_srt_applies_secondary_split(self, tmp_path):
        from src.tts.subtitle_generator import SubtitleGenerator

        gen = SubtitleGenerator({"max_entry_chars": 30})
        long_text = "甲" * 45
        srt_path = tmp_path / "0000.srt"
        gen.generate_srt([], long_text, srt_path)
        content = srt_path.read_text(encoding="utf-8")
        assert content.count("-->") >= 2
        lines = [ln for ln in content.splitlines() if ln and "-->" not in ln and not ln.isdigit()]
        assert all(len(ln) <= 30 for ln in lines)

    def test_respects_config_max_entry_chars(self):
        from src.tts.subtitle_generator import SubtitleGenerator

        gen = SubtitleGenerator({"max_entry_chars": 12})
        text = "一二三四五六七八九十十一十二"
        chunks = gen._split_text_chunks(text, gen.max_entry_chars)
        assert all(len(c) <= 12 for c in chunks)
        assert "".join(chunks) == text

    def test_reconcile_entries_restores_corner_brackets(self):
        from src.tts.subtitle_generator import SubtitleGenerator

        source = (
            "张得胜看着我，说：「这是真话，那刘丽华呢？」"
            "我想了一下，说：「也不认识。」「也是真话，周玲呢？」"
            "「不认识。」「还是真话，林郁？」"
        )
        # 模拟 edge-tts 句边界省略「」
        boundary_text = (
            "张得胜看着我，说：这是真话，那刘丽华呢？"
            "我想了一下，说：也不认识。也是真话，周玲呢？"
            "不认识。还是真话，林郁？"
        )
        entries = [{"start": 0.0, "end": 8.0, "text": boundary_text}]
        gen = SubtitleGenerator()
        fixed = gen._reconcile_entries_with_source(source, entries)
        assert fixed[0]["text"] == source
        assert fixed[0]["text"].count("「") == source.count("「")
        assert fixed[0]["text"].count("」") == source.count("」")

    def test_split_text_chunks_preserves_corner_brackets(self):
        from src.tts.subtitle_generator import SubtitleGenerator

        text = "我想了一下，说：「也不认识。」「也是真话，周玲呢？」「不认识。」"
        chunks = SubtitleGenerator._split_text_chunks(text, 30)
        joined = "".join(chunks)
        assert joined == text
        assert joined.count("「") == 3
        assert joined.count("」") == 3

    def test_generate_srt_reconciles_boundary_text(self, tmp_path):
        from src.tts.subtitle_generator import SubtitleGenerator

        source = "他说：「不认识。」「也是真话。」"
        boundaries = [
            {"offset": 0.0, "duration": 1.0, "text": "他说：不认识。也是真话。"},
        ]
        gen = SubtitleGenerator({"max_entry_chars": 30})
        srt_path = tmp_path / "quotes.srt"
        gen.generate_srt(boundaries, source, srt_path)
        content = srt_path.read_text(encoding="utf-8")
        assert "「不认识。」" in content
        assert "「也是真话。」" in content

    def test_generate_srt_breaks_on_newline_in_source_text(self, tmp_path):
        """原文换行应断句为多条字幕，但条目内不含物理换行。"""
        from src.tts.subtitle_generator import SubtitleGenerator

        gen = SubtitleGenerator({"max_entry_chars": 30})
        text = "关键在于你们三人的第二表情和第三表情，现在\n我来逐一说明。"
        srt_path = tmp_path / "0069.srt"
        gen.generate_srt([], text, srt_path)
        content = srt_path.read_text(encoding="utf-8")
        text_lines = [
            ln for ln in content.splitlines()
            if ln and "-->" not in ln and not ln.isdigit()
        ]
        assert len(text_lines) >= 2
        assert text_lines[0].endswith("现在")
        assert text_lines[1].startswith("我来逐一说明")
        assert all("\n" not in ln for ln in text_lines)

    def test_generate_srt_period_newline_breaks_once(self, tmp_path):
        from src.tts.subtitle_generator import SubtitleGenerator

        gen = SubtitleGenerator({"max_entry_chars": 30})
        text = "第一句。\n第二句"
        srt_path = tmp_path / "once.srt"
        gen.generate_srt([], text, srt_path)
        text_lines = [
            ln for ln in srt_path.read_text(encoding="utf-8").splitlines()
            if ln and "-->" not in ln and not ln.isdigit()
        ]
        assert text_lines == ["第一句。", "第二句"]
