"""字幕生成器 - 根据 TTS 词边界生成 SRT 字幕文件"""

import re
from pathlib import Path

from src.logger import log
from src.tts.text_split import split_utterances

# 中文字幕每行理想字符数范围（首轮分组）
_MIN_LINE_CHARS = 10
_MAX_LINE_CHARS = 15

# 单条字幕硬上限默认值（二次切分）
_DEFAULT_MAX_ENTRY_CHARS = 30

# 回退模式下的估算语速（字符/秒）
_FALLBACK_CHARS_PER_SECOND = 5.0

# 最短单条显示时长（秒）
_MIN_ENTRY_DURATION = 0.3

# 中文标点集合，用于判断断句位置
_PUNCTUATION = set("，。！？、；：""''（）《》—…\n")

# 二次切分时的断点优先级（逗号优先）
_PREFERRED_BREAK_CHARS = "，。！？、；："

# edge-tts 句边界常省略、但字幕需保留的标点
_PRESERVE_PUNCT = set("「」『』""''（）《》【】—…")
_QUOTE_PAIRS = (("「", "」"), ("『", "』"), ('"', '"'), ("'", "'"))


class SubtitleGenerator:
    """从 TTS 词边界数据生成 SRT 字幕文件。

    支持两种模式:
    1. 精确模式: 利用 edge-tts 返回的 word_boundaries 精确对齐
    2. 回退模式: word_boundaries 为空时，按标点拆分并估算时间

    两种模式完成后都会做二次切分，确保单条字幕不超过 max_entry_chars。
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.max_entry_chars: int = int(
            cfg.get("max_entry_chars", _DEFAULT_MAX_ENTRY_CHARS)
        )

    def generate_srt(
        self,
        word_boundaries: list[dict],
        text: str,
        output_path: Path,
    ) -> Path:
        """生成 SRT 字幕文件。

        Args:
            word_boundaries: TTS 引擎返回的词边界列表，每个元素为
                {"offset": float, "duration": float, "text": str}。
            text: 原始文本（用于回退模式）。
            output_path: SRT 文件输出路径。

        Returns:
            output_path，方便链式调用。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text = (text or "").strip()
        if not text:
            log.warning("字幕生成收到空文本，写入空 SRT")
            output_path.write_text("", encoding="utf-8")
            return output_path

        if word_boundaries:
            entries = self._build_from_boundaries(word_boundaries, source_text=text)
            entries = self._reconcile_entries_with_source(text, entries)
        else:
            log.info("无词边界数据，使用回退模式按标点拆分生成字幕")
            entries = self._build_fallback(text)

        entries = self._split_oversized_entries(entries)

        srt_content = self._render_srt(entries)
        output_path.write_text(srt_content, encoding="utf-8")
        log.debug("字幕写入: %s (%d 条)", output_path.name, len(entries))
        return output_path

    # ------------------------------------------------------------------
    # 精确模式：从词边界构建字幕
    # ------------------------------------------------------------------

    def _build_from_boundaries(
        self, word_boundaries: list[dict], *, source_text: str = ""
    ) -> list[dict]:
        """将词边界按字符数分组为字幕条目。"""
        utterances = split_utterances(source_text) if source_text else []
        if len(utterances) > 1:
            return self._build_from_boundaries_by_utterances(
                word_boundaries, utterances
            )
        return self._build_from_boundaries_single(word_boundaries)

    def _build_from_boundaries_single(
        self, word_boundaries: list[dict]
    ) -> list[dict]:
        """单段文本：按字符数/标点分组。"""
        entries: list[dict] = []
        group_words: list[dict] = []
        group_text = ""

        for wb in word_boundaries:
            chunk = self._sanitize_entry_text(wb.get("text", ""))
            if not chunk:
                continue
            wb = {**wb, "text": chunk}
            group_words.append(wb)
            group_text += chunk

            should_break = False
            if len(group_text) >= _MAX_LINE_CHARS:
                should_break = True
            elif len(group_text) >= _MIN_LINE_CHARS and self._ends_with_punctuation(
                wb["text"]
            ):
                should_break = True

            if should_break:
                entries.append(self._flush_group(group_words, group_text))
                group_words = []
                group_text = ""

        if group_words:
            entries.append(self._flush_group(group_words, group_text))

        return entries

    def _build_from_boundaries_by_utterances(
        self,
        word_boundaries: list[dict],
        utterances: list[str],
    ) -> list[dict]:
        """多段断句：每段 utterance 单独分组，段内再按字数切条。"""
        entries: list[dict] = []
        wb_i = 0
        wb_list = list(word_boundaries)

        for utt in utterances:
            target_len = len(self._sanitize_entry_text(utt))
            if target_len <= 0:
                continue

            group_words: list[dict] = []
            group_text = ""
            consumed = 0

            while wb_i < len(wb_list) and consumed < target_len:
                wb = wb_list[wb_i]
                chunk = self._sanitize_entry_text(wb.get("text", ""))
                wb_i += 1
                if not chunk:
                    continue
                wb = {**wb, "text": chunk}
                group_words.append(wb)
                group_text += chunk
                consumed += len(chunk)

                should_break = consumed >= target_len
                if not should_break and len(group_text) >= _MAX_LINE_CHARS:
                    should_break = True
                elif (
                    not should_break
                    and len(group_text) >= _MIN_LINE_CHARS
                    and self._ends_with_punctuation(wb["text"])
                ):
                    should_break = True

                if should_break:
                    entries.append(self._flush_group(group_words, group_text))
                    group_words = []
                    group_text = ""

            if group_words:
                entries.append(self._flush_group(group_words, group_text))

        while wb_i < len(wb_list):
            wb = wb_list[wb_i]
            chunk = self._sanitize_entry_text(wb.get("text", ""))
            wb_i += 1
            if chunk:
                entries.append(
                    self._flush_group([{**wb, "text": chunk}], chunk)
                )

        return entries

    @classmethod
    def _reconcile_entries_with_source(
        cls, source: str, entries: list[dict]
    ) -> list[dict]:
        """将 edge-tts 边界文本与原文对齐，补回「」等标点。"""
        if not entries or not source:
            return entries
        joined = "".join(e.get("text", "") for e in entries)
        if joined == source:
            return entries

        result: list[dict] = []
        src_i = 0
        for entry in entries:
            core = entry.get("text", "")
            if not core:
                result.append(entry)
                continue
            buf: list[str] = []
            ci = 0
            while ci < len(core) and src_i < len(source):
                sc = source[src_i]
                cc = core[ci]
                if sc == cc:
                    buf.append(sc)
                    src_i += 1
                    ci += 1
                elif sc in _PRESERVE_PUNCT:
                    buf.append(sc)
                    src_i += 1
                elif cc in _PRESERVE_PUNCT:
                    ci += 1
                elif sc.isspace():
                    src_i += 1
                elif cc.isspace():
                    ci += 1
                else:
                    src_i += 1
            result.append({**entry, "text": "".join(buf) or core})

        if src_i < len(source) and result:
            last = result[-1]
            result[-1] = {**last, "text": last["text"] + source[src_i:]}
        return result

    @staticmethod
    def _is_inside_quotes(text: str, idx: int) -> bool:
        """idx 是否在未闭合的引号内。"""
        prefix = text[: idx + 1]
        for open_c, close_c in _QUOTE_PAIRS:
            if prefix.count(open_c) > prefix.count(close_c):
                return True
        return False

    @staticmethod
    def _flush_group(group_words: list[dict], group_text: str) -> dict:
        """将一组词边界合并为单条字幕条目。"""
        start = group_words[0]["offset"]
        last = group_words[-1]
        end = last["offset"] + last["duration"]
        # 保证最短显示 0.3 秒，避免闪烁
        if end - start < _MIN_ENTRY_DURATION:
            end = start + _MIN_ENTRY_DURATION
        return {
            "start": start,
            "end": end,
            "text": SubtitleGenerator._sanitize_entry_text(group_text),
        }

    # ------------------------------------------------------------------
    # 回退模式：按标点拆分 + 估算时间
    # ------------------------------------------------------------------

    def _build_fallback(self, text: str) -> list[dict]:
        """无词边界时的回退策略：按标点拆分，按语速估算时间。

        Returns:
            [{"start": float, "end": float, "text": str}, ...]
        """
        segments = self._split_by_punctuation(text)
        entries: list[dict] = []
        current_time = 0.0

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            duration = len(seg) / _FALLBACK_CHARS_PER_SECOND
            # 最短 0.5 秒，避免闪烁
            duration = max(duration, 0.5)
            entries.append(
                {
                    "start": current_time,
                    "end": current_time + duration,
                    "text": seg,
                }
            )
            current_time += duration

        return entries

    @staticmethod
    def _split_by_punctuation(text: str) -> list[str]:
        """按断句规则（换行 + 句末标点）拆分文本。"""
        return split_utterances(text)

    # ------------------------------------------------------------------
    # 二次切分：超长条目按标点/硬切 + 按比例分配时间
    # ------------------------------------------------------------------

    def _split_oversized_entries(self, entries: list[dict]) -> list[dict]:
        """将超长字幕条目切分为多条，并按字符数比例分配显示时间。"""
        result: list[dict] = []
        for entry in entries:
            text = entry.get("text", "").strip()
            if not text:
                continue
            if len(text) <= self.max_entry_chars:
                result.append(entry)
                continue
            for split_entry in self._split_entry_by_chars(entry):
                result.append(split_entry)
        return result

    def _split_entry_by_chars(self, entry: dict) -> list[dict]:
        """将单条超长字幕按 max_entry_chars 切分并分配时间。"""
        text = entry["text"].strip()
        start = float(entry["start"])
        end = float(entry["end"])
        total_duration = max(end - start, _MIN_ENTRY_DURATION)

        chunks = self._split_text_chunks(text, self.max_entry_chars)
        if len(chunks) <= 1:
            return [entry]

        total_chars = sum(len(c) for c in chunks)
        if total_chars <= 0:
            return [entry]

        split_entries: list[dict] = []
        current = start
        for idx, chunk in enumerate(chunks):
            if idx == len(chunks) - 1:
                chunk_end = end
            else:
                ratio = len(chunk) / total_chars
                chunk_duration = max(total_duration * ratio, _MIN_ENTRY_DURATION)
                chunk_end = min(current + chunk_duration, end)
            split_entries.append(
                {
                    "start": current,
                    "end": chunk_end,
                    "text": chunk,
                }
            )
            current = chunk_end

        if split_entries:
            split_entries[-1]["end"] = end
        return split_entries

    @staticmethod
    def _split_text_chunks(text: str, max_chars: int) -> list[str]:
        """将文本切分为不超过 max_chars 的片段，优先在逗号处断开。"""
        text = text.strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]

        cut = SubtitleGenerator._find_break_index(text, max_chars)
        if cut <= 0:
            cut = max_chars

        first = text[:cut].rstrip()
        rest = text[cut:].lstrip()
        if not first:
            first = text[:max_chars]
            rest = text[max_chars:].lstrip()

        return [first] + SubtitleGenerator._split_text_chunks(rest, max_chars)

    @staticmethod
    def _find_break_index(text: str, max_chars: int) -> int:
        """在 max_chars 范围内寻找最佳断点，逗号优先。"""
        window_end = min(max_chars, len(text))
        window = text[:window_end]

        # 优先在逗号处断开（标点保留在前段），避免在「」内断开
        for i in range(len(window) - 1, 0, -1):
            if window[i] == "，":
                if not SubtitleGenerator._is_inside_quotes(window, i):
                    return i + 1

        # 其次在其他句读标点处断开
        for i in range(len(window) - 1, 0, -1):
            if window[i] in _PREFERRED_BREAK_CHARS:
                if not SubtitleGenerator._is_inside_quotes(window, i):
                    return i + 1

        # 硬切时若落点位于引号内，向前寻找引号外位置
        if not SubtitleGenerator._is_inside_quotes(window, window_end - 1):
            return max_chars
        for i in range(window_end - 1, max(1, window_end // 2), -1):
            if not SubtitleGenerator._is_inside_quotes(window, i):
                return i
        return max_chars

    # ------------------------------------------------------------------
    # SRT 渲染
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_entry_text(text: str) -> str:
        """去掉换行，保证每条 SRT 只占一行。"""
        return re.sub(r"\s+", " ", (text or "").replace("\n", " ").replace("\r", " ")).strip()

    @staticmethod
    def _render_srt(entries: list[dict]) -> str:
        """将字幕条目列表渲染为标准 SRT 格式字符串。"""
        lines: list[str] = []
        for idx, entry in enumerate(entries, start=1):
            start_ts = SubtitleGenerator._format_timestamp(entry["start"])
            end_ts = SubtitleGenerator._format_timestamp(entry["end"])
            lines.append(str(idx))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(SubtitleGenerator._sanitize_entry_text(entry["text"]))
            lines.append("")  # SRT 条目之间的空行
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """将秒数转换为 SRT 时间戳格式 HH:MM:SS,mmm。

        Args:
            seconds: 时间（秒），非负。

        Returns:
            格式化后的时间戳字符串，如 "00:01:23,456"。
        """
        seconds = max(0.0, seconds)
        total_ms = int(round(seconds * 1000))
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, ms = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    @staticmethod
    def _ends_with_punctuation(text: str) -> bool:
        """检查文本是否以中文标点结尾。"""
        return bool(text) and text[-1] in _PUNCTUATION
