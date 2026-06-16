"""简单分段器 - 基于规则的文本分段，无外部 API 依赖"""

import re

from src.segmenter.text_segmenter import TextSegmenter
from src.tts.text_split import split_sentences


class SimpleSegmenter(TextSegmenter):
    """基于规则的文本分段器。

    分段逻辑:
      1. 按段落（双换行）拆分原始文本
      2. 将每个段落进一步拆分为完整句子（引号闭合处优先断句）
      3. 逐句合并，直到达到 max_chars 上限后切出一个片段
      4. 片段不会在句子中间断开
    """

    def __init__(self, config: dict) -> None:
        self.max_chars: int = config.get("max_chars", 200)
        self.min_chars: int = config.get("min_chars", 50)

    def segment(self, text: str) -> list[dict]:
        """将文本按规则拆分为多个片段。"""
        if not text or not text.strip():
            return []

        sentences = self._split_to_sentences(text)
        segments = self._merge_sentences(sentences)
        return [{"text": seg, "index": idx} for idx, seg in enumerate(segments)]

    @staticmethod
    def _split_to_sentences(text: str) -> list[str]:
        """将文本拆分为句子列表（段落 → 引号感知断句）。"""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        sentences: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            sentences.extend(split_sentences(para))
        return sentences

    def _merge_sentences(self, sentences: list[str]) -> list[str]:
        """将句子列表合并为满足长度约束的片段列表。"""
        if not sentences:
            return []

        segments: list[str] = []
        buffer = ""

        for sentence in sentences:
            if buffer and len(buffer) + len(sentence) > self.max_chars and len(buffer) >= self.min_chars:
                segments.append(buffer)
                buffer = sentence
            else:
                buffer += sentence

        if buffer:
            if len(buffer) < self.min_chars and segments:
                segments[-1] += buffer
            else:
                segments.append(buffer)

        return segments
