from __future__ import annotations

from pathlib import Path
from typing import Any


from src.tts.tts_params import combine_percent


class TTSTool:
    """封装 TTS 配音 + 字幕模块，供 Agent 节点调用。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._engine: Any = None
        self._sub_gen: Any = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            from src.tts.tts_engine import TTSEngine

            self._engine = TTSEngine(self.config["tts"])
        return self._engine

    def _get_sub_gen(self) -> Any:
        if self._sub_gen is None:
            from src.tts.subtitle_generator import SubtitleGenerator

            self._sub_gen = SubtitleGenerator(self.config.get("subtitle", {}))
        return self._sub_gen

    def run(
        self,
        text: str,
        audio_path: Path,
        srt_path: Path,
        rate: str | None = None,
        volume: str | None = None,
    ) -> tuple[Path, Path]:
        engine = self._get_engine()
        base_tts = self.config.get("tts", {})
        # 动态 TTS 参数：config.yaml 为基准，rate/volume 为情感偏移量（相加）
        if rate or volume:
            tts_cfg = dict(base_tts)
            if rate:
                tts_cfg["rate"] = combine_percent(
                    base_tts.get("rate", "+0%"), rate
                )
            if volume:
                tts_cfg["volume"] = combine_percent(
                    base_tts.get("volume", "+0%"), volume
                )
            from src.tts.tts_engine import TTSEngine

            engine = TTSEngine(tts_cfg)

        audio, word_boundaries = engine.synthesize(text, audio_path)

        subtitle_cfg = self.config.get("subtitle", {})
        if subtitle_cfg.get("enabled", True):
            self._get_sub_gen().generate_srt(word_boundaries, text, srt_path)
        else:
            srt_path.parent.mkdir(parents=True, exist_ok=True)
            srt_path.write_text("", encoding="utf-8")

        return audio_path, srt_path
