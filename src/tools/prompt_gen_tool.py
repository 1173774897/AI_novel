from __future__ import annotations

from typing import Any


class PromptGenTool:
    """封装 Prompt 生成模块，供 Agent 节点调用。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._gen: Any = None

    def _get_gen(self) -> Any:
        if self._gen is None:
            from src.promptgen.prompt_generator import PromptGenerator

            prompt_cfg = dict(self.config.get("promptgen", {}))
            global_llm = self.config.get("llm", {})
            module_llm = prompt_cfg.get("llm", {})
            prompt_cfg["llm"] = {**global_llm, **module_llm}
            self._gen = PromptGenerator(prompt_cfg)
        return self._gen

    def set_style(self, style_name: str) -> None:
        self._get_gen().set_style(style_name)

    def seed_characters(self, characters: list[dict], *, canonical: bool = False) -> int:
        """将 ContentAnalyzer 角色描述注入 CharacterTracker。"""
        if not characters:
            return 0
        return self._get_gen().seed_characters(characters, canonical=canonical)

    def set_pov_narrator(self, name: str | None) -> None:
        self._get_gen().set_pov_narrator(name)

    def set_era(self, era: str | None) -> None:
        self._get_gen().set_era(era)

    def run(
        self,
        text: str,
        segment_index: int,
        full_text: str | None = None,
        prev_text: str | None = None,
    ) -> str:
        gen = self._get_gen()
        if full_text:
            gen.set_full_text(full_text)
        return gen.generate(
            text, segment_index=segment_index, prev_text=prev_text
        )

    def run_alternate(
        self,
        text: str,
        segment_index: int,
        full_text: str | None = None,
        prev_text: str | None = None,
        *,
        variant: int = 0,
    ) -> str:
        gen = self._get_gen()
        if full_text:
            gen.set_full_text(full_text)
        return gen.generate_alternate(
            text, segment_index=segment_index, variant=variant, prev_text=prev_text
        )

    def run_video_prompt(
        self,
        text: str,
        segment_index: int,
        prev_text: str | None = None,
    ) -> str:
        gen = self._get_gen()
        return gen.generate_video_prompt(
            text, segment_index=segment_index, prev_text=prev_text
        )
