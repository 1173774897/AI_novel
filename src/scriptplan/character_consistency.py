"""create-video 角色外观一致性：CharacterTracker + 首镜 img2img 锚定。"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.promptgen.character_tracker import CharacterTracker
from src.scriptplan.models import VisualBible

log = logging.getLogger("scriptplan")

_ANCHOR_META_FILE = "character_anchor.json"
_CAT_ALIASES = ("橘猫", "胖橘猫", "猫")


def _default_aliases(name: str, prompt_anchor: str) -> list[str]:
    aliases: list[str] = []
    anchor_lower = (prompt_anchor or "").lower()
    if "猫" in name or "cat" in anchor_lower:
        aliases.extend(_CAT_ALIASES)
    return aliases


def build_tracker_from_visual_bible(visual_bible: VisualBible | None) -> tuple[CharacterTracker, list[str]]:
    """从 visual_bible 构建 CharacterTracker，并返回 seeded 名称列表（含别名）。"""
    tracker = CharacterTracker()
    seeded_names: list[str] = []
    if not visual_bible or not visual_bible.characters:
        return tracker, seeded_names

    entries: list[dict[str, str]] = []
    for ch in visual_bible.characters:
        if not isinstance(ch, dict):
            continue
        name = str(ch.get("name", "")).strip()
        anchor = str(ch.get("prompt_anchor", "")).strip()
        if not name or not anchor:
            continue
        entries.append({"name": name, "desc": anchor})
        seeded_names.append(name)
        explicit_aliases = ch.get("aliases") or []
        if isinstance(explicit_aliases, list):
            for alias in explicit_aliases:
                alias = str(alias).strip()
                if alias:
                    entries.append({"name": alias, "desc": anchor})
                    seeded_names.append(alias)
        for alias in _default_aliases(name, anchor):
            if alias not in seeded_names:
                entries.append({"name": alias, "desc": anchor})
                seeded_names.append(alias)

    tracker.seed_characters(entries, canonical=True)
    return tracker, seeded_names


class DirectorCharacterConsistency:
    """导演流水线角色一致性控制器。"""

    def __init__(
        self,
        visual_bible: VisualBible | None,
        *,
        enabled: bool = True,
        anchor_img2img: bool = True,
        force_prompt_anchor: bool = True,
        anchor_segment_id: int | None = None,
    ) -> None:
        self.enabled = enabled
        self.anchor_img2img = anchor_img2img
        self.force_prompt_anchor = force_prompt_anchor
        self.anchor_segment_id = anchor_segment_id or 1
        self.visual_bible = visual_bible
        self.tracker, self.seeded_names = build_tracker_from_visual_bible(visual_bible)

    @classmethod
    def from_config(cls, config: dict, visual_bible: VisualBible | None) -> DirectorCharacterConsistency:
        director = config.get("director") or {}
        cc = director.get("character_consistency") or {}
        return cls(
            visual_bible,
            enabled=bool(cc.get("enabled", True)),
            anchor_img2img=bool(cc.get("anchor_img2img", True)),
            force_prompt_anchor=bool(cc.get("force_prompt_anchor", True)),
            anchor_segment_id=cc.get("anchor_segment_id"),
        )

    def resolve_characters(self, visual: str) -> list[str]:
        if not self.enabled or not visual:
            return []
        if self.seeded_names:
            matches = [n for n in self.seeded_names if n in visual]
            if matches:
                matches.sort(key=len, reverse=True)
                return [matches[0]]
        found = self.tracker.resolve_segment_characters(
            visual,
            seeded_names=self.seeded_names or None,
        )
        if found:
            return found
        if self.visual_bible and len(self.visual_bible.characters) == 1:
            primary = self.visual_bible.characters[0]
            if isinstance(primary, dict):
                name = str(primary.get("name", "")).strip()
                if name and re.search(r"猫|它", visual):
                    return [name]
        return []

    def scene_anchor_text(self) -> str:
        if not self.visual_bible:
            return ""
        return str(self.visual_bible.scene_anchor or "").strip()

    @staticmethod
    def _prepend_if_missing(prompt: str, prefix: str) -> str:
        prefix = (prefix or "").strip()
        if not prefix:
            return prompt
        if prefix.lower() in prompt.lower():
            return prompt
        return f"{prefix}, {prompt}"

    def enrich_image_prompt(self, prompt: str, visual: str) -> str:
        if not self.enabled:
            return prompt
        result = prompt
        if self.force_prompt_anchor:
            chars = self.resolve_characters(visual)
            anchor_text = self.tracker.get_character_prompt(chars) if chars else ""
            if not anchor_text and self.visual_bible:
                anchors = [
                    str(ch.get("prompt_anchor", "")).strip()
                    for ch in self.visual_bible.characters
                    if isinstance(ch, dict) and ch.get("prompt_anchor")
                ]
                if len(anchors) == 1:
                    anchor_text = anchors[0]
            if anchor_text:
                result = self._prepend_if_missing(result, anchor_text)
        scene = self.scene_anchor_text()
        if scene:
            result = self._prepend_if_missing(result, scene)
        return result

    def build_anchor_edit_prompt(self, scene_prompt: str) -> str:
        """首镜 img2img：保持参考图角色与场景布局，替换动作。"""
        style = ""
        if self.visual_bible and self.visual_bible.style_tags:
            style = f", {self.visual_bible.style_tags}"
        scene_note = ""
        scene = self.scene_anchor_text()
        if scene:
            scene_note = (
                f" Keep the same room layout, lighting, and key prop positions as the reference "
                f"({scene})."
            )
        return (
            "Keep the exact same character appearance, face, fur color, body shape and art style "
            f"as the reference image.{scene_note} New scene and action: {scene_prompt}{style}, "
            "highly detailed, cinematic lighting, 4K"
        )

    def is_anchor_segment(self, seg_id: int) -> bool:
        return seg_id == self.anchor_segment_id

    def should_use_anchor_i2img(self, seg_id: int, run_dir: Path) -> bool:
        if not self.enabled or not self.anchor_img2img:
            return False
        if self.is_anchor_segment(seg_id):
            return False
        return self.get_anchor_path(run_dir) is not None

    @staticmethod
    def get_anchor_path(run_dir: Path) -> Path | None:
        meta_path = run_dir / _ANCHOR_META_FILE
        if meta_path.is_file():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                rel = data.get("anchor_image")
                if rel:
                    path = run_dir / rel
                    if path.is_file():
                        return path
            except (json.JSONDecodeError, OSError):
                pass
        fallback = run_dir / "img_001.png"
        if fallback.is_file():
            return fallback
        return None

    def set_anchor(self, run_dir: Path, image_path: Path, seg_id: int) -> None:
        if not self.is_anchor_segment(seg_id):
            return
        rel = image_path.name if image_path.parent.resolve() == run_dir.resolve() else str(image_path)
        meta = {
            "anchor_segment_id": seg_id,
            "anchor_image": rel if isinstance(rel, str) and not rel.startswith("/") else image_path.name,
        }
        (run_dir / _ANCHOR_META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("角色锚定首镜: segment %d -> %s", seg_id, meta["anchor_image"])

    def merged_negative_prompt(self, base_negative: str) -> str:
        if not self.visual_bible or not self.visual_bible.negative_prompt:
            return base_negative
        vb_neg = self.visual_bible.negative_prompt.strip()
        if not base_negative:
            return vb_neg
        if vb_neg.lower() in base_negative.lower():
            return base_negative
        return f"{base_negative}, {vb_neg}"

    def supports_reference_image(self, imagegen_config: dict) -> bool:
        return imagegen_config.get("backend") == "jimeng-cli"
