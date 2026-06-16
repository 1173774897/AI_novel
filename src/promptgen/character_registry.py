"""分集/系列级角色外观注册表 — 跨集保持一致。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.logger import log

_REGISTRY_VERSION = 1


class CharacterRegistry:
    """持久化角色名 → 外观描述，供多集视频共享。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.characters: dict[str, dict[str, Any]] = {}

    @classmethod
    def load(cls, path: Path) -> CharacterRegistry:
        reg = cls(path)
        if not path.exists():
            return reg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("读取角色注册表失败 %s: %s", path, exc)
            return reg
        if not isinstance(data, dict):
            return reg
        raw = data.get("characters") or {}
        if isinstance(raw, dict):
            reg.characters = {
                str(k): dict(v) if isinstance(v, dict) else {"name": str(k), "desc": str(v)}
                for k, v in raw.items()
            }
        return reg

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _REGISTRY_VERSION,
            "characters": self.characters,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def merge_character_list(
        self,
        items: list[dict[str, Any]],
        *,
        episode: str | int | None = None,
    ) -> int:
        """合并角色列表：已有描述不被覆盖，仅补全新角色或空描述。"""
        updated = 0
        ep_tag = str(episode) if episode is not None else None
        for item in items or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            desc = str(item.get("desc", "")).strip()
            if not name:
                continue
            entry = self.characters.get(name)
            if entry is None:
                self.characters[name] = {
                    "name": name,
                    "desc": desc,
                    "episodes": [ep_tag] if ep_tag else [],
                }
                updated += 1
                continue
            if desc and not str(entry.get("desc", "")).strip():
                entry["desc"] = desc
                updated += 1
            if ep_tag and ep_tag not in entry.setdefault("episodes", []):
                entry["episodes"].append(ep_tag)
        return updated

    def merge_tracker(self, tracker: Any, *, episode: str | int | None = None) -> int:
        """从 CharacterTracker 合并外观（不覆盖已有 canonical 描述）。"""
        known = getattr(tracker, "known_characters", None) or {}
        items = [{"name": n, "desc": d} for n, d in known.items() if d]
        return self.merge_character_list(items, episode=episode)

    def to_seed_list(self) -> list[dict[str, str]]:
        """供 PromptGenerator / CharacterTracker 预填。"""
        result: list[dict[str, str]] = []
        for name, entry in self.characters.items():
            desc = str(entry.get("desc", "")).strip()
            if desc:
                result.append({"name": name, "desc": desc})
        result.sort(key=lambda x: x["name"])
        return result

    def apply_canonical_to(self, characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """本集 ContentAnalyzer 结果与注册表对齐：注册表描述优先。"""
        merged: dict[str, dict[str, Any]] = {}
        for char in characters or []:
            if not isinstance(char, dict):
                continue
            name = str(char.get("name", "")).strip()
            if name:
                merged[name] = dict(char)

        for name, entry in self.characters.items():
            canonical = str(entry.get("desc", "")).strip()
            if not canonical:
                continue
            if name in merged:
                merged[name]["desc"] = canonical
            else:
                merged[name] = {"name": name, "desc": canonical}

        return list(merged.values())
