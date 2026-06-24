"""create-video 角色一致性测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scriptplan.character_consistency import (
    DirectorCharacterConsistency,
    build_tracker_from_visual_bible,
)
from src.scriptplan.models import VisualBible


def _cat_bible() -> VisualBible:
    return VisualBible(
        style_tags="realistic, cozy home",
        scene_anchor="modest apartment entryway, beige doormat, cardboard box on floor",
        negative_prompt="cartoon, blurry",
        characters=[
            {
                "name": "大橘",
                "prompt_anchor": "a chubby orange tabby cat, fluffy fur, green eyes",
            }
        ],
    )


@pytest.mark.signature
def test_build_tracker_seeds_cat_aliases():
    tracker, seeded = build_tracker_from_visual_bible(_cat_bible())
    assert "大橘" in seeded
    assert "橘猫" in seeded
    assert "猫" in seeded
    assert tracker.known_characters["猫"] == tracker.known_characters["大橘"]


@pytest.mark.signature
def test_resolve_single_cat_protagonist_from_猫():
    cc = DirectorCharacterConsistency(_cat_bible())
    chars = cc.resolve_characters("猫用嘴咬住胶带一角，奋力向后拖")
    assert chars == ["猫"]


@pytest.mark.signature
def test_enrich_image_prompt_prepends_scene_and_anchor():
    cc = DirectorCharacterConsistency(_cat_bible())
    base = "biting tape, close-up, indoor scene"
    enriched = cc.enrich_image_prompt(base, "猫咬住胶带")
    assert enriched.startswith("modest apartment entryway")
    assert "a chubby orange tabby cat" in enriched
    assert "biting tape" in enriched


@pytest.mark.signature
def test_anchor_meta_roundtrip(tmp_path: Path):
    cc = DirectorCharacterConsistency(_cat_bible(), anchor_segment_id=1)
    img = tmp_path / "img_001.png"
    img.write_bytes(b"png")
    cc.set_anchor(tmp_path, img, 1)
    assert cc.get_anchor_path(tmp_path) == img


@pytest.mark.signature
def test_should_use_i2i_after_anchor(tmp_path: Path):
    cc = DirectorCharacterConsistency(_cat_bible(), anchor_img2img=True)
    img = tmp_path / "img_001.png"
    img.write_bytes(b"png")
    cc.set_anchor(tmp_path, img, 1)
    assert cc.should_use_anchor_i2img(1, tmp_path) is False
    assert cc.should_use_anchor_i2img(6, tmp_path) is True


@pytest.mark.signature
def test_merged_negative_prompt():
    cc = DirectorCharacterConsistency(_cat_bible())
    merged = cc.merged_negative_prompt("nsfw, gore")
    assert "cartoon" in merged
    assert "nsfw" in merged


@pytest.mark.signature
def test_build_anchor_edit_prompt_keeps_reference_instruction():
    cc = DirectorCharacterConsistency(_cat_bible())
    prompt = cc.build_anchor_edit_prompt("cat bites tape, close-up")
    assert "reference image" in prompt.lower()
    assert "room layout" in prompt.lower()
    assert "cat bites tape" in prompt
