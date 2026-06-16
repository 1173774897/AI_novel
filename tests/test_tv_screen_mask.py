"""CRT 屏幕蒙版测试。"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.video.tv_screen_mask import (
    generate_crt_screen_mask,
    generate_rounded_rect_mask,
    load_screen_mask,
    render_calibration_preview,
    save_screen_mask,
)

pytestmark = pytest.mark.signature


class TestGenerateRoundedRectMask:
    def test_corners_are_rounded(self):
        mask = generate_rounded_rect_mask(100, 80, corner_radius=16)
        assert mask.size == (100, 80)
        assert mask.getpixel((0, 0)) == 0
        assert mask.getpixel((50, 40)) == 255

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="蒙版尺寸无效"):
            generate_rounded_rect_mask(0, 10, corner_radius=4)


class TestCrtEdgeBow:
    def test_top_edge_bows_outward_at_center(self):
        mask = generate_crt_screen_mask(
            100, 80, corner_radius=0, edge_bow={"top": 8, "bottom": 0, "left": 0, "right": 0}
        )
        assert mask.getpixel((50, 0)) == 255
        assert mask.getpixel((0, 0)) == 0
        assert mask.getpixel((0, 8)) == 255

class TestSaveAndLoadScreenMask:
    def test_roundtrip(self, tmp_path: Path):
        out = tmp_path / "mask.png"
        save_screen_mask(out, 120, 90, corner_radius=12, feather=0)
        loaded = load_screen_mask(out)
        assert loaded.size == (120, 90)


class TestCalibrationPreview:
    def test_renders_overlay(self, tmp_path: Path):
        frame = tmp_path / "frame.png"
        Image.new("RGB", (200, 150), color=(40, 40, 40)).save(frame)
        mask = tmp_path / "mask.png"
        save_screen_mask(mask, 80, 60, corner_radius=8)
        debug = tmp_path / "debug.png"
        render_calibration_preview(
            frame,
            screen_x=30,
            screen_y=20,
            mask_path=mask,
            output_path=debug,
        )
        assert debug.exists()
        with Image.open(debug) as img:
            assert img.size == (200, 150)
