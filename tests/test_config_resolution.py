"""config_manager 视频分辨率与 imagegen.ratio 同步测试。"""

import pytest

from src.config_manager import (
    _apply_video_resolution_from_imagegen,
    resolution_for_image_ratio,
)


def _base_cfg(**overrides) -> dict:
    cfg = {
        "segmenter": {},
        "promptgen": {},
        "imagegen": {"ratio": "16:9"},
        "tts": {},
        "video": {"resolution": [1080, 1920]},
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.signature
def test_resolution_for_image_ratio_landscape():
    assert resolution_for_image_ratio("16:9") == [1920, 1080]


@pytest.mark.signature
def test_auto_resolution_syncs_landscape():
    cfg = _base_cfg()
    cfg["video"]["auto_resolution"] = True
    _apply_video_resolution_from_imagegen(cfg)
    assert cfg["video"]["resolution"] == [1920, 1080]


@pytest.mark.signature
def test_auto_resolution_off_keeps_manual():
    cfg = _base_cfg()
    cfg["video"]["auto_resolution"] = False
    _apply_video_resolution_from_imagegen(cfg)
    assert cfg["video"]["resolution"] == [1080, 1920]
