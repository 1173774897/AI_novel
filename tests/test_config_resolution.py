"""config_manager 视频分辨率与 imagegen.ratio 同步、流水线配置隔离测试。"""

import pytest

from src.config_manager import (
    _apply_video_resolution_from_imagegen,
    get_mode_videogen,
    resolve_pipeline_config,
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


def _pipeline_base(**overrides) -> dict:
    cfg = {
        "segmenter": {},
        "promptgen": {},
        "imagegen": {"ratio": "16:9"},
        "tts": {},
        "video": {"resolution": [1080, 1920], "auto_resolution": True},
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.signature
def test_get_mode_videogen_agent_section():
    cfg = _pipeline_base(
        agent={"videogen": {"backend": "kling"}},
        director={"videogen": {"backend": "seedance"}},
    )
    assert get_mode_videogen(cfg, "agent") == {"backend": "kling"}
    assert get_mode_videogen(cfg, "director") == {"backend": "seedance"}


@pytest.mark.signature
def test_get_mode_videogen_director_ignores_root_legacy():
    cfg = _pipeline_base(videogen={"backend": "kling"})
    assert get_mode_videogen(cfg, "director") == {}


@pytest.mark.signature
def test_get_mode_videogen_agent_legacy_root_fallback():
    cfg = _pipeline_base(videogen={"backend": "minimax"})
    assert get_mode_videogen(cfg, "agent") == {"backend": "minimax"}


@pytest.mark.signature
def test_resolve_pipeline_config_isolates_videogen():
    cfg = _pipeline_base(
        agent={"videogen": {"backend": "kling", "duration": 5}},
        director={"videogen": {"backend": "seedance", "duration": 8}},
    )
    agent_cfg = resolve_pipeline_config(cfg, "agent")
    director_cfg = resolve_pipeline_config(cfg, "director")
    assert agent_cfg["videogen"] == {"backend": "kling", "duration": 5}
    assert director_cfg["videogen"] == {"backend": "seedance", "duration": 8}


@pytest.mark.signature
def test_resolve_pipeline_config_director_imagegen_override():
    cfg = _pipeline_base(
        imagegen={"backend": "together", "ratio": "16:9"},
        director={
            "imagegen": {"backend": "jimeng-cli", "ratio": "9:16"},
            "videogen": {"backend": "seedance"},
        },
    )
    resolved = resolve_pipeline_config(cfg, "director")
    assert resolved["imagegen"]["backend"] == "jimeng-cli"
    assert resolved["imagegen"]["ratio"] == "9:16"
    assert resolved["videogen"]["backend"] == "seedance"
