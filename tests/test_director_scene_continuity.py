"""导演模式场景连贯性：scene_anchor + video_prompt 从首帧派生。"""

from unittest.mock import MagicMock, patch

import pytest

from src.director_pipeline import DirectorPipeline, DirectorPromptError
from src.scriptplan.models import MotionType, VisualBible

pytestmark = pytest.mark.signature


def _pipeline() -> DirectorPipeline:
    return DirectorPipeline(
        config={
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "director": {"videogen": {"backend": "jimeng-cli"}},
        },
        workspace="/tmp/test_director_scene",
    )


class TestDeriveVideoPrompt:
    def test_derives_from_image_prompt_with_motion(self):
        image_prompt = (
            "modest entryway, a chubby orange tabby cat sniffing a cardboard box, 4K"
        )
        video = DirectorPipeline._derive_video_prompt_from_image(
            image_prompt, MotionType.PUSH_IN
        )
        assert image_prompt.split(", 4K")[0] in video
        assert "dolly in" in video.lower()
        assert "stable character appearance" in video

    def test_empty_image_prompt_raises(self):
        with pytest.raises(DirectorPromptError):
            DirectorPipeline._derive_video_prompt_from_image("", MotionType.STATIC)


class TestVisualToPromptFailFast:
    @patch.object(DirectorPipeline, "_llm_required_for_visual", return_value=True)
    @patch.object(DirectorPipeline, "_get_llm")
    def test_raises_on_llm_error(self, mock_get_llm, _mock_required):
        mock_get_llm.return_value.chat.side_effect = RuntimeError("402 balance")
        pipe = _pipeline()
        with pytest.raises(DirectorPromptError, match="LLM 图片 prompt 翻译失败"):
            pipe._visual_to_prompt("猫绕着箱子嗅闻", seg_id=2)

    @patch.object(DirectorPipeline, "_llm_required_for_visual", return_value=True)
    @patch.object(DirectorPipeline, "_get_llm")
    def test_raises_on_empty_llm_response(self, mock_get_llm, _mock_required):
        mock_get_llm.return_value.chat.return_value = MagicMock(content="  ")
        pipe = _pipeline()
        with pytest.raises(DirectorPromptError, match="LLM 返回空图片 prompt"):
            pipe._visual_to_prompt("猫绕着箱子嗅闻", seg_id=2)

    @patch.object(DirectorPipeline, "_llm_required_for_visual", return_value=False)
    def test_no_llm_uses_local_rules(self, _mock_required):
        pipe = _pipeline()
        vb = VisualBible(
            scene_anchor="same entryway, beige doormat",
            style_tags="cozy",
            characters=[{"name": "猫", "prompt_anchor": "orange tabby cat"}],
        )
        prompt = pipe._visual_to_prompt("猫在玄关", seg_id=1, visual_bible=vb)
        assert "same entryway" in prompt
        assert "orange tabby cat" in prompt


class TestBuildBibleContext:
    def test_includes_scene_anchor(self):
        vb = VisualBible(
            scene_anchor="modest apartment entryway, cardboard box",
            style_tags="realistic",
            characters=[{"name": "大橘", "prompt_anchor": "orange tabby cat"}],
        )
        ctx = DirectorPipeline._build_bible_context(vb)
        assert "固定场景锚点" in ctx
        assert "modest apartment entryway" in ctx
        assert "大橘" in ctx
