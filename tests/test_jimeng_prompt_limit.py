"""即梦 prompt 长度保护与分镜角色裁剪测试。"""

import pytest

from src.imagegen.moderation import (
    is_jimeng_retryable_error,
    retry_image_prompt_after_failure,
    truncate_image_prompt_for_jimeng,
)
from src.promptgen.prompt_generator import PromptGenerator

pytestmark = pytest.mark.signature

_WUJIN_CHARS = [
    {
        "name": "303",
        "desc": "约28岁男性，中等偏瘦，眉头微锁透着紧张，深灰色居家T恤，手握菜刀。" * 3,
    },
    {"name": "501", "desc": "约42岁男性，消瘦佝偻，黑框眼镜，浅蓝色格子衬衫。" * 3},
    {"name": "302", "desc": "约30岁男性，微胖结实，灰色polo衫，手机从不离手。" * 3},
    {"name": "404", "desc": "魁梧壮实，黑色背心，粗金链。" * 3},
    {"name": "503", "desc": "魁梧壮实，工装夹克，手持消防斧。" * 3},
    {"name": "202", "desc": "娇小女性，粉色睡裙。" * 3},
    {"name": "301", "desc": "焦虑男性，蓝白条纹睡衣。" * 3},
    {"name": "402", "desc": "微胖男性，米色针织开衫。" * 3},
]

_WUJIN_SEG_CHAT = (
    "501的业主突然在群里发了消息：【别出去，楼道里有杀人狂！】\n"
    "302：【不是，你大晚上的，这么吓人好吗？】\n"
    "我有些不爽的打字：【怎么又吵起来了】"
)


class TestJimengPromptTruncate:
    def test_invalidnode_is_retryable(self):
        assert is_jimeng_retryable_error(
            "api error: ret=1046, message=InvalidNode, logid=abc"
        )

    def test_truncate_caps_length(self):
        long_prompt = "a cinematic scene, " + "x" * 2000
        out = truncate_image_prompt_for_jimeng(long_prompt, max_chars=1200)
        assert len(out) <= 1200

    def test_truncate_removes_cast_blocks_first(self):
        prompt = (
            "anime scene, dark apartment, "
            "【本段相关角色，外观保持一致】\n"
            + ("501：" + "描述" * 80 + "\n") * 5
            + ", subtle horror, no gore"
        )
        out = truncate_image_prompt_for_jimeng(prompt, max_chars=400)
        assert len(out) <= 400
        assert "501：" not in out or len(out) <= 400

    def test_retry_after_invalidnode_shrinks_more(self):
        prompt = "scene " + "y" * 1800
        a = retry_image_prompt_after_failure(
            prompt, 0, "ret=1046 InvalidNode"
        )
        b = retry_image_prompt_after_failure(
            prompt, 1, "ret=1046 InvalidNode"
        )
        assert len(a) <= 1200
        assert len(b) <= 900

    def test_retry_generation_failed_strips_horror(self):
        prompt = "dark hall, blood trail, 血渍, hollow eyes staring"
        out = retry_image_prompt_after_failure(
            prompt, 0, "generation failed: final generation failed"
        )
        assert "血" not in out
        assert "blood trail" not in out.lower()
        assert "hollow eyes" not in out.lower()


class TestSegmentCastBible:
    def _make_gen(self):
        cfg = {
            "style": "anime",
            "llm": {"provider": "none"},
            "pov_mode": "author",
        }
        gen = PromptGenerator(cfg)
        gen.seed_characters(_WUJIN_CHARS)
        return gen

    def test_cast_bible_only_mentions_segment_characters(self):
        gen = self._make_gen()
        bible = gen._build_cast_bible(_WUJIN_SEG_CHAT)
        assert "501" in bible
        assert "302" in bible
        assert "503" not in bible
        assert "402" not in bible

    def test_cast_desc_truncated(self):
        gen = self._make_gen()
        bible = gen._build_cast_bible("501在群里说话，我看着他。")
        assert "…" in bible
        for line in bible.splitlines():
            if "：" in line:
                assert len(line) <= 110

    def test_local_prompt_under_jimeng_limit(self):
        gen = self._make_gen()
        gen.set_full_text(_WUJIN_SEG_CHAT * 30)
        prompt = gen.generate(_WUJIN_SEG_CHAT, segment_index=0)
        assert len(prompt) <= 1200
