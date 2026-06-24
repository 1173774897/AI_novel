"""IdeaPlanner - 将用户灵感转换为视频方案"""
from __future__ import annotations
import logging
from src.llm.llm_client import LLMClient
from src.scriptplan.json_utils import parse_llm_json
from src.scriptplan.models import VideoIdea

log = logging.getLogger("scriptplan")


def _fallback_idea(target_duration: int, inspiration: str) -> VideoIdea:
    """LLM 解析失败时的启发式默认方案（尽量贴合灵感关键词）。"""
    comedy_kw = ("搞笑", "喜剧", "轻喜剧", "幽默", "沙雕", "橘猫", "猫咪", "猫")
    is_comedy = any(k in inspiration for k in comedy_kw)
    segment_count = max(6, min(12, target_duration // 5))
    if is_comedy:
        return VideoIdea(
            video_type="搞笑",
            target_duration=target_duration,
            segment_count=segment_count,
            rhythm="3秒钩子+多段笑点递进+1段收尾",
            twist_type="无反转",
            ending_type="评论钩子",
            tone="搞笑",
        )
    return VideoIdea(
        video_type="悬疑反转",
        target_duration=target_duration,
        segment_count=segment_count,
        rhythm="3秒钩子+3段推进+1段反转+1段收尾",
        twist_type="无反转",
        ending_type="评论钩子",
        tone="悬疑",
    )


class IdeaPlanner:
    """将用户灵感转化为结构化视频方案。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, inspiration: str, target_duration: int = 45) -> VideoIdea:
        """从灵感生成视频方案。

        Args:
            inspiration: 用户输入的灵感/创意/故事梗概
            target_duration: 目标视频时长(秒), 默认45秒

        Returns:
            VideoIdea 视频方案
        """
        system_prompt = (
            "你是一位顶级短视频策划师，专门策划抖音/小红书爆款内容。\n"
            "你的任务是将用户的灵感转化为一个结构化的短视频方案。\n\n"
            "【必须遵守的短视频法则】\n"
            "1. 前3秒定生死：必须用悬念/冲突/反常识来抓住观众\n"
            "2. 每3-5秒一个信息点：观众注意力极短，不能有空白\n"
            "3. 结尾必须有互动钩子：让观众评论/点赞/转发\n"
            "4. 30-60秒最佳：太短没深度，太长会划走\n"
            "5. 一个视频只讲一个核心点：不要贪多\n\n"
            "请返回严格的 JSON 格式：\n"
            "{\n"
            '  "video_type": "悬疑反转/情感共鸣/爽文快节奏/知识科普/恐怖/搞笑 之一",\n'
            f'  "target_duration": {target_duration},\n'
            '  "segment_count": 5到8之间的整数,\n'
            '  "rhythm": "节奏描述，如：3秒钩子+3段推进+1段反转+1段收尾",\n'
            '  "twist_type": "身份反转/时间反转/视角反转/逻辑反转/无反转 之一",\n'
            '  "ending_type": "评论钩子/悬念留白/情感升华/反问互动 之一",\n'
            '  "tone": "悬疑/爽感/温情/恐怖/搞笑/紧张/治愈 之一"\n'
            "}"
        )

        user_prompt = (
            f"【用户灵感】\n{inspiration}\n\n"
            f"【目标时长】{target_duration}秒\n\n"
            "请为这个灵感设计一个短视频方案。"
        )

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            json_mode=True,
            max_tokens=512,
        )

        # 解析 JSON
        data = parse_llm_json(response.content)
        if data is None:
            log.error("IdeaPlanner 返回非 JSON: %s", (response.content or "")[:200])
            return _fallback_idea(target_duration, inspiration)

        return VideoIdea(
            video_type=data.get("video_type", "悬疑反转"),
            target_duration=data.get("target_duration", target_duration),
            segment_count=max(4, min(10, data.get("segment_count", 6))),
            rhythm=data.get("rhythm", ""),
            twist_type=data.get("twist_type", "无反转"),
            ending_type=data.get("ending_type", "评论钩子"),
            tone=data.get("tone", "悬疑"),
        )
