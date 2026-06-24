"""ScriptPlanner - 将视频方案转换为结构化脚本"""
from __future__ import annotations
import logging
from src.llm.llm_client import LLMClient
from src.scriptplan.json_utils import close_truncated_json, parse_llm_json
from src.scriptplan.models import (
    AssetType,
    MotionType,
    ScriptSegment,
    SegmentPurpose,
    VideoIdea,
    VideoScript,
    VisualBible,
    VoiceParams,
)

log = logging.getLogger("scriptplan")


class ScriptPlanError(RuntimeError):
    """脚本规划失败（LLM 输出不可解析或 segments 为空）。"""


# 用途 → 默认语音参数映射
_PURPOSE_VOICE_DEFAULTS: dict[str, dict] = {
    "hook": {"speed": "+5%", "emotion": "urgent", "pause_after": 0.3},
    "setup": {"speed": "+0%", "emotion": "neutral", "pause_after": 0.2},
    "develop": {"speed": "+0%", "emotion": "narrative", "pause_after": 0.2},
    "twist": {"speed": "-5%", "emotion": "dramatic", "pause_before": 0.5, "pause_after": 0.5},
    "climax": {"speed": "+5%", "emotion": "intense", "pause_after": 0.3},
    "ending": {"speed": "-5%", "emotion": "reflective", "pause_after": 1.0},
}

# 用途 → 默认镜头运动映射
_PURPOSE_MOTION_DEFAULTS: dict[str, str] = {
    "hook": "push_in",
    "setup": "static",
    "develop": "pan",
    "twist": "zoom",
    "climax": "push_in",
    "ending": "static",
}


class ScriptPlanner:
    """将视频方案转换为结构化脚本（逐段旁白+画面+时长）。"""

    @staticmethod
    def _estimate_max_tokens(idea: VideoIdea) -> int:
        """按分段数估算输出 token，避免 60s 长脚本被截断。"""
        return max(4096, idea.segment_count * 520 + 1200)

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, idea: VideoIdea, inspiration: str) -> VideoScript:
        """生成结构化视频脚本。

        Args:
            idea: 视频方案
            inspiration: 原始灵感（作为创作素材）

        Returns:
            VideoScript 完整视频脚本
        """
        system_prompt = (
            "你是一位专业的短视频编剧。你的任务是将视频方案落地为逐段脚本。\n\n"
            "【每段必须包含】\n"
            "1. purpose: 段落用途 (hook/setup/develop/twist/climax/ending)\n"
            "2. voiceover: 旁白文本（简短有力，口语化，每段15-30字）\n"
            "3. visual: 画面描述（具体可执行，描述画面内容而非抽象概念）\n"
            "4. duration_sec: 时长（2-6秒，hook段2-3秒，twist段4-5秒）\n\n"
            "【角色描写规则 - 极其重要！】\n"
            "visual 中出现的每个角色都必须明确标注：\n"
            "- 性别：用「男/男性/男人」或「女/女性/女人」明确标注，绝不能省略\n"
            "- 外观：年龄段、发型、服装、体型等关键特征\n"
            "- 动作：具体在做什么\n"
            "示例：\n"
            '  ✓ "一个穿着黑色西装的中年男人站在落地窗前，双手背后，俯瞰城市夜景"\n'
            '  ✓ "一个长发的年轻女人蜷缩在沙发上，怀里抱着橘猫，手机屏幕照亮她的脸"\n'
            '  ✗ "一个人站在窗前看着远方"（没标性别和外观）\n'
            '  ✗ "TA在沙发上"（性别不明）\n'
            "如果多段有同一角色，该角色的外观描述必须在每段保持一致！\n\n"
            "【短视频编剧法则】\n"
            "1. 一段只表达一个信息点，一个主画面\n"
            "2. 旁白要能和画面同步，不能旁白说A画面是B\n"
            "3. hook段必须制造悬念或冲突，禁止平铺直叙\n"
            "4. 每段旁白必须简短，一句话说完一个信息\n"
            "5. visual必须是具体画面（人物/场景/物体），不能是抽象概念\n"
            "6. 总时长要接近目标时长\n"
            "7. ending段要有互动感，引导观众评论\n\n"
            "【视觉圣经 - visual_bible】\n"
            "你还必须输出一个 visual_bible 对象，用于保证全片画面风格和角色外观的一致性：\n"
            "- style_tags: 全片风格关键词（英文），如 'cinematic, dark moody, neon-lit, urban'\n"
            "- scene_anchor: 全片固定场景/布景锚点（英文），描述同一空间、光线、关键道具位置；"
            "全片所有 segment 必须发生在该场景内，例如 "
            "'modest apartment entryway, beige doormat, single cardboard box on floor, warm window side light'\n"
            "- negative_prompt: 全片负面提示词（英文），如 'blurry, deformed, extra limbs, text, watermark'\n"
            "- characters: 出场角色列表，每个角色包含 name（中文名）、prompt_anchor（英文外观锚点）、可选 aliases（别名如 橘猫/猫），例如：\n"
            '  {"name": "大橘", "prompt_anchor": "a chubby orange tabby cat...", "aliases": ["橘猫", "猫"]}\n'
            '  {"name": "张伟", "prompt_anchor": "a 28 year old Chinese man, short black hair, lean build, gray hoodie"}\n'
            "角色的 prompt_anchor 一旦定义，全片所有 segment 中该角色的描写必须与此一致。\n\n"
            "请返回严格的 JSON 格式：\n"
            "{\n"
            '  "title": "视频标题（10字以内，有吸引力）",\n'
            '  "theme": "一句话主题",\n'
            '  "hook": "前3秒钩子的核心文案",\n'
            '  "visual_bible": {\n'
            '    "style_tags": "cinematic, dark moody, ...",\n'
            '    "scene_anchor": "same room layout, lighting, key props...",\n'
            '    "negative_prompt": "blurry, deformed, ...",\n'
            '    "characters": [\n'
            '      {"name": "角色名", "prompt_anchor": "英文外观锚点"}\n'
            "    ]\n"
            "  },\n"
            '  "segments": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "purpose": "hook",\n'
            '      "voiceover": "旁白文本",\n'
            '      "visual": "画面描述",\n'
            '      "duration_sec": 3.0\n'
            "    }\n"
            "  ],\n"
            '  "ending_hook": "结尾互动文案"\n'
            "}"
        )

        user_prompt = (
            f"【视频方案】\n"
            f"类型：{idea.video_type}\n"
            f"目标时长：{idea.target_duration}秒\n"
            f"分段数：{idea.segment_count}段\n"
            f"节奏：{idea.rhythm}\n"
            f"反转类型：{idea.twist_type}\n"
            f"结尾方式：{idea.ending_type}\n"
            f"调性：{idea.tone}\n\n"
            f"【创作素材/灵感】\n{inspiration}\n\n"
            f"请生成 {idea.segment_count} 段脚本，总时长约 {idea.target_duration} 秒。"
        )

        max_tokens = self._estimate_max_tokens(idea)
        data = self._request_script(system_prompt, user_prompt, max_tokens)
        if not data.get("segments"):
            retry_prompt = (
                f"{user_prompt}\n\n"
                "【重要】上次 JSON 输出不完整或未闭合。"
                f"请输出完整、可解析的 JSON，segments 必须恰好 {idea.segment_count} 段。"
                "visual_bible.characters 最多 2 个，字段尽量简短，但 segments 必须完整。"
            )
            log.warning("ScriptPlanner 首次解析无 segments，正在重试（max_tokens=%d）", max_tokens)
            data = self._request_script(
                system_prompt,
                retry_prompt,
                max(max_tokens, 6144),
            )

        if not data.get("segments"):
            raise ScriptPlanError(
                "脚本生成失败：LLM 返回的 JSON 不完整或 segments 为空。"
                "可尝试缩短目标时长（-d 45）或更换 LLM 模型后重试。"
            )

        # 构建 VideoScript
        segments = []
        for seg_data in data.get("segments", []):
            purpose_str = seg_data.get("purpose", "develop")
            try:
                purpose = SegmentPurpose(purpose_str)
            except ValueError:
                purpose = SegmentPurpose.DEVELOP

            # 默认语音参数
            voice_defaults = _PURPOSE_VOICE_DEFAULTS.get(purpose.value, {})
            voice_params = VoiceParams(**voice_defaults)

            # 默认镜头运动
            motion_str = seg_data.get(
                "motion",
                _PURPOSE_MOTION_DEFAULTS.get(purpose.value, "static"),
            )
            try:
                motion = MotionType(motion_str)
            except ValueError:
                motion = MotionType.STATIC

            segments.append(ScriptSegment(
                id=seg_data.get("id", len(segments) + 1),
                purpose=purpose,
                voiceover=seg_data.get("voiceover", ""),
                visual=seg_data.get("visual", ""),
                motion=motion,
                duration_sec=float(seg_data.get("duration_sec", 3.0)),
                voice_params=voice_params,
            ))

        # 解析 visual_bible
        visual_bible = None
        vb_data = data.get("visual_bible")
        if vb_data and isinstance(vb_data, dict):
            visual_bible = VisualBible(
                style_tags=vb_data.get("style_tags", ""),
                scene_anchor=vb_data.get("scene_anchor", ""),
                negative_prompt=vb_data.get("negative_prompt", ""),
                characters=vb_data.get("characters", []),
            )
            log.info(
                "视觉圣经: style=%s, scene=%s, characters=%d",
                visual_bible.style_tags[:50],
                (visual_bible.scene_anchor[:50] if visual_bible.scene_anchor else "(无)"),
                len(visual_bible.characters),
            )

        script = VideoScript(
            title=data.get("title", "未命名视频"),
            theme=data.get("theme", ""),
            hook=data.get("hook", segments[0].voiceover if segments else ""),
            tone=idea.tone,
            segments=segments,
            ending_hook=data.get("ending_hook", ""),
            visual_bible=visual_bible,
            idea=idea,
        )
        script.compute_duration()

        return script

    def _request_script(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> dict:
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            json_mode=True,
            max_tokens=max_tokens,
        )
        if response.finish_reason == "length":
            log.warning(
                "ScriptPlanner LLM 输出触达 max_tokens=%d 被截断", max_tokens
            )
        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> dict:
        """健壮的 JSON 解析"""
        data = parse_llm_json(content)
        if data is not None:
            return data

        data = close_truncated_json(content)
        if data is not None:
            log.warning("ScriptPlanner 通过补全括号解析截断 JSON")
            return data

        log.error("ScriptPlanner 无法解析响应: %s", content[:300])
        return {"title": "解析失败", "segments": [], "ending_hook": ""}
