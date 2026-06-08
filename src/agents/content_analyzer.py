"""内容分析 Agent - 分析小说类型、角色、风格"""
from __future__ import annotations

import re

from src.agents.state import AgentState, Decision
from src.agents.utils import make_decision, extract_json_obj, extract_json_array
from src.tools.segment_tool import SegmentTool
from src.logger import log


# 风格映射
STYLE_MAP = {
    ("武侠", "古代"): "chinese_ink",
    ("玄幻", "架空"): "anime",
    ("都市", "现代"): "realistic",
    ("科幻", "未来"): "cyberpunk",
    ("言情", "现代"): "watercolor",
}

# 规则分类
GENRE_RULES = [
    (r"修炼|法宝|灵气|宗门|渡劫|仙|丹药", "玄幻", "架空"),
    (r"江湖|剑气|武功|内力|掌门|侠客", "武侠", "古代"),
    (r"公司|手机|互联网|地铁|外卖|办公室", "都市", "现代"),
    (r"星际|宇宙飞船|机器人|AI|赛博", "科幻", "未来"),
    (r"爱情|恋爱|男朋友|女朋友|心动|甜蜜", "言情", "现代"),
    (r"皇上|朕|太后|将军|丫鬟|府邸", "历史", "古代"),
]

_CHARACTER_EXTRACT_PROMPT = """\
你是小说角色视觉设定专家。分析以下文本，提取主要角色（最多5个）。

对每个角色，desc 必须是一段完整的中文外观描述（80-150字），必须明确包含：
1. 年龄（如"约35岁"；文本未写明时根据身份合理推断）
2. 性别（男/女；根据称呼、代词、身份推断，不能含糊）
3. 身材体型（如高挑纤细、魁梧壮实、微胖敦实等）
4. 样貌特征（脸型、眉眼、表情气质、肤色等，可合理补充）
5. 发型发色
6. 服装穿着（符合时代与场景）
7. 1-2个标志性细节（如伤疤、眼镜、职业配饰等；无则按身份合理假设）

规则：
- 文本已写明的特征必须保留，不可与原文矛盾
- 文本未写明的部分可合理假设，假设需符合角色身份、时代背景与故事氛围
- desc 用连贯中文描述，不要用 JSON 或分点列表
- 只提取在文中有实际戏份的角色

文本：
{text}

输出 JSON 数组：[{{"name": "姓名", "desc": "完整外观描述"}}]"""

_CHARACTER_ENRICH_PROMPT = """\
你是小说角色视觉设定专家。以下为已从文本识别出的角色名，请结合文本为每个角色撰写详细外观描述。

每个 desc 必须是一段完整中文（80-150字），明确包含：年龄、性别、身材体型、样貌特征、发型发色、服装穿着、标志性细节。
文本未写明的可合理假设，但不可与原文矛盾。desc 用连贯中文描述，不要用分点列表。

文本：
{text}

角色：{names}

输出 JSON 数组：[{{"name": "姓名", "desc": "完整外观描述"}}]，姓名必须与输入一致。"""


class ContentAnalyzerAgent:
    def __init__(self, config: dict, budget_mode: bool = False):
        self.config = config
        self.budget_mode = budget_mode
        self._llm = None

    def _get_llm(self):
        """懒加载 LLM"""
        if self._llm is None:
            from src.llm.llm_client import create_llm_client

            llm_cfg = dict(self.config.get("llm", {}))
            self._llm = create_llm_client(llm_cfg)
        return self._llm

    def classify_genre(self, text: str) -> dict:
        if self.budget_mode:
            return self._classify_by_rules(text)
        return self._classify_by_llm(text)

    def _classify_by_rules(self, text: str) -> dict:
        sample = text[:2000]
        for pattern, genre, era in GENRE_RULES:
            if re.search(pattern, sample):
                return {"genre": genre, "era": era, "confidence": 0.8}
        return {"genre": "其他", "era": "现代", "confidence": 0.5}

    def _classify_by_llm(self, text: str) -> dict:
        sample = text[:1000]
        prompt = (
            "你是小说类型分析专家。分析以下文本，判断类型和时代背景。\n"
            "可选类型：武侠、玄幻、都市、言情、科幻、悬疑、历史、其他\n"
            "可选时代：古代、现代、未来、架空\n\n"
            f"文本：\n{sample}\n\n"
            '输出 JSON：{{"genre": "类型", "era": "时代", "confidence": 0.0-1.0}}'
        )
        try:
            result = self._get_llm().chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
            )
            data = extract_json_obj(result.content)
            if data and "genre" in data:
                return data
        except Exception as e:
            log.warning("LLM 分类失败 (%s)，回退到规则", e)
        return self._classify_by_rules(text)

    def extract_characters(self, text: str) -> list[dict]:
        if self.budget_mode:
            return self._extract_characters_by_rules(text)
        return self._extract_characters_by_llm(text)

    @staticmethod
    def _normalize_characters(data: list | None) -> list[dict]:
        """清洗 LLM 返回的角色列表，保留 name/desc 字段。"""
        if not data:
            return []
        normalized: list[dict] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            desc = str(entry.get("desc", "")).strip()
            if not name:
                continue
            normalized.append({"name": name, "desc": desc})
        return normalized[:5]

    def _enrich_character_descriptions(
        self, text: str, characters: list[dict]
    ) -> list[dict]:
        """为仅有名字、缺少 desc 的角色补全详细外观描述。"""
        if not characters:
            return characters

        names = [c["name"] for c in characters if c.get("name")]
        if not names:
            return characters

        prompt = _CHARACTER_ENRICH_PROMPT.format(
            text=text[:3000],
            names="、".join(names),
        )
        try:
            result = self._get_llm().chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
            )
            enriched = self._normalize_characters(extract_json_array(result.content))
            if not enriched:
                return characters

            desc_by_name = {c["name"]: c["desc"] for c in enriched if c.get("desc")}
            merged: list[dict] = []
            for char in characters:
                name = char["name"]
                desc = desc_by_name.get(name) or char.get("desc", "")
                merged.append({"name": name, "desc": desc})
            return merged
        except Exception as e:
            log.warning("LLM 角色描述补全失败 (%s)，保留已有结果", e)
            return characters

    def _extract_characters_by_rules(self, text: str) -> list[dict]:
        # 使用非贪婪匹配，确保提取人名而非"人名+动词"
        matches = re.findall(
            r"([\u4e00-\u9fa5]{2,4}?)(?:说道|问道|笑道|喊道|道|说)", text[:3000]
        )
        names = list(dict.fromkeys(matches))[:5]
        characters = [{"name": n, "desc": ""} for n in names]
        return self._enrich_character_descriptions(text, characters)

    def _extract_characters_by_llm(self, text: str) -> list[dict]:
        prompt = _CHARACTER_EXTRACT_PROMPT.format(text=text[:3000])
        try:
            result = self._get_llm().chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
            )
            data = self._normalize_characters(extract_json_array(result.content))
            if data:
                return data
        except Exception as e:
            log.warning("LLM 角色提取失败 (%s)，回退到规则", e)
        return self._extract_characters_by_rules(text)

    def suggest_style(self, genre: str, era: str) -> str:
        return STYLE_MAP.get((genre, era), "anime")


def _log_character_descriptions(characters: list[dict], source: str = "ContentAnalyzer") -> None:
    """将角色外观描述打印到日志，便于核对一致性预填内容。"""
    for entry in characters:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        desc = str(entry.get("desc", "")).strip()
        if not name or not desc:
            continue
        log.info("[%s] 角色 %s: %s", source, name, desc)


def content_analyzer_node(state: AgentState) -> dict:
    """ContentAnalyzer 节点"""
    config = state["config"]
    budget_mode = state.get("budget_mode", False)
    agent = ContentAnalyzerAgent(config, budget_mode)
    decisions: list[Decision] = []

    # 1. 分段
    seg_tool = SegmentTool(config)
    segments = seg_tool.run(state["full_text"])

    decisions.append(make_decision(
        "ContentAnalyzer", "segment",
        f"分段完成：{len(segments)} 段",
        f"方法={config.get('segmenter', {}).get('method', 'simple')}",
    ))

    # 2. 类型分析
    genre_info = agent.classify_genre(state["full_text"])

    decisions.append(make_decision(
        "ContentAnalyzer", "classify",
        f"类型={genre_info['genre']}, 时代={genre_info.get('era', '未知')}",
        f"置信度={genre_info.get('confidence', 0)}",
    ))

    # 3. 角色提取
    characters = agent.extract_characters(state["full_text"])
    _log_character_descriptions(characters)
    decisions.append(make_decision(
        "ContentAnalyzer", "extract_characters",
        f"提取 {len(characters)} 个角色",
        f"角色: {[c['name'] for c in characters]}",
    ))

    # 4. 风格推荐
    style = agent.suggest_style(genre_info["genre"], genre_info.get("era", "现代"))
    decisions.append(make_decision(
        "ContentAnalyzer", "suggest_style",
        f"推荐风格={style}",
        f"基于类型={genre_info['genre']}, 时代={genre_info.get('era')}",
    ))

    log.info(
        "[ContentAnalyzer] %s/%s风格, %d段, %d角色",
        genre_info["genre"],
        style,
        len(segments),
        len(characters),
    )

    return {
        "segments": segments,
        "genre": genre_info["genre"],
        "era": genre_info.get("era"),
        "characters": characters,
        "suggested_style": style,
        "decisions": decisions,
    }
