"""内容分析 Agent - 分析小说类型、角色、风格"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from src.agents.state import AgentState, Decision
from src.agents.utils import make_decision, extract_json_obj, extract_json_array
from src.tools.segment_tool import SegmentTool
from src.logger import log
from src.promptgen.era_context import (
    CLASSICAL,
    CLASSICAL_CHARACTER_ADDENDUM,
    default_hairstyle_for_era,
    era_display_name,
    normalize_era,
    sanitize_classical_desc,
)


# 风格映射
STYLE_MAP = {
    ("武侠", "古代"): "chinese_ink",
    ("玄幻", "架空"): "anime",
    ("都市", "现代"): "anime",
    ("科幻", "未来"): "cyberpunk",
    ("言情", "现代"): "watercolor",
    ("言情", "古代"): "chinese_ink",
    ("历史", "古代"): "chinese_ink",
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
你是小说角色视觉设定专家。分析以下文本，提取主要角色（最多{max_chars}个）。

对每个角色，desc 必须是一段完整的中文外观描述（80-150字），必须明确包含：
1. 年龄（如"约35岁"；文本未写明时根据身份合理推断）
2. 性别（男/女；根据称呼、代词、身份推断，不能含糊）
3. 身材体型（如高挑纤细、魁梧壮实、微胖敦实等）
4. 样貌特征（脸型、眉眼、表情气质、肤色等，可合理补充）
5. 发型发色（必填且须写死：具体款式 + 发色，如「黑色利落短发」「乌黑长发微卷披肩」；
   不可只写「头发整齐」等模糊表述，不可省略发型）
6. 服装穿着（符合时代与场景）
7. 1-2个标志性细节（如伤疤、眼镜、职业配饰等；无则按身份合理假设）

规则：
- name 必须使用文中出现的真实姓名（如王璟、张厉、蔡明微），禁止用「我」「前桌」「男监考老师」等泛称
- 叙述者若在文末揭示真名（如「我叫王璟」），name 用真名而非「我」
- 职能称呼若后文给出姓名（如女监考老师蔡明微），name 用蔡明微
- 优先保留全文高频具名主角与核心配角；开篇即死且未留名的角色（如前桌）不要占名额
- 文本已写明的特征必须保留，不可与原文矛盾
- 文本未写明的部分可合理假设，假设需符合角色身份、时代背景与故事氛围
- 每个角色的发型一旦确定，desc 中必须写明，供后续分镜全程保持一致
- desc 用连贯中文描述，不要用 JSON 或分点列表
- 只提取在文中有实际戏份的角色

文本：
{text}

输出 JSON 数组：[{{"name": "姓名", "desc": "完整外观描述"}}]"""

_DIALOGUE_NAME_RE = re.compile(r'[「"]([\u4e00-\u9fa5]{2,4})[，,]')

MAX_CHARACTERS = 8
_CHARACTER_SAMPLE_MAX_LEN = 8000

# 角色发现时需排除的非人名词语
_EXCLUDED_CHARACTER_NAMES = frozenset({
    "这个", "那个", "什么", "怎么", "一个", "自己", "他们", "她们", "我们", "你们",
    "大家", "所有", "这些", "那些", "已经", "现在", "然后", "但是", "因为", "所以",
    "如果", "虽然", "不过", "而且", "只是", "可是", "还是", "就是", "终于", "突然",
    "忽然", "于是", "不禁", "居然", "竟然", "原来", "果然", "毕竟", "一直", "一起",
    "一样", "不要", "不能", "可以", "应该", "必须", "连忙", "赶紧", "急忙", "立刻",
    "马上", "这里", "那里", "哪里", "时候", "地方", "东西", "事情", "今天", "明天",
    "昨天", "知道", "以为", "觉得", "认为", "希望", "害怕", "担心", "高兴", "难过",
    "生气", "哥哥", "姐姐", "弟弟", "妹妹", "爸爸", "妈妈", "朋友", "兄弟", "姑娘",
    "少年", "老人", "孩子", "女子", "男子", "少女", "老者", "此人", "那人", "众人",
    "警察", "宿舍", "学校", "大火", "火灾", "现场", "男生", "女生", "同学", "教授",
    "心理学", "女朋友", "男朋友", "男同学", "女同学", "大学生", "研究生", "嫌疑人",
    "凶手", "死者", "尸体", "户体", "墙壁", "银幕", "审问", "酒吧", "警车", "警局",
    "是的", "不是", "真的", "真话", "继续", "跟我", "脸上", "惊讶", "知道", "想想",
    "我女朋", "一下", "半点", "半开", "半晌", "不由", "不由暗", "不由得",
    # 考场/场景高频碎片（非人名）
    "老师", "考场", "考试", "考证", "准考证", "考老师", "监考老", "的准考", "张厉的",
    "纸条", "照片", "分钟", "厕所", "教室", "考生", "档案", "规则", "痛苦", "绝望",
    "起来", "为什么", "保洁员", "洁员", "脑海", "眼睛", "身上", "手机", "熟悉",
    "诡异", "声音", "规则", "试卷", "身体", "问题", "门口", "怪物", "一句话", "表白",
    "办公室", "兴奋", "到了", "资料", "呼吸", "脑袋", "这时", "进去", "阴森森",
    "欺负", "十分钟", "二十分", "明微", "一眼", "很快", "起来", "为什么", "蔡老师",
    "马甲", "校服", "监考证",
})

# 常见姓氏（用于过滤「欺负」「十分钟」等非人名高频词）
_SURNAME_CHARS = frozenset(
    "王李张刘陈杨黄赵周吴徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾肖田董潘袁蒋蔡"
    "余于杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金廖贾夏韦付方邹熊孟秦白江闫薛侯雷龙段郝"
    "孔邵史毛万钱汤尹黎易常武乔贺赖龚文严华陶舒顾孟"
)

# LLM 常输出的泛称角色，满员时可被具名角色替换（值越小越优先被替换）
_ROLE_SLOT_PRIORITY: dict[str, int] = {
    "前桌": 0,
    "男监考老师": 1,
    "女监考老师": 1,
    "骷髅头监考老师": 1,
    "保洁员": 2,
    "我": 3,
    "答主": 3,
    "主角": 3,
    "男主": 3,
    "校霸": 4,
}

_EXPLICIT_NAME_PATTERNS = (
    re.compile(r"名字叫[「「]([\u4e00-\u9fa5]{2,4})[」」]"),
    re.compile(r"名叫[「「]([\u4e00-\u9fa5]{2,4})[」」]"),
    re.compile(r"女生叫([\u4e00-\u9fa5]{2,4})"),
    re.compile(r"这老师叫[「「]([\u4e00-\u9fa5]{2,4})[」」]"),
    re.compile(r"老师叫[「「]([\u4e00-\u9fa5]{2,4})[」」]"),
)

_PROTAGONIST_NAME_TAIL_RE = re.compile(
    r"我叫([\u4e00-\u9fa5]{2,3})"
)

_NAME_FRAGMENT_BAD_RE = re.compile(r"[的了过着是在和与把被将让给到]|的准|是张|是我|了一|看向|过来")

_NAME_SPEECH_RE = re.compile(
    r"([\u4e00-\u9fa5]{2,3})(?:说道|问道|笑道|喊道)"
)
_NAME_SAY_RE = re.compile(
    r"([\u4e00-\u9fa5]{2,3})说[：:「]"
)
_NAME_CONTEXT_RES = [
    re.compile(r"在([\u4e00-\u9fa5]{2,3})看"),
    re.compile(r"问([\u4e00-\u9fa5]{2,3})[：:「]"),
    re.compile(r"以及([\u4e00-\u9fa5]{2,3})"),
    re.compile(r"跟([\u4e00-\u9fa5]{2,3})会面"),
]
_MIN_DISCOVER_MENTIONS = 2

_HAIRSTYLE_IN_DESC_RE = re.compile(
    r"短发|长发|卷发|微卷|马尾|盘发|束发|齐肩|披肩|双马尾|丸子头|发髻|"
    r"寸头|平头|碎发|直发|波浪|刘海|中分|侧分|黑发|乌发|棕发|金发|白发|银发"
)

_CHARACTER_ENRICH_PROMPT = """\
你是小说角色视觉设定专家。以下为已从文本识别出的角色名，请结合文本为每个角色撰写详细外观描述。

每个 desc 必须是一段完整中文（80-150字），明确包含：年龄、性别、身材体型、样貌特征、
发型发色（具体款式+发色，写死不可省略）、服装穿着、标志性细节。
文本未写明的可合理假设，但不可与原文矛盾；若原文只写「短发/长发/微卷」等，必须原样保留并补全发色。
desc 用连贯中文描述，不要用分点列表。

文本：
{text}

角色：{names}

输出 JSON 数组：[{{"name": "姓名", "desc": "完整外观描述"}}]，姓名必须与输入一致。"""

_POV_NARRATOR_RESOLVE_PROMPT = """\
你是叙事分析专家。判断以下文本的第一人称叙述者「我/我们」对应哪个具名角色。

规则：
- 若全文并非以第一人称叙述为主，pov_narrator 为 null
- 若「我」无法可靠对应到已知具名角色，pov_narrator 为 null
- 优先选择叙述者本人，不要把「我」误指为被观察对象
- 只输出 JSON：{{"pov_narrator": "姓名或null", "reason": "一句话依据"}}

已知角色：{names}

文本：
{text}"""

_INTRO_VARIANTS_PROMPT = """\
你是短视频文案专家。根据以下小说文本，写出 3 条约 20 字的中文故事介绍，用于片头/封面引流。

要求：
- 每条 18-22 个汉字（不含标点），3 条风格要有差异：悬念钩子 / 氛围感 / 主题回扣
- 不要剧透结局和核心反转
- 抓核心冲突，语言口语化、有吸引力

文本：
{text}

输出 JSON：{{"intro_variants": ["介绍1", "介绍2", "介绍3"]}}"""

_INTRO_SAMPLE_MAX_LEN = 6000


class ContentAnalyzerAgent:
    def __init__(self, config: dict, budget_mode: bool = False):
        self.config = config
        self.budget_mode = budget_mode
        self._llm = None
        self._last_character_alias_map: dict[str, str] = {}
        self._last_applied_aliases: list[tuple[str, str]] = []
        self._last_character_review_log: list[str] = []
        self._forced_era: str | None = None

    def set_era(self, era: str | None) -> None:
        self._forced_era = normalize_era(era)

    @property
    def forced_era(self) -> str | None:
        return self._forced_era

    @staticmethod
    def _desc_has_hairstyle(desc: str) -> bool:
        return bool(_HAIRSTYLE_IN_DESC_RE.search(desc))

    @staticmethod
    def _default_hairstyle_for_desc(desc: str, *, era_key: str | None = None) -> str:
        if era_key:
            return default_hairstyle_for_era(desc, era_key)
        if re.search(r"女", desc):
            return "黑色齐肩直发"
        return "黑色短发"

    @staticmethod
    def _insert_hairstyle_into_desc(desc: str, hairstyle: str) -> str:
        """在 desc 中补入缺失的固定发型描述。"""
        if not desc or not hairstyle.strip():
            return desc
        if ContentAnalyzerAgent._desc_has_hairstyle(desc):
            return desc
        clause = hairstyle.strip()
        age_match = re.search(r"(约?\d+岁[^，,。]*[，,])", desc)
        if age_match:
            pos = age_match.end()
            return f"{desc[:pos]}{clause}，{desc[pos:]}"
        return f"{clause}，{desc}"

    @staticmethod
    def _hairstyle_from_text_fragment(fragment: str, *, color: str = "黑色") -> str | None:
        """将原文外貌片段转为固定发型描述。"""
        if not fragment:
            return None
        if "短发" in fragment:
            return f"{color}短发"
        if "长发微卷" in fragment or "微卷" in fragment:
            return f"{color}长发微卷"
        if "长发" in fragment:
            return f"{color}长发"
        if "近视" in fragment or "相貌平平" in fragment:
            return f"{color}齐肩直发，简单扎成低马尾"
        return None

    @classmethod
    def _infer_hairstyle_from_text(cls, text: str, name: str) -> str | None:
        """从原文片段推断角色固定发型（保留文本明示，不做与原文矛盾的假设）。"""
        if not text or not name:
            return None
        escaped = re.escape(name)

        positional_patterns = (
            rf"左边的女生(.+?)，(?:我)?猜她应该是{escaped}",
            rf"中间的女生(.+?)，是{escaped}",
            rf"右边的女生则(.+?)。{{0,60}}?(?:我想她应该就是|她应该就是)?{escaped}",
        )
        for pattern in positional_patterns:
            match = re.search(pattern, text)
            if match:
                inferred = cls._hairstyle_from_text_fragment(match.group(1))
                if inferred:
                    return inferred

        near_name = re.search(
            rf"{escaped}(?:留着)?[^。！？，,]{{0,8}}"
            rf"(短发|长发微卷|长发|微卷|齐肩发|马尾|盘发)",
            text,
        )
        if near_name:
            token = near_name.group(1)
            if token == "微卷":
                return "黑色长发微卷"
            return f"黑色{token}"

        kept_hair = re.search(
            rf"留着(短发|长发)[^。！？]{{0,60}}?"
            rf"(?:{escaped}|我想她应该就是{escaped})",
            text,
        )
        if kept_hair:
            return f"黑色{kept_hair.group(1)}"

        return None

    @classmethod
    def _finalize_character_descriptions(
        cls,
        characters: list[dict],
        full_text: str,
        *,
        era_key: str | None = None,
    ) -> list[dict]:
        """确保每个角色 desc 都包含可锁定的具体发型。"""
        finalized: list[dict] = []
        for char in characters:
            name = str(char.get("name", "")).strip()
            desc = str(char.get("desc", "")).strip()
            if not name:
                continue
            if era_key == CLASSICAL:
                desc = sanitize_classical_desc(desc)
            if desc and not cls._desc_has_hairstyle(desc):
                hint = cls._infer_hairstyle_from_text(full_text, name)
                desc = cls._insert_hairstyle_into_desc(
                    desc,
                    hint or cls._default_hairstyle_for_desc(desc, era_key=era_key),
                )
            if era_key == CLASSICAL:
                desc = sanitize_classical_desc(desc)
            finalized.append({"name": name, "desc": desc})
        return finalized

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
        sample = self._sample_text_for_character_extraction(text)
        if self.budget_mode:
            characters = self._extract_characters_by_rules(text, sample=sample)
        else:
            characters = self._extract_characters_by_llm(text, sample=sample)
        characters = self._supplement_discovered_characters(characters, text, sample=sample)
        characters = self._reconcile_with_dialogue_names(characters, text)
        characters, applied_aliases = self._reconcile_character_aliases(characters, text)
        self._last_character_alias_map = self._build_character_alias_map(text)
        self._last_applied_aliases = applied_aliases
        _log_character_aliases(self._last_character_alias_map, applied_aliases)
        characters = self._review_characters_with_discussion(text, characters, sample=sample)
        return self._finalize_character_descriptions(
            characters, text, era_key=self._forced_era
        )

    def _review_characters_with_discussion(
        self,
        text: str,
        characters: list[dict],
        *,
        sample: str | None = None,
    ) -> list[dict]:
        """双 AI 讨论审核角色列表（budget 模式或禁用时跳过）。"""
        from src.agents.character_reviewer import (
            character_review_enabled,
            create_reviewer_llm,
            run_character_review_discussion,
        )

        self._last_character_review_log = []
        if not characters or not character_review_enabled(self.config, self.budget_mode):
            return characters

        try:
            reviewer_llm, reviewer_provider, same_source = create_reviewer_llm(self.config)
        except Exception as exc:
            log.warning("[ContentAnalyzer] 角色审核 LLM 不可用 (%s)，跳过讨论", exc)
            return characters

        era_addendum = ""
        if self._forced_era == CLASSICAL:
            era_addendum = CLASSICAL_CHARACTER_ADDENDUM

        try:
            result = run_character_review_discussion(
                text,
                characters,
                primary_llm=self._get_llm(),
                reviewer_llm=reviewer_llm,
                reviewer_provider=reviewer_provider,
                same_source=same_source,
                era_addendum=era_addendum,
                max_chars=MAX_CHARACTERS,
            )
        except Exception as exc:
            log.warning("[ContentAnalyzer] 角色讨论审核失败 (%s)，保留原列表", exc)
            return characters

        self._last_character_review_log = result.discussion
        if result.discussion:
            log.info(
                "[ContentAnalyzer] 角色双 AI 讨论完成 (审核=%s, %d 人)",
                reviewer_provider or "?",
                len(result.characters),
            )
        return result.characters or characters

    @staticmethod
    def _character_name_set(characters: list[dict]) -> set[str]:
        return {
            str(c.get("name", "")).strip()
            for c in characters
            if str(c.get("name", "")).strip()
        }

    def resolve_pov_narrator(
        self,
        text: str,
        characters: list[dict],
        *,
        alias_map: dict[str, str] | None = None,
    ) -> str | None:
        """解析第一人称叙述者「我」对应的具名角色，供 POV 分镜锁定。"""
        from src.promptgen.author_pov import detect_first_person_work

        if not text:
            return None

        names = self._character_name_set(characters)
        mapping = alias_map if alias_map is not None else self._last_character_alias_map
        if not mapping:
            mapping = self._build_character_alias_map(text)

        candidate = (
            mapping.get("我")
            or mapping.get("答主")
            or self._extract_protagonist_name_from_text(text)
        )
        is_first_person = detect_first_person_work(text)

        # 别名/「我叫XX」高置信时直接返回（即使全文较短未触发 first_person 检测）
        if candidate and (not names or candidate in names):
            return candidate

        if not is_first_person:
            return None

        if not self.budget_mode:
            llm_name = self._resolve_pov_narrator_by_llm(text, names)
            if llm_name:
                return llm_name

        if candidate:
            log.warning(
                "[ContentAnalyzer] 叙述者 %s 不在角色表，仍设为 pov_narrator",
                candidate,
            )
            return candidate
        return None

    def _resolve_pov_narrator_by_llm(
        self, text: str, names: set[str]
    ) -> str | None:
        if not names:
            return None
        sample = self._sample_text_for_character_extraction(text)
        prompt = _POV_NARRATOR_RESOLVE_PROMPT.format(
            names="、".join(sorted(names)),
            text=sample,
        )
        try:
            result = self._get_llm().chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
            )
            data = extract_json_obj(result.content)
            if not data:
                return None
            raw = data.get("pov_narrator")
            if raw is None or str(raw).strip().lower() in {"null", "none", ""}:
                return None
            name = str(raw).strip()
            if name in names:
                log.info(
                    "[ContentAnalyzer] LLM 推断叙述者: %s (%s)",
                    name,
                    str(data.get("reason", ""))[:80],
                )
                return name
        except Exception as exc:
            log.warning("[ContentAnalyzer] 叙述者 LLM 推断失败 (%s)", exc)
        return None

    @staticmethod
    def _sample_text_for_character_extraction(
        text: str, max_len: int = _CHARACTER_SAMPLE_MAX_LEN
    ) -> str:
        """拼接文首+文尾采样，避免后段出场角色（如陈佳）被截断遗漏。"""
        if len(text) <= max_len:
            return text
        head_len = max_len // 2
        tail_len = max_len - head_len
        return text[:head_len] + "\n…\n" + text[-tail_len:]

    @classmethod
    def _is_plausible_character_name(cls, name: str) -> bool:
        name = str(name).strip()
        if not name or len(name) < 2 or len(name) > 4:
            return False
        if name in _EXCLUDED_CHARACTER_NAMES:
            return False
        if name.endswith("同学") and name not in {"李同学"}:
            return False
        return True

    @classmethod
    def _is_substring_of_frequent_full_name(cls, text: str, name: str) -> bool:
        """过滤「明微」这类从「蔡明微」拆出的三字姓名后缀碎片。"""
        if len(name) != 2:
            return False
        if text.count(name) >= 15:
            return False
        for match in re.finditer(r"[\u4e00-\u9fa5]{3}", text):
            full = match.group()
            if full == name or not full.endswith(name):
                continue
            if full[0] in _SURNAME_CHARS and full[0] != name[0] and text.count(full) >= 3:
                return True
        return False

    @classmethod
    def _looks_like_person_name(cls, name: str, *, text: str = "") -> bool:
        """比 _is_plausible_character_name 更严：过滤「老师」「张厉的」等碎片。"""
        if not cls._is_plausible_character_name(name):
            return False
        if _NAME_FRAGMENT_BAD_RE.search(name):
            return False
        if name.endswith(("的", "了", "着", "过", "吗", "呢", "吧", "啊")):
            return False
        if re.search(r"分钟|秒钟|小时|点钟|高考|考场|考试", name):
            return False
        if text and cls._is_substring_of_frequent_full_name(text, name):
            return False
        if name[0] not in _SURNAME_CHARS:
            return False
        return True

    @classmethod
    def _is_high_confidence_name(
        cls, name: str, counts: Counter[str], text: str
    ) -> bool:
        """是否值得占角色槽（显式提取或高频具名）。"""
        if not cls._looks_like_person_name(name, text=text):
            return False
        if counts.get(name, 0) >= 8:
            return True
        return text.count(name) >= 5

    @classmethod
    def _character_slot_priority(cls, name: str) -> int:
        return _ROLE_SLOT_PRIORITY.get(name, 50)

    @classmethod
    def _extract_protagonist_name_from_text(cls, text: str) -> str | None:
        """从文末揭示句提取叙述者真名（如「我叫王璟」）。"""
        tail = text[-5000:] if len(text) > 5000 else text
        for match in reversed(_PROTAGONIST_NAME_TAIL_RE.findall(tail)):
            if cls._looks_like_person_name(match):
                return match
        return None

    @classmethod
    def _extract_explicit_names_from_text(cls, text: str) -> Counter[str]:
        """从「女生叫舒然」「老师叫蔡明微」等句式提取高置信人名。"""
        counts: Counter[str] = Counter()
        for pattern in _EXPLICIT_NAME_PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1).strip()
                if cls._looks_like_person_name(name, text=text):
                    counts[name] += 8
        protagonist = cls._extract_protagonist_name_from_text(text)
        if protagonist:
            counts[protagonist] += 10
        return counts

    @classmethod
    def _discover_character_names_from_full_text(cls, text: str) -> list[str]:
        """扫描全文，按尾部戏份加权排序，补全 LLM/规则漏掉的后段角色。"""
        counts: Counter[str] = Counter()
        counts.update(cls._extract_explicit_names_from_text(text))

        for pattern in (_NAME_SPEECH_RE, _NAME_SAY_RE):
            for match in pattern.finditer(text):
                name = match.group(1)
                if cls._looks_like_person_name(name, text=text):
                    counts[name] += 2
        for pattern in _NAME_CONTEXT_RES:
            for match in pattern.finditer(text):
                name = match.group(1)
                if cls._looks_like_person_name(name, text=text):
                    counts[name] += 1
        for name in _DIALOGUE_NAME_RE.findall(text):
            if cls._looks_like_person_name(name, text=text):
                counts[name] += 2

        for name in {m.group() for m in re.finditer(r"[\u4e00-\u9fa5]{2,3}", text)}:
            if not cls._looks_like_person_name(name, text=text):
                continue
            freq = text.count(name)
            if freq >= 5:
                counts[name] = max(counts.get(name, 0), freq)

        counts = Counter({
            name: freq for name, freq in counts.items()
            if freq >= _MIN_DISCOVER_MENTIONS
        })
        if not counts:
            return []

        tail_start = int(len(text) * 0.75)
        tail_text = text[tail_start:]

        def rank_key(name: str) -> tuple[int, int, int]:
            tail_count = tail_text.count(name)
            return (tail_count, counts[name], -text.find(name))

        return sorted(counts.keys(), key=rank_key, reverse=True)

    @classmethod
    def _merge_discovered_name(
        cls,
        merged: list[dict],
        existing: set[str],
        name: str,
        *,
        counts: Counter[str],
        full_text: str,
    ) -> None:
        """将发现的人名并入角色表；满员时替换低优先级泛称槽位。"""
        if name in existing:
            return
        if name == "李昌" and "李同学" in existing:
            return
        if not cls._is_high_confidence_name(name, counts, full_text):
            return

        if len(merged) < MAX_CHARACTERS:
            merged.append({"name": name, "desc": ""})
            existing.add(name)
            return

        replace_idx = min(
            range(len(merged)),
            key=lambda i: cls._character_slot_priority(merged[i]["name"]),
        )
        old_name = merged[replace_idx]["name"]
        if cls._character_slot_priority(old_name) >= cls._character_slot_priority(name):
            return
        merged[replace_idx] = {"name": name, "desc": ""}
        existing.discard(old_name)
        existing.add(name)

    def _supplement_discovered_characters(
        self,
        characters: list[dict],
        full_text: str,
        *,
        sample: str | None = None,
    ) -> list[dict]:
        """在 LLM/规则结果上补入全文扫描发现、且尚未收录的角色名。"""
        name_counts = self._extract_explicit_names_from_text(full_text)
        for pattern in (_NAME_SPEECH_RE, _NAME_SAY_RE):
            for match in pattern.finditer(full_text):
                n = match.group(1)
                if self._looks_like_person_name(n, text=full_text):
                    name_counts[n] += 2
        for n in _DIALOGUE_NAME_RE.findall(full_text):
            if self._looks_like_person_name(n, text=full_text):
                name_counts[n] += 2
        for n in {m.group() for m in re.finditer(r"[\u4e00-\u9fa5]{2,3}", full_text)}:
            if self._looks_like_person_name(n, text=full_text) and full_text.count(n) >= 5:
                name_counts[n] = max(name_counts.get(n, 0), full_text.count(n))

        discovered = self._discover_character_names_from_full_text(full_text)
        protagonist = self._extract_protagonist_name_from_text(full_text)
        if protagonist and protagonist not in discovered:
            discovered.insert(0, protagonist)

        existing = {c["name"] for c in characters if c.get("name")}
        merged = list(characters)

        for name in discovered:
            self._merge_discovered_name(
                merged, existing, name, counts=name_counts, full_text=full_text
            )

        if any(not str(c.get("desc", "")).strip() for c in merged):
            enrich_sample = sample or self._sample_text_for_character_extraction(full_text)
            merged = self._enrich_character_descriptions(enrich_sample, merged)
        return merged[:MAX_CHARACTERS]

    @staticmethod
    def _dialogue_addressed_name_counts(text: str) -> Counter[str]:
        """统计对话称呼中出现的人名，如「李昌，你别乱说」。"""
        return Counter(_DIALOGUE_NAME_RE.findall(text))

    @classmethod
    def _reconcile_with_dialogue_names(
        cls, characters: list[dict], text: str
    ) -> list[dict]:
        """用全文对话称呼校正 LLM 误识别的姓李主角名（如 李逸→李昌）。"""
        counts = cls._dialogue_addressed_name_counts(text)
        if not counts:
            return characters

        li_names = [n for n in counts if n.startswith("李") and len(n) == 2]
        if not li_names:
            return characters

        canonical = max(li_names, key=lambda n: counts[n])
        dialogue_names = set(counts)

        reconciled: list[dict] = []
        seen: set[str] = set()
        for char in characters:
            name = char["name"]
            # 仅校正 2 字「李X」误识别（如 李逸→李昌），保留「李同学」等称呼
            if (
                len(name) == 2
                and name.startswith("李")
                and name not in dialogue_names
            ):
                name = canonical
            if name in seen:
                continue
            seen.add(name)
            reconciled.append({**char, "name": name})
        return reconciled[:MAX_CHARACTERS]

    @classmethod
    def _build_character_alias_map(cls, text: str) -> dict[str, str]:
        """根据全文构建泛称→真名映射表。"""
        alias_to_canonical: dict[str, str] = {}

        protagonist = cls._extract_protagonist_name_from_text(text)
        if protagonist:
            for alias in ("我", "答主", "主角", "男主"):
                alias_to_canonical[alias] = protagonist

        if text.count("蔡明微") >= 2:
            for alias in ("女监考老师", "蔡老师"):
                alias_to_canonical[alias] = "蔡明微"

        return alias_to_canonical

    @classmethod
    def _reconcile_character_aliases(
        cls, characters: list[dict], text: str
    ) -> tuple[list[dict], list[tuple[str, str]]]:
        """泛称/职能名合并为文中真名（我→王璟，女监考老师→蔡明微）。"""
        alias_to_canonical = cls._build_character_alias_map(text)
        input_names = {
            str(char.get("name", "")).strip()
            for char in characters
            if str(char.get("name", "")).strip()
        }
        applied_aliases = [
            (alias, canonical)
            for alias, canonical in sorted(alias_to_canonical.items())
            if alias in input_names and alias != canonical
        ]

        by_name: dict[str, dict] = {}
        protagonist_desc = ""
        for char in characters:
            name = str(char.get("name", "")).strip()
            if not name:
                continue
            if name in alias_to_canonical and name in {"我", "答主", "主角", "男主"}:
                protagonist_desc = str(char.get("desc", "")).strip() or protagonist_desc
            canonical = alias_to_canonical.get(name, name)
            if canonical in by_name:
                old_desc = str(by_name[canonical].get("desc", "")).strip()
                new_desc = str(char.get("desc", "")).strip()
                if len(new_desc) > len(old_desc):
                    by_name[canonical]["desc"] = new_desc
            else:
                by_name[canonical] = {
                    "name": canonical,
                    "desc": str(char.get("desc", "")).strip(),
                }

        protagonist = alias_to_canonical.get("我") or cls._extract_protagonist_name_from_text(text)
        if protagonist and protagonist not in by_name and protagonist_desc:
            by_name[protagonist] = {"name": protagonist, "desc": protagonist_desc}

        ordered = list(by_name.values())
        ordered.sort(key=lambda c: cls._character_slot_priority(c["name"]), reverse=True)
        return ordered[:MAX_CHARACTERS], applied_aliases

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
        return normalized[:MAX_CHARACTERS]

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
            text=text[:_CHARACTER_SAMPLE_MAX_LEN],
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

    def _extract_characters_by_rules(
        self, text: str, *, sample: str | None = None
    ) -> list[dict]:
        excerpt = sample or self._sample_text_for_character_extraction(text)
        matches = re.findall(
            r"([\u4e00-\u9fa5]{2,4}?)(?:说道|问道|笑道|喊道)", excerpt
        )
        matches += re.findall(r"([\u4e00-\u9fa5]{2,4}?)说[：:「]", excerpt)
        names = [
            n for n in dict.fromkeys(matches)
            if self._is_plausible_character_name(n)
        ][:MAX_CHARACTERS]
        characters = [{"name": n, "desc": ""} for n in names]
        return self._enrich_character_descriptions(excerpt, characters)

    def _extract_characters_by_llm(
        self, text: str, *, sample: str | None = None
    ) -> list[dict]:
        excerpt = sample or self._sample_text_for_character_extraction(text)
        prompt = _CHARACTER_EXTRACT_PROMPT.format(
            max_chars=MAX_CHARACTERS, text=excerpt
        )
        if self._forced_era == CLASSICAL:
            prompt += CLASSICAL_CHARACTER_ADDENDUM
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
        return self._extract_characters_by_rules(text, sample=excerpt)

    def suggest_style(self, genre: str, era: str) -> str:
        return STYLE_MAP.get((genre, era), "anime")

    def generate_intro_variants(self, text: str, genre_info: dict) -> list[str]:
        """生成 3 条约 20 字的故事介绍（片头/封面用）。"""
        if self.budget_mode:
            return self._generate_intro_variants_by_rules(text, genre_info)
        return self._generate_intro_variants_by_llm(text, genre_info)

    @staticmethod
    def _count_cjk_chars(text: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fa5]", text))

    @classmethod
    def _normalize_intro_variants(cls, raw: list | None) -> list[str]:
        if not raw:
            return []
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                s = re.sub(r"\s+", "", item.strip())
                if s:
                    out.append(s)
        return out[:3]

    @staticmethod
    def _extract_story_title(text: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            tagged = re.match(r"【[^】]+】(.+)", line)
            if tagged:
                title = tagged.group(1).strip()
                if title:
                    return title
            if len(line) <= 12 and not line.startswith("「"):
                return line
            break
        return "本故事"

    @classmethod
    def _rule_intro_templates(cls, text: str, genre_info: dict) -> list[str]:
        title = cls._extract_story_title(text)
        genre = genre_info.get("genre") or "悬疑"
        return [
            f"{title}：深夜一句警告，邻居谁也不敢出门",
            f"{genre}短篇，猜忌与惊恐在深夜里蔓延",
            f"看《{title}》，恶意比杀人更令人窒息",
        ]

    @classmethod
    def _ensure_three_intro_variants(
        cls,
        variants: list[str],
        text: str,
        genre_info: dict,
    ) -> list[str]:
        filled = list(variants)
        for template in cls._rule_intro_templates(text, genre_info):
            if len(filled) >= 3:
                break
            if template not in filled:
                filled.append(template)
        while len(filled) < 3:
            genre = genre_info.get("genre") or "故事"
            filled.append(f"{genre}短篇，一夜惊悚层层反转")
        return filled[:3]

    def _generate_intro_variants_by_rules(self, text: str, genre_info: dict) -> list[str]:
        return self._rule_intro_templates(text, genre_info)

    def _generate_intro_variants_by_llm(self, text: str, genre_info: dict) -> list[str]:
        sample = text[:_INTRO_SAMPLE_MAX_LEN]
        prompt = _INTRO_VARIANTS_PROMPT.format(text=sample)
        try:
            result = self._get_llm().chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
            )
            data = extract_json_obj(result.content)
            if data and isinstance(data.get("intro_variants"), list):
                variants = self._normalize_intro_variants(data["intro_variants"])
                if variants:
                    return self._ensure_three_intro_variants(variants, text, genre_info)
        except Exception as e:
            log.warning("LLM 故事介绍生成失败 (%s)，回退到规则", e)
        return self._generate_intro_variants_by_rules(text, genre_info)

    @classmethod
    def intro_variants_summary(cls, variants: list[str]) -> str:
        """决策日志用：各条字数摘要。"""
        parts = []
        for i, s in enumerate(variants, 1):
            parts.append(f"{i}.({cls._count_cjk_chars(s)}字){s}")
        return " | ".join(parts)


def _log_character_aliases(
    alias_map: dict[str, str],
    applied: list[tuple[str, str]] | None = None,
    *,
    source: str = "ContentAnalyzer",
) -> None:
    """将识别出的角色别名映射打印到日志。"""
    if not alias_map:
        return
    mapping_label = ", ".join(
        f"{alias}→{canonical}"
        for alias, canonical in sorted(alias_map.items())
    )
    log.info("[%s] 角色别名映射: %s", source, mapping_label)
    if applied:
        applied_label = ", ".join(f"{alias}→{canonical}" for alias, canonical in applied)
        log.info("[%s] 已合并别名: %s", source, applied_label)


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
    era_override = state.get("era_override") or config.get("promptgen", {}).get("era")
    if era_override:
        agent.set_era(era_override)
        forced = normalize_era(era_override)
        if forced:
            genre_info["era"] = era_display_name(forced)
            decisions.append(make_decision(
                "ContentAnalyzer", "era_override",
                f"时代锁定={genre_info['era']}",
                f"来源={era_override}",
            ))

    decisions.append(make_decision(
        "ContentAnalyzer", "classify",
        f"类型={genre_info['genre']}, 时代={genre_info.get('era', '未知')}",
        f"置信度={genre_info.get('confidence', 0)}",
    ))

    # 3. 角色提取
    characters = agent.extract_characters(state["full_text"])
    registry_path = state.get("series_registry_path")
    episode_id = state.get("episode_id")
    if registry_path:
        from src.promptgen.character_registry import CharacterRegistry

        registry = CharacterRegistry.load(Path(registry_path))
        registry.merge_character_list(characters, episode=episode_id)
        characters = registry.apply_canonical_to(characters)
        if agent.forced_era == CLASSICAL:
            for char in characters:
                if isinstance(char, dict):
                    char["desc"] = sanitize_classical_desc(str(char.get("desc", "")))
        registry.save()
        decisions.append(make_decision(
            "ContentAnalyzer", "series_registry",
            f"系列角色表 {len(registry.characters)} 人，本集对齐 {len(characters)} 人",
            str(registry_path),
        ))
    _log_character_descriptions(characters)

    existing_pov = state.get("pov_narrator")
    alias_for_pov = agent._last_character_alias_map
    if not alias_for_pov:
        alias_for_pov = ContentAnalyzerAgent._build_character_alias_map(state["full_text"])
    pov_narrator = existing_pov or agent.resolve_pov_narrator(
        state["full_text"],
        characters,
        alias_map=alias_for_pov,
    )
    if pov_narrator and not existing_pov:
        decisions.append(make_decision(
            "ContentAnalyzer",
            "pov_narrator",
            f"第一人称叙述者={pov_narrator}",
            f"别名映射={alias_for_pov.get('我', '—')}",
        ))
        log.info("[ContentAnalyzer] 叙述者 POV 锁定: %s", pov_narrator)

    from src.promptgen.visual_state_planner import apply_visual_states_to_characters

    try:
        llm = None if budget_mode else agent._get_llm()
    except Exception:
        llm = None
    characters, vs_discussion = apply_visual_states_to_characters(
        characters,
        segments,
        llm=llm,
        budget_mode=budget_mode,
        config=config,
        full_text=state["full_text"],
    )
    planned = [
        c.get("name")
        for c in characters
        if isinstance(c, dict) and c.get("visual_states")
    ]
    if planned:
        decisions.append(make_decision(
            "ContentAnalyzer",
            "visual_states",
            f"分段外观规划 {len(planned)} 个角色",
            f"角色: {planned}",
        ))
        log.info("[ContentAnalyzer] visual_states 已规划: %s", planned)
    if vs_discussion:
        decisions.append(make_decision(
            "ContentAnalyzer",
            "visual_state_review",
            f"分段外观审核定稿 {len(planned)} 个角色",
            "\n".join(vs_discussion)[:800],
        ))

    if agent._last_character_review_log:
        decisions.append(make_decision(
            "ContentAnalyzer",
            "character_review",
            f"双 AI 讨论定稿 {len(characters)} 个角色",
            "\n".join(agent._last_character_review_log)[:800],
        ))
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

    # 5. 故事介绍（约 20 字 × 3）
    intro_variants = agent.generate_intro_variants(state["full_text"], genre_info)
    decisions.append(make_decision(
        "ContentAnalyzer", "intro_variants",
        f"生成 {len(intro_variants)} 条故事介绍",
        agent.intro_variants_summary(intro_variants),
    ))

    log.info(
        "[ContentAnalyzer] %s/%s风格, %d段, %d角色, POV=%s, 介绍=%s",
        genre_info["genre"],
        style,
        len(segments),
        len(characters),
        pov_narrator or existing_pov or "—",
        intro_variants[0][:24] if intro_variants else "—",
    )

    result: dict = {
        "segments": segments,
        "genre": genre_info["genre"],
        "era": genre_info.get("era"),
        "era_override": era_override if era_override and normalize_era(era_override) else None,
        "characters": characters,
        "suggested_style": style,
        "intro_variants": intro_variants,
        "decisions": decisions,
    }
    if pov_narrator:
        result["pov_narrator"] = pov_narrator
    return result
