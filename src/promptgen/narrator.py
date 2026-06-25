"""叙述者识别 — 为第一人称段落绑定正确的叙述者性别与外观。"""

from __future__ import annotations

import re
from typing import Any, Literal

from src.promptgen.visual_state import resolve_character_desc

NarratorVoice = Literal["male", "female"]

_FEMALE_MARKERS = re.compile(r"本姑娘|本小姐|本女")
_FIRST_PERSON = re.compile(r"我|我们|本人|咱")

_DEFAULT_MALE_NARRATOR_DESC = (
    "男，青年第一人称叙述者，身材中等偏瘦，短发，面容清秀，"
    "穿日常休闲或校服，神情略带书卷气"
)
_DEFAULT_FEMALE_NARRATOR_DESC = (
    "女，童年/少女第一人称叙述者，身材娇小，黑色短发或 messy hair，"
    "穿简单白色T恤或小学校服，质朴儿童外观"
)

# 预填角色表中可视为「已明确绑定第一人称我」的别名
_NARRATOR_IDENTITY_ALIASES = ("303", "答主（我）", "答主", "我", "主角")

_OMIT_NARRATOR_INSTRUCTION = (
    "本段为第一人称叙述，但叙述者「我」的身份或外观未明确。"
    "禁止将「我」画成画面中的可见人物（不要出现叙述者正面、全身或清晰侧脸）；"
    "优先第三人称场景镜头，或仅呈现段内其他角色与环境；"
    "叙述者至多仅以虚化背影、过肩远景或完全不在画内表达视角。"
)

_OMIT_NARRATOR_PROMPT_SUFFIX = (
    "third person cinematic shot, focus on scene and other characters, "
    "no visible narrator protagonist, no first person face or full body in frame"
)

_SPEECH_BEFORE_QUOTE = r"[^「」]{0,30}?(?:说|道)[：:][「「]"
_OBJECT_OF_GAZE_RE = re.compile(r"(?:指着|望着|看着|瞪着|朝向)\s*$")
_ANTECEDENT_SHE_SAYS_RE = re.compile(
    r"([\u4e00-\u9fa5]{2,3}).*?她(?:激动地|决然地|低声地|大声地|轻声地)?(?:地说|说|道)[：:][「「]"
)
_ANTECEDENT_HE_SAYS_RE = re.compile(
    r"([\u4e00-\u9fa5]{2,3}).*?他(?:轻声|冷冷|大声|低声|激动)?(?:地说|说|道)[：:][「「]"
)
_PROTAGONIST_NARRATION_RE = re.compile(
    r"我心里|我(?:一听|不由|惊讶|激动|愣|顿时|立即)"
)


def _strip_quoted_text(text: str) -> str:
    """移除引号内对白，避免「我之前」等触发男主第一人称误判。"""
    return re.sub(r"「[^」]*」", "", text)


def _quote_balance(text: str) -> int:
    """未闭合引号数量（>0 表示仍有进行中的对白）。"""
    return (
        text.count("「") + text.count('"')
        - text.count("」") - text.count('"')
    )


def _name_initiates_speech(text: str, name: str) -> bool:
    """角色名后主动开口（排除「周玲听…」「转向周玲，…问道」等被动提及）。"""
    escaped = re.escape(name)
    for match in re.finditer(escaped, text):
        start = match.start()
        end = match.end()
        if re.search(r"转向\s*" + escaped, text[max(0, start - 8): end + 2]):
            continue
        if _OBJECT_OF_GAZE_RE.search(text[max(0, start - 4):start]):
            continue
        tail = text[end : end + 12]
        if re.match(r"[，,]?[听见望盯]", tail):
            continue
        after = text[end:]
        speech = re.search(_SPEECH_BEFORE_QUOTE, after)
        if not speech or "」" in after[: speech.start()]:
            continue
        return True
    return False


def _names_speaking_in_text(
    text: str, seeded_names: frozenset[str]
) -> list[str]:
    """按预填角色名在文本中的「…说/道：「」位置识别说话人（避免误匹配 你喜欢 等）。"""
    speakers: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name in seeded_names and name not in seen:
            speakers.append(name)
            seen.add(name)

    for name in sorted(seeded_names, key=len, reverse=True):
        if _name_initiates_speech(text, name):
            add(name)

    for pattern in (_ANTECEDENT_SHE_SAYS_RE, _ANTECEDENT_HE_SAYS_RE):
        for match in pattern.finditer(text):
            add(match.group(1))

    return speakers


def _speaker_before_last_open_quote(
    text: str, seeded_names: frozenset[str]
) -> str | None:
    """取段末未闭合引号外、紧邻「说/道：「」的说话人。"""
    if _quote_balance(text) <= 0:
        return None
    speakers = _names_speaking_in_text(text, seeded_names)
    return speakers[-1] if speakers else None


def collect_dialogue_speakers(
    text: str, seeded_names: frozenset[str]
) -> list[str]:
    """收集本段内实际开口对白的人物（按出现顺序）。"""
    return _names_speaking_in_text(text, seeded_names)


def _has_protagonist_narration(text: str) -> bool:
    """段落含男主第一人称旁白（如「我心里…」），不应整段绑定为对白说话人。"""
    return bool(_PROTAGONIST_NARRATION_RE.search(_strip_quoted_text(text)))


def _seeded_name_set(seeded_characters: list[dict[str, Any]]) -> frozenset[str]:
    return frozenset(
        str(entry.get("name", "")).strip()
        for entry in seeded_characters
        if isinstance(entry, dict) and str(entry.get("name", "")).strip()
    )


def find_quotation_speaker(
    text: str,
    prev_text: str | None,
    seeded_characters: list[dict[str, Any]],
) -> str | None:
    """识别引号对白续段的说话人（如上周玲说「…」，本段续「我…」）。"""
    seeded_names = _seeded_name_set(seeded_characters)
    if not seeded_names:
        return None

    speakers = _names_speaking_in_text(text, seeded_names)
    has_quotes = "「" in text or '"' in text

    if (
        len(speakers) == 1
        and not _has_protagonist_narration(text)
        and has_quotes
        and (_quote_balance(text) > 0 or text.count("」") > 0)
    ):
        return speakers[0]

    match = _ANTECEDENT_SHE_SAYS_RE.search(text)
    if match and match.group(1) in seeded_names:
        return match.group(1)

    if _quote_balance(text) > 0:
        speaker = _speaker_before_last_open_quote(text, seeded_names)
        if speaker:
            return speaker

    if prev_text and _quote_balance(prev_text) > 0:
        match = _ANTECEDENT_SHE_SAYS_RE.search(prev_text)
        if match and match.group(1) in seeded_names:
            return match.group(1)

        speaker = _speaker_before_last_open_quote(prev_text, seeded_names)
        if speaker:
            return speaker

        prev_speakers = _names_speaking_in_text(prev_text, seeded_names)
        if prev_speakers:
            return prev_speakers[-1]

    return None


def _get_character_desc(
    name: str,
    seeded_characters: list[dict[str, Any]],
    *,
    segment_index: int = 0,
) -> str:
    for entry in seeded_characters:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name", "")).strip() == name:
            return resolve_character_desc(entry, segment_index)
    return ""


def build_quotation_speaker_context(
    text: str,
    prev_text: str | None,
    seeded_characters: list[dict[str, Any]],
    *,
    segment_index: int = 0,
) -> tuple[str, str]:
    """引号内第一人称对白的说话者外观与 LLM 约束说明。"""
    speaker = find_quotation_speaker(text, prev_text, seeded_characters)
    if not speaker:
        return "", ""

    desc = _get_character_desc(speaker, seeded_characters, segment_index=segment_index)
    if not desc:
        return "", ""

    gender_hint = "女性" if _is_female_desc(desc) else "男性"
    context = text + (prev_text or "")
    if "张得胜" in context:
        scene = f"本段是角色「{speaker}」回应张得胜审问/对话"
    else:
        scene = f"本段是角色「{speaker}」的对白场景"
    instruction = (
        f"{scene}；引号内的「我」是{speaker}在自述时指代自己，不是男主李同学。"
        f"画面说话者必须是{gender_hint}角色「{speaker}」，"
        f"不可画成林映洁（女友）或其他人物。外观：\n{desc}"
    )
    return desc, instruction


def find_interrogation_addressee(
    text: str, seeded_characters: list[dict[str, Any]]
) -> str | None:
    """张得胜转向/面对某角色追问时，画面焦点应是被问方（如林郁、刘丽华）。"""
    seeded_names = _seeded_name_set(seeded_characters)
    if "张得胜" not in seeded_names:
        return None

    turn_suffixes = (
        r"(?:转向|朝向|面对|扭向|拷问)[^。！？「」]{0,30}?",
        r"脸朝向",
    )
    for name in sorted(seeded_names, key=len, reverse=True):
        if name in {"张得胜", "李同学"}:
            continue
        escaped = re.escape(name)
        for prefix in turn_suffixes:
            if re.search(prefix + escaped, text):
                return name
    return None


def build_interrogation_addressee_context(
    text: str,
    seeded_characters: list[dict[str, Any]],
    *,
    segment_index: int = 0,
) -> tuple[str, str]:
    """张得胜向特定角色提问时，绑定被问者外观（非林映洁）。"""
    focus = find_interrogation_addressee(text, seeded_characters)
    if not focus:
        return "", ""

    desc = _get_character_desc(focus, seeded_characters, segment_index=segment_index)
    if not desc:
        return "", ""

    instruction = (
        f"本段是张得胜向「{focus}」提问/分析。画面焦点必须是{focus}，"
        f"不是林映洁（女友）、不是男主李同学。"
        f"林郁与林映洁是不同角色，不可混淆。{focus}外观：\n{desc}"
    )
    return desc, instruction


def build_dialogue_scene_context(
    text: str,
    prev_text: str | None,
    seeded_characters: list[dict[str, Any]],
    *,
    segment_index: int = 0,
) -> tuple[str, str]:
    """多人对话场景（如周玲与张得胜对答），须同时呈现各方外观。"""
    seeded_names = _seeded_name_set(seeded_characters)
    if not seeded_names:
        return "", ""

    participants = collect_dialogue_speakers(text, seeded_names)
    if len(participants) < 2:
        return "", ""

    lines: list[str] = []
    for name in participants:
        desc = _get_character_desc(name, seeded_characters, segment_index=segment_index)
        if desc:
            lines.append(f"{name}：{desc}")

    if not lines:
        return "", ""

    desc_block = "\n".join(lines)
    names_label = "与".join(participants)
    instruction = (
        f"本段是{names_label}之间的对话场景（不是男主李同学发言）。"
        f"画面须同时呈现{'和'.join(participants)}两人对谈，"
        f"各自外观与下列描述一致：\n{desc_block}"
    )
    return desc_block, instruction


def find_visual_focus_character(
    text: str, seeded_characters: list[dict[str, Any]]
) -> str | None:
    """男主第一人称叙述中，识别其所观察到的画面焦点角色（如望着我的周玲）。"""
    if not _FIRST_PERSON.search(_strip_quoted_text(text)):
        return None

    seeded_names = _seeded_name_set(seeded_characters)
    for name in sorted(seeded_names, key=len, reverse=True):
        escaped = re.escape(name)
        patterns = (
            rf"望着我的{escaped}",
            rf"(?:看到|发觉)[^。！？「」]{{0,80}}?{escaped}"
            rf"[^。！？「」]{{0,40}}?(?:脸上|表情|眼神)",
            rf"{escaped}的(?:脸上|表情|眼神|目光)",
            rf"{escaped}[^，,。]{{0,15}}?(?:侧脸|愤怒)",
        )
        for pattern in patterns:
            if re.search(pattern, text):
                return name
    return None


def find_reaction_focus_character(
    text: str, seeded_characters: list[dict[str, Any]]
) -> str | None:
    """被指认、脸色突变等反应镜头：画面焦点为该角色（如指着周玲、周玲脸色变白）。"""
    seeded_names = _seeded_name_set(seeded_characters)
    outside = _strip_quoted_text(text)
    for name in sorted(seeded_names, key=len, reverse=True):
        if name in {"李同学", "张得胜"}:
            continue
        escaped = re.escape(name)
        patterns = (
            rf"指着{escaped}",
            rf"{escaped}的脸色[^。！？]{{0,24}}?(?:刷|变得|瞬间|一下子)",
            rf"{escaped}的(?:脸上|表情|眼神|目光)[^。！？]{{0,30}}?(?:刷|变得|流|颤抖)",
        )
        for pattern in patterns:
            if re.search(pattern, outside):
                return name
    return None


def build_reaction_focus_context(
    text: str,
    seeded_characters: list[dict[str, Any]],
    *,
    segment_index: int = 0,
) -> tuple[str, str]:
    """被指认/情绪反应段：绑定焦点角色外观（如短发周玲）。"""
    focus = find_reaction_focus_character(text, seeded_characters)
    if not focus:
        return "", ""

    desc = _get_character_desc(focus, seeded_characters, segment_index=segment_index)
    if not desc:
        return "", ""

    extra = ""
    if focus == "周玲":
        extra = "必须是利落黑色短发的周玲，不可画成长发林映洁（女友）或林郁。"

    instruction = (
        f"本段画面焦点是角色「{focus}」（被指认、脸色变化或情绪反应镜头）。"
        f"{extra}男主李同学仅可作背影/侧面或伸手指向，不可抢镜。"
        f"{focus}外观与下列描述一致：\n{desc}"
    )
    return desc, instruction


def build_observed_character_context(
    text: str,
    seeded_characters: list[dict[str, Any]],
    *,
    segment_index: int = 0,
) -> tuple[str, str]:
    """男主视角下的观察镜头：焦点是被看/被写的角色，而非男主自身。"""
    focus = find_visual_focus_character(text, seeded_characters)
    if not focus:
        return "", ""

    desc = _get_character_desc(focus, seeded_characters, segment_index=segment_index)
    if not desc:
        return "", ""

    instruction = (
        f"本段为男主第一人称视角，但他在观察角色「{focus}」（如望着男主、愤怒表情等）。"
        f"画面焦点必须是「{focus}」，男主仅作画外视角或背影/侧面，不可把男主画成画面主体。"
        f"{focus}外观与下列描述一致：\n{desc}"
    )
    return desc, instruction


def build_scene_character_context(
    text: str,
    prev_text: str | None,
    seeded_characters: list[dict[str, Any]],
    *,
    segment_index: int = 0,
) -> tuple[str, str]:
    """按优先级选择对白场景、独白说话人或默认叙述者绑定。"""
    reaction_prompt, reaction_instruction = build_reaction_focus_context(
        text, seeded_characters, segment_index=segment_index
    )
    if reaction_instruction:
        return reaction_prompt, reaction_instruction

    dialogue_prompt, dialogue_instruction = build_dialogue_scene_context(
        text, prev_text, seeded_characters, segment_index=segment_index
    )
    if dialogue_instruction:
        return dialogue_prompt, dialogue_instruction

    if not _has_protagonist_narration(text):
        inter_prompt, inter_instruction = build_interrogation_addressee_context(
            text, seeded_characters, segment_index=segment_index
        )
        if inter_instruction:
            return inter_prompt, inter_instruction

    quote_prompt, quote_instruction = build_quotation_speaker_context(
        text, prev_text, seeded_characters, segment_index=segment_index
    )
    if quote_instruction and not _has_protagonist_narration(text):
        return quote_prompt, quote_instruction

    observed_prompt, observed_instruction = build_observed_character_context(
        text, seeded_characters, segment_index=segment_index
    )
    if observed_instruction:
        return observed_prompt, observed_instruction

    return "", ""


def detect_narrator_voice(text: str) -> NarratorVoice | None:
    """根据段落文本判断第一人称叙述者性别（忽略引号内对白）。"""
    if not text or not text.strip():
        return None
    outside = _strip_quoted_text(text)
    if _FEMALE_MARKERS.search(outside):
        return "female"
    if _FIRST_PERSON.search(outside):
        return "male"
    return None


def _is_male_desc(desc: str) -> bool:
    return bool(re.search(r"男|male|\bboy\b|\bman\b", desc, re.IGNORECASE))


def _is_female_desc(desc: str) -> bool:
    return bool(re.search(r"女|female|\bgirl\b|\bwoman\b", desc, re.IGNORECASE))


def _find_protagonist_desc(
    seeded_characters: list[dict[str, Any]],
    voice: NarratorVoice,
) -> str:
    """从 ContentAnalyzer 预填角色中查找叙述者外观，否则用默认值。"""
    if voice == "female":
        default = _DEFAULT_FEMALE_NARRATOR_DESC
        gender_ok = _is_female_desc
    else:
        default = _DEFAULT_MALE_NARRATOR_DESC
        gender_ok = _is_male_desc

    if not seeded_characters:
        return default

    for entry in seeded_characters:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        desc = str(entry.get("desc", "")).strip()
        if not desc or not gender_ok(desc):
            continue
        if "答主" in name or name in {"我", "答主（我）", "主角"}:
            return desc

    for entry in seeded_characters:
        if not isinstance(entry, dict):
            continue
        desc = str(entry.get("desc", "")).strip()
        if desc and gender_ok(desc):
            return desc

    return default


def is_generic_narrator_desc(desc: str) -> bool:
    """是否为无具体人设的默认叙述者占位描述。"""
    d = (desc or "").strip()
    if not d:
        return True
    return d in {_DEFAULT_MALE_NARRATOR_DESC, _DEFAULT_FEMALE_NARRATOR_DESC}


def resolve_narrator_visual(
    pov_narrator_name: str | None,
    seeded_characters: list[dict[str, Any]],
    *,
    segment_index: int = 0,
) -> tuple[str | None, str]:
    """解析可入画的叙述者：须 pov 锁定或预填别名，且具非泛化外观描述。"""
    seeded = {
        str(entry.get("name", "")).strip(): entry
        for entry in seeded_characters
        if isinstance(entry, dict) and str(entry.get("name", "")).strip()
    }
    candidates: list[str] = []
    if pov_narrator_name and str(pov_narrator_name).strip():
        candidates.append(str(pov_narrator_name).strip())
    for alias in _NARRATOR_IDENTITY_ALIASES:
        if alias in seeded and alias not in candidates:
            candidates.append(alias)

    for name in candidates:
        desc = _get_character_desc(name, seeded_characters, segment_index=segment_index)
        if desc and not is_generic_narrator_desc(desc):
            return name, desc
    return None, ""


def build_omit_narrator_instruction() -> str:
    return _OMIT_NARRATOR_INSTRUCTION


def build_omit_narrator_prompt_suffix() -> str:
    return _OMIT_NARRATOR_PROMPT_SUFFIX


def build_narrator_instruction_from_identity(
    name: str, desc: str
) -> tuple[NarratorVoice | None, str]:
    """已知叙述者姓名与外观时的 LLM 约束。"""
    if _is_female_desc(desc):
        gender_en: NarratorVoice = "female"
        label = f"本段为第一人称叙述，叙述者为女性角色「{name}」"
    elif _is_male_desc(desc):
        gender_en = "male"
        label = f"本段为第一人称叙述，叙述者为男性角色「{name}」"
    else:
        gender_en = None
        label = f"本段为第一人称叙述，叙述者为角色「{name}」"

    if gender_en:
        instruction = (
            f"{label}。叙述者必须是 {gender_en}，外观必须与下列描述一致，"
            f"绝不可换成异性：\n{desc}"
        )
    else:
        instruction = f"{label}。叙述者外观必须与下列描述一致：\n{desc}"
    return gender_en, instruction


def segment_has_first_person(text: str) -> bool:
    """段落是否含第一人称叙述（含「我」）。"""
    return detect_narrator_voice(text) is not None


def build_narrator_character_prompt(
    text: str,
    seeded_characters: list[dict[str, Any]],
) -> str:
    """返回应注入 prompt 的叙述者外观描述（中文）。"""
    voice = detect_narrator_voice(text)
    if voice is None:
        return ""
    return _find_protagonist_desc(seeded_characters, voice)


def build_narrator_instruction(
    text: str,
    seeded_characters: list[dict[str, Any]],
) -> tuple[NarratorVoice | None, str]:
    """返回 (voice, LLM 用户消息附加说明)。"""
    voice = detect_narrator_voice(text)
    if voice is None:
        return None, ""

    desc = _find_protagonist_desc(seeded_characters, voice)
    if not desc:
        return voice, ""

    if voice == "female":
        label = "本段为第一人称女性叙述（童年/少女视角）"
        gender_en = "female"
    else:
        label = "本段为第一人称男性叙述者（答主/我）"
        gender_en = "male"

    instruction = (
        f"{label}。叙述者必须是 {gender_en}，外观必须与下列描述一致，"
        f"绝不可换成异性：\n{desc}"
    )
    return voice, instruction
