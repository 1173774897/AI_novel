"""叙述者有限视角 — 画面仅限「我」亲眼所见或亲耳所闻。"""

from __future__ import annotations

import re
from typing import Literal

from src.promptgen.narrator import _strip_quoted_text

AuthorPovScene = Literal["chat", "heard", "memory", "witnessed", "indoor"]

_CHAT_MARKERS = re.compile(
    r"【[^】]{1,80}】|@\S+业主|业主群|楼栋群|群里|私聊|群消息|拍\d{3}|"
    r"低头一看|手机嗡|手机屏幕"
)
_HEARD_MARKERS = re.compile(
    r"听[到见闻过了]|听到|听得很|响声|动静|尖叫|隔音|楼道里|楼上|楼下|"
    r"深更半夜|归于平静|没.*?声音|摔上了门|敲门"
)
_MEMORY_MARKERS = re.compile(
    r"见过|印象里|记忆中|平时|想起|老学究|印象中"
)
_WITNESS_MARKERS = re.compile(
    r"看到|看见|望去|抬头|低头(?!一看)|打开门|开门|门口|面前|"
    r"瞄向|瞥见|注视|亲眼目睹"
)
_FIRST_PERSON_WORK_RE = re.compile(r"我|我们")
_NARRATION_FP_RE = re.compile(
    r"我(?:觉得|正在|刚|不|想|猜|听|看|拿|回|叹|皱|莫名|一时间|赶紧|试图)"
)


def detect_first_person_work(full_text: str, min_chars: int = 200) -> bool:
    """全文是否以第一人称叙述为主（用于 pov_mode=auto）。

    群聊占比高的文本「我」字密度偏低，故结合叙述句式与绝对频次判断。
    """
    if not full_text or len(full_text) < min_chars:
        return False
    outside = _strip_quoted_text(full_text)
    han = re.findall(r"[\u4e00-\u9fff]", outside)
    if len(han) < min_chars:
        return False
    fp = len(_FIRST_PERSON_WORK_RE.findall(outside))
    narr = len(_NARRATION_FP_RE.findall(outside))
    ratio = fp / len(han)
    if narr >= 12:
        return True
    if fp >= 30 and ratio >= 0.025:
        return True
    if fp >= 8 and ratio >= 0.10:
        return True
    return False


def classify_author_pov_scene(text: str) -> AuthorPovScene:
    """判断本段叙述者能「合法」呈现的画面类型。"""
    if not text or not text.strip():
        return "indoor"

    outside = _strip_quoted_text(text)
    chat_score = len(_CHAT_MARKERS.findall(text))
    heard_score = len(_HEARD_MARKERS.findall(outside))
    memory_score = len(_MEMORY_MARKERS.findall(outside))
    witness_score = len(_WITNESS_MARKERS.findall(outside))

    if witness_score >= 1 and witness_score >= heard_score:
        return "witnessed"
    if chat_score >= 2 or (chat_score >= 1 and heard_score == 0):
        return "chat"
    if heard_score >= 2 or (heard_score >= 1 and witness_score == 0):
        return "heard"
    if memory_score >= 1 and witness_score == 0 and chat_score == 0:
        return "memory"
    if chat_score >= 1:
        return "chat"
    if heard_score >= 1:
        return "heard"
    return "indoor"


def narrator_physically_present(text: str) -> bool:
    """叙述者是否与他人同处可见空间（非纯群聊/听闻）。"""
    scene = classify_author_pov_scene(text)
    return scene == "witnessed"


_SCENE_PROMPTS: dict[AuthorPovScene, tuple[str, str]] = {
    "chat": (
        "first person POV inside apartment at night, close-up of smartphone "
        "showing building group chat messages, narrator's hands holding phone, "
        "dim room lighting",
        "画面仅限叙述者「我」在自己屋内看手机群聊/私聊，不要画邻居实际遭遇或楼道全景。",
    ),
    "heard": (
        "first person POV, narrator standing tense inside apartment near front door, "
        "listening to muffled sounds through thin walls or hallway, worried expression, "
        "no killer or crime scene visible",
        "画面仅限叙述者「我」在屋内侧耳倾听（门/墙/天花板），不要画楼上打斗、"
        "邻居屋内或楼道杀人现场——那些只是「我」听到的，不是亲眼所见。",
    ),
    "memory": (
        "subjective memory flashback from narrator's POV, brief glimpse in apartment "
        "hallway, soft vignette edges, a middle-aged man with glasses seen from distance",
        "画面为叙述者「我」回忆里曾亲眼见过的片段（可带朦胧闪回感），"
        "不可添加回忆中未出现的情节画面。",
    ),
    "witnessed": (
        "first person POV, subjective camera, what the narrator directly sees in front of them",
        "画面必须是叙述者「我」当前亲眼所见的视角，镜头即「我」的眼睛。",
    ),
    "indoor": (
        "first person POV inside apartment at night, narrator alone in dim room, "
        "uneasy atmosphere, limited perspective",
        "画面为叙述者「我」在公寓内的主观视角，不要切换到他人不在场的场景。",
    ),
}


_AUTHOR_POV_BASE = (
    "【叙述者视角硬性约束】本片为第一人称有限视角。"
    "只能呈现叙述者「我」亲眼所见、亲耳所闻或手机屏幕上读到的内容；"
    "禁止上帝视角、监控视角、邻居屋内、楼道深处行凶等叙述者不在场的画面。"
    "他人对白若仅出现在群聊/转述中，不得画成当事人正在发生的实景。"
)


def build_author_pov_instruction(text: str) -> str:
    """生成注入 LLM 的叙述者视角说明（中文）。"""
    scene = classify_author_pov_scene(text)
    _, scene_note = _SCENE_PROMPTS[scene]
    return f"{_AUTHOR_POV_BASE}\n{scene_note}"


def build_author_pov_prompt_suffix(text: str) -> str:
    """本地模式追加的英文 POV 关键词。"""
    scene = classify_author_pov_scene(text)
    en, _ = _SCENE_PROMPTS[scene]
    return (
        f"{en}, first person limited perspective, subjective POV, "
        f"no omniscient view, no off-screen violence"
    )
