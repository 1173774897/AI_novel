"""时代背景（古代/现代）约束 — 场景与角色外观。"""

from __future__ import annotations

import re

CLASSICAL = "classical"
MODERN = "modern"

CLASSICAL_IMAGE_LLM_NOTE = (
    "\n\n【时代：中国古代】场景与人物必须严格符合古代设定：\n"
    "- 场景：内宅、府邸、宫廷、街市、军营、山水等，禁止现代建筑、电梯、汽车、"
    "路灯、霓虹、手机、电脑、玻璃幕墙\n"
    "- 人物：着汉服/古装（襦裙、长袍、甲胄等），发型为古代发髻/束发，"
    "禁止 T 恤、西装、牛仔裤、运动鞋、眼镜、耳机、现代短发寸头\n"
    "- 英文 prompt 须含 ancient China, traditional Chinese costume, historical setting"
)

CLASSICAL_CHARACTER_ADDENDUM = (
    "\n\n【时代：中国古代，非现代】\n"
    "- 服装必须是汉服/古装（如襦裙、长袍、褙子、甲胄），禁止现代服饰\n"
    "- 发型必须是古代款式（如双环髻、堕马髻、束发、玉冠、发簪），"
    "禁止「短发」「齐肩发」「寸头」「马尾（运动风）」等现代发型表述\n"
    "- 禁止出现手机、眼镜、耳机、T恤、西装、牛仔裤等现代物品\n"
    "- 场景相关细节（如配饰）也需符合古代"
)

_CLASSICAL_DESC_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"黑色?齐肩直发"), "乌黑长发绾成堕马髻"),
    (re.compile(r"黑色?短发"), "乌发以玉冠或束巾绾起"),
    (re.compile(r"黑色?长发微卷"), "乌黑长发绾成垂鬟分髾髻"),
    (re.compile(r"简单扎成低马尾"), "乌黑长发梳成简单发髻"),
    (re.compile(r"低马尾"), "低垂发髻"),
    (re.compile(r"运动"), ""),
    (re.compile(r"T恤|牛仔裤|西装|运动鞋|眼镜|耳机|手机"), ""),
]


def normalize_era(value: str | None) -> str | None:
    """将配置/状态中的时代转为 internal key；auto/空 返回 None。"""
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v or v == "auto":
        return None
    if v in (CLASSICAL, "古代", "古风", "ancient", "historical"):
        return CLASSICAL
    if v in (MODERN, "现代", "当代", "contemporary"):
        return MODERN
    return None


def era_display_name(era_key: str) -> str:
    return "古代" if era_key == CLASSICAL else "现代"


def default_hairstyle_for_era(desc: str, era_key: str) -> str:
    if era_key == CLASSICAL:
        if re.search(r"女|姑娘|小姐|丫鬟|女子|少女|夫人|娘娘", desc):
            return "乌黑长发梳成双环髻"
        return "乌发以玉冠束起"
    if re.search(r"女", desc):
        return "黑色齐肩直发"
    return "黑色短发"


def sanitize_classical_desc(desc: str) -> str:
    """修正角色描述中的现代发型/服饰用语。"""
    if not desc:
        return desc
    out = desc
    for pattern, repl in _CLASSICAL_DESC_FIXES:
        out = pattern.sub(repl, out)
    out = re.sub(r"[，,]{2,}", "，", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out
