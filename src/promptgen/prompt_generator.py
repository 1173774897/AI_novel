"""Prompt 生成器 - 将小说文本转换为图片/视频 Prompt"""

import logging
import re

from src.promptgen.character_tracker import CharacterTracker
from src.promptgen.author_pov import (
    build_author_pov_instruction,
    build_author_pov_prompt_suffix,
    detect_first_person_work,
    narrator_physically_present,
)
from src.promptgen.narrator import (
    build_narrator_instruction_from_identity,
    build_omit_narrator_instruction,
    build_omit_narrator_prompt_suffix,
    build_scene_character_context,
    resolve_narrator_visual,
    segment_has_first_person,
)
from src.promptgen.style_presets import get_preset
from src.promptgen.visual_state import resolve_character_desc
from src.promptgen.era_context import (
    CLASSICAL,
    CLASSICAL_IMAGE_LLM_NOTE,
    normalize_era,
)

log = logging.getLogger("novel")


class PromptGenerationError(RuntimeError):
    """LLM prompt 生成失败（不可用本地规则回退）。"""


# LLM 系统提示词
_SYSTEM_PROMPT = """\
你是一个专业的 AI 绘画 Prompt 工程师。你的任务是将中文小说片段转换为 Stable Diffusion 图片生成 Prompt。

要求:
1. 分析文本中的场景、角色、动作、情绪
2. 生成英文 Stable Diffusion prompt
3. 包含: 场景描述、角色外观、动作姿态、光影氛围、画面构图
4. 使用标准 SD 关键词格式（逗号分隔的短语）
5. 突出画面感，忽略对话内容本身
6. 角色描写是最重要的部分，必须做到:
   - 明确每个角色的性别（male/female），绝不能搞混
   - 详细描述外观: 发型、发色、服装、体型、表情
   - 如果文中有男女两个角色，必须同时描写两人并明确区分
   - 例: "a tall young man in delivery uniform, short black hair" 和 "a young woman in pajamas, long hair, opening the door"
   - 角色的职业、身份要体现在服装和动作中
7. 如果文本中有多个角色互动，prompt 中必须包含所有角色

9. 恐怖/惊悚场景必须「含蓄恐怖」: 用光影、阴影、空旷、雾、冷色调、异常倒影、
   远处人影、半开门、惊惧眼神营造不安；禁止直接描绘血腥、伤口、尸体、断肢、
   内脏等露骨画面（no gore, no blood, no corpse, no explicit violence）

输出格式: 仅输出英文 prompt 文本，不要包含任何解释或前缀。
"""

_SYSTEM_PROMPT_COMFYUI = """\
你是画面描述工程师。将中文小说片段转换为英文生图 prompt。

要求:
1. 只描述画面内容：人物外观、服装、动作、场景、环境、物品、光线
2. 角色必须明确性别（male/female），外观与文中一致
3. 使用逗号分隔的英文短语
4. 禁止画风词（anime/cel shading/illustration/ghibli/watercolor）
5. 禁止画质词（masterpiece/8k/highly detailed/best quality）
6. 禁止镜头术语（POV/cinematic shot/subjective perspective/limited perspective）
7. 禁止负向词（no/not/without）和抽象氛围堆砌（cheerful mood/light storytelling）

输出格式: 仅输出英文 prompt，不要解释。
"""

_ANIME_LLM_STYLE_NOTE = (
    "\n\n画面风格必须为日系动画/插画（anime illustration, 2D cel shading），"
    "禁止 photorealistic、photo、live action、hyperrealistic、3D render 等写实关键词。"
)

_PHOTOREALISTIC_TERMS_RE = re.compile(
    r"\b("
    r"photorealistic|hyperrealistic|realistic photo|professional photography|"
    r"live action|photograph|8k photo|dslr|raw photo|cinema still"
    r")\b",
    re.IGNORECASE,
)

# 视频生成 LLM 系统提示词
_VIDEO_SYSTEM_PROMPT = """\
你是一个专业的 AI 视频 Prompt 工程师。你的任务是将中文小说片段转换为 AI 视频生成 Prompt。

要求:
1. 分析文本中的场景、角色、动作、情绪
2. 生成英文视频生成 prompt（自然语言完整句子，非关键词堆叠）
3. 必须包含以下层次:
   - 主体: 角色外观（必须明确性别male/female）、服装、体型、发型、表情
   - 如果有多个角色，必须分别描述每个人的外观和性别，不能混淆
   - 动作: 具体动作过程，添加速度修饰（slowly, gently, dramatically）
   - 场景: 环境、天气、时间
   - 光影: 光源类型、色温、氛围
   - 运镜: 选择合适的相机运动（dolly in, pan, orbit, tracking 等）
   - 画质: 4K, cinematic quality, natural colors
4. 运镜选择原则:
   - 紧张场景: slow dolly in + 手持感
   - 孤独场景: crane up 远离
   - 壮阔场景: drone/aerial shot
   - 日常对话: static shot
   - 动作场景: tracking/follow shot
   - 揭示场景: pan-to-reveal 或 pull back
   - 浪漫/温馨: slow orbit
5. 动作必须柔和自然，优先使用 slow、gentle、smooth 等修饰词
6. 末尾添加约束: "Stable character appearance, natural smooth movements, cinematic quality, 4K"
7. 如果有角色，保持其外观描述一致
8. 注意视频只有5-10秒，不要描述过多动作，聚焦最核心的一个画面转变
9. 恐怖/惊悚场景必须「含蓄恐怖」: 心理悬疑与氛围压迫为主，暗示画外威胁；
   禁止血腥、尸体特写、肢解、喷溅血液等露骨画面（no gore, no blood, no explicit violence）

输出格式: 仅输出英文 prompt 文本，不要包含任何解释或前缀。Prompt 应为 2-4 句完整的英文句子。
"""

# 恐怖片段检测（分镜级）
_HORROR_SEGMENT_RE = re.compile(
    r"恐怖|诡异|毛骨悚然|阴森|惊悚|骇人|不寒而栗|怪[异象声影]|"
    r"鬼|灵异|怨气|诅咒|腐臭|惨叫|噩梦|密[室锁]|悄无声息|阴冷|浮现|扭曲|"
    r"血[迹腥]?|尸[体形]?|杀[死害]|悬案|纵火|火灾|烧焦"
)

_HORROR_SUBTLE_IMAGE_SUFFIX = (
    "subtle psychological horror, implied dread, ominous shadows, "
    "off-screen tension, cinematic suspense, no gore, no blood, "
    "no explicit violence"
)

_HORROR_LLM_USER_NOTE = (
    "\n\n【恐怖场景】本段含恐怖/惊悚元素，请用含蓄恐怖手法："
    "光影压迫、空旷阴冷、远处人影、门缝微光、惊惧表情、扭曲倒影；"
    "不要画血腥伤口、尸体细节、断肢或任何露骨暴力画面。"
)

_TONE_LIGHT_SUFFIX = (
    "bright soft daylight or warm indoor lighting, natural vibrant colors, "
    "high key lighting, clean composition, cheerful everyday mood, "
    "not dark, not gloomy, not oppressive, not horror atmosphere, "
    "light storytelling mood"
)

_TONE_LIGHT_LLM_NOTE = (
    "\n\n【整体基调·必须遵守】轻松明亮叙事："
    "画面必须明亮、自然光或柔和暖色室内光，色彩清晰饱和；"
    "即使悬疑/回忆/冲突段落也禁止大面积暗部、雨夜冷青、低照度电影感、"
    "阴森压迫或 horror poster 风格。"
    "目标效果：日常校园/都市生活向 2D 插画，而非写实悬疑电影剧照。"
)

_TONE_LIGHT_SCENE_OVERRIDES: dict[str, str] = {
    "late at night, dark atmosphere, dim indoor lighting": (
        "quiet evening indoor scene, soft warm ambient light, well-lit room"
    ),
    "tense atmosphere, unsettling mood": (
        "calm thoughtful mood, mild curiosity, soft balanced lighting"
    ),
    "subtle psychological horror, ominous shadows, uneasy mood, implied dread, no gore": (
        "quiet mystery mood, soft shadows, curious atmosphere, bright enough to see clearly, no menace"
    ),
    "dim unsettling light, shadowy silhouette, off-screen threat, no blood no gore": (
        "soft indoor light, neutral mood, everyday scene, well lit, no blood no gore"
    ),
    "lonely atmosphere, solitary figure": (
        "quiet moment, single figure, gentle mood, soft natural light"
    ),
    "cinematic lighting, dramatic shadows": (
        "soft even lighting, natural colors, gentle shadows"
    ),
    "dark atmosphere, rain, moody": (
        "overcast daylight, soft diffused light, muted but not dark"
    ),
}

# 本段是否含可画成血腥/暴力的描写（排除「捧杀」等比喻）
_VIOLENT_CONTENT_RE = re.compile(
    r"血迹|血腥|鲜血|血浆|血渍|血[流红]|"
    r"尸[体首]?|"
    r"(?<!捧)杀(?:死|害|人|了)|"
    r"撞飞|卷入机器|"
    r"伤口|断肢|残肢|腐臭|"
    r"刀[子刺砍]?|消防斧|螺丝刀刺|"
    r"遇害|谋杀|丧命|无头|砍|劈|捅"
)

_DINING_SCENE_RE = re.compile(
    r"饭局|组局|吃饭|一块儿吃|聚餐|宴席|酒桌|饭桌|餐厅|饭店|请客|碰杯|席上|点菜"
)

_GORE_PROMPT_CHUNK_RE = re.compile(
    r"\b(?:"
    r"blood(?:stain|splatter|ied|y)?|bloody|"
    r"gore|gory|"
    r"corpse|dead body|dead man|dead woman|lifeless body|cadaver|"
    r"murder(?: scene|ed)?|slaughter|"
    r"stab(?:bing|bed)?|"
    r"decapitated|dismembered|mutilated|"
    r"open wound|severed|"
    r"blood-soaked|pool of blood|"
    r"crime scene|massacre"
    r")\b[^,]*",
    re.IGNORECASE,
)

_PEACEFUL_GUARD_SUFFIX = (
    "peaceful everyday scene, no blood, no gore, no corpse, "
    "no dead body, no wounds, no violence, clean safe environment"
)

_DINING_SCENE_SUFFIX = (
    "restaurant or private dining room, round table with dishes and chopsticks, "
    "family dinner gathering, warm indoor lighting, social meal scene"
)

_PEACEFUL_LLM_USER_NOTE = (
    "\n\n【本段无暴力/血腥描写·必须遵守】"
    "原文是日常、对话或社交场景；prompt 中禁止出现 blood、gore、corpse、"
    "dead body、wound、murder scene 等词汇；画面干净、安全，严格贴合本段情节。"
)

_COMFYUI_CONTENT_LLM_NOTE = (
    "\n\n【ComfyUI 仅画面内容】"
    "只写看得见的人、物、场景、动作、光线；"
    "不要写画风、画质、镜头、情绪套话；不要写 no/not/without。"
)

_FLUX_TONE_LIGHT_SUFFIX = (
    "bright soft daylight or warm indoor lighting, natural vibrant colors, "
    "high key lighting, clean composition, cheerful everyday mood, light storytelling mood"
)

_FLUX_HORROR_SUBTLE_SUFFIX = (
    "subtle psychological horror, implied dread, ominous shadows, "
    "off-screen tension, cinematic suspense atmosphere"
)

_ANIME_LLM_STYLE_NOTE_FLUX = (
    "\n\n画面风格必须为日系动画/插画（anime illustration, 2D cel shading），"
    "使用 anime style, cel shading, illustration 等正向关键词。"
)

_STYLE_BOILERPLATE_MARKERS: tuple[str, ...] = (
    "beautiful anime illustration",
    "anime style",
    "anime illustration",
    "2d illustration",
    "2d cel shading",
    "cel shading",
    "hand-drawn animation",
    "hand drawn animation",
    "studio ghibli",
    "vibrant colors",
    "beautiful scenery",
    "bright soft daylight",
    "warm indoor lighting",
    "high key lighting",
    "clean composition",
    "cheerful everyday mood",
    "light storytelling mood",
    "soft natural lighting",
    "clear composition",
    "balanced colors",
    "highly detailed",
    "cinematic composition",
    "dramatic lighting",
    "masterpiece",
    "best quality",
    "first person limited perspective",
    "first person pov",
    "subjective pov",
    "limited perspective",
    "no omniscient view",
    "third person cinematic shot",
    "no visible narrator",
    "no first person face",
    "peaceful everyday scene",
    "restaurant or private dining room",
    "family dinner gathering",
    "social meal scene",
    "subtle psychological horror",
    "implied dread",
    "off-screen tension",
    "cinematic suspense",
    "natural vibrant colors",
    "detailed animation",
    "narrator alone in dim room",
    "uneasy atmosphere",
    "exquisite watercolor painting of",
    "cinematic photo of",
    "masterpiece ink painting of",
    "cyberpunk scene of",
)

_NEGATIVE_PHRASE_START_RE = re.compile(
    r"^(?:not|no|without|never|avoid|excluding|don't|do not)\b",
    re.IGNORECASE,
)

# ---- 视频运镜自动匹配规则 (按优先级排列，第一个匹配即返回) ----
_CAMERA_MOVEMENT_RULES: list[tuple[str, str]] = [
    # 紧张/悬疑 -> 缓慢推进
    (r"紧张|心跳|不对劲|危险|杀气|恐怖|诡异", "The camera slowly dollies in"),
    # 孤独/悲伤 -> 升降远离
    (r"孤独|一个人|独自|离去|远去|消失", "The camera slowly cranes upward, pulling away"),
    # 壮阔场景 -> 航拍
    (r"山顶|战场|全城|远方|天地|苍茫", "Aerial drone shot sweeping over"),
    # 角色登场 -> 从下往上
    (r"出现|走来|现身|登场|站在.*面前", "The camera tilts up from ground level"),
    # 追逐/动作 -> 跟拍
    (r"追|跑|逃|冲|飞|闪|躲", "Tracking shot following"),
    # 环顾/展示 -> 环绕
    (r"环顾|四周|周围|打量|审视", "The camera slowly orbits around"),
    # 揭示/发现 -> 拉镜
    (r"发现|原来|看到|映入|展现|豁然", "The camera pulls back to reveal"),
    # 回忆/梦境 -> 缓慢zoom
    (r"回忆|想起|记得|梦|往事|从前", "Slow zoom in with soft focus"),
    # 对话/日常 -> 静镜
    (r"说道|问道|笑道|答道|聊|谈", "Static medium shot"),
]

# 默认运镜（当没有规则匹配时）
_DEFAULT_CAMERA = "Gentle dolly in"

# ---- 现代都市场景规则 ----
_MODERN_RULES: list[tuple[str, str]] = [
    # 人物 - 现代装扮
    (r"外卖|快递|骑手", "a delivery person in uniform, holding a takeout bag"),
    (r"口罩", "wearing a face mask"),
    (r"手机|刷视频|相册|照片|拍照", "holding a smartphone"),
    (r"耳机", "wearing earphones"),
    (r"眼镜", "wearing glasses"),
    (r"西装|领带|衬衫", "wearing a business suit"),
    (r"校服|学生", "wearing a school uniform"),
    (r"睡衣|睡眠|失眠|床上|躺在床", "in pajamas"),
    (r"工服|工作服|制服", "wearing a work uniform"),
    # 人物动作 - 现代
    (r"刷牙|洗脸|照镜子|镜子", "looking into a bathroom mirror"),
    (r"做饭|厨房|炒菜|煮|烧水", "in a modern kitchen, cooking"),
    (r"打字|电脑|键盘|屏幕", "sitting at a computer desk"),
    (r"喝咖啡|咖啡", "holding a coffee cup"),
    (r"吃饭|餐桌|碗|筷子|盘子", "at a dining table with food"),
    (r"敲门|开门|关门|锁上门", "standing at a door"),
    (r"按.*按钮|按钮", "pressing a button"),
    (r"跑出|摔门|冲出", "rushing out of a room"),
    (r"喘气|心跳|害怕|吓", "with a frightened expression"),
    (r"笑了|微笑|在笑|笑着", "smiling"),
    (r"哭|流泪|眼泪|破防", "with tears in eyes, emotional"),
    # 场景 - 现代
    (r"电梯", "inside an elevator, metallic walls, floor numbers display"),
    (r"浴室|卫生间|洗手间", "a modern bathroom, white tiles"),
    (r"卧室|房间|床|枕头|被子", "a modern bedroom, dim lighting"),
    (r"客厅|沙发|电视", "a modern living room"),
    (r"厨房|冰箱|灶台", "a modern kitchen"),
    (r"走廊|过道|楼道", "a hallway, fluorescent lighting"),
    (r"阳台|晾衣|浇花", "an apartment balcony with plants"),
    (r"公寓|合租|租屋", "inside a modern apartment"),
    (r"办公室|加班|工位", "a modern office, late at night, desk lamp"),
    (r"超市|便利店", "inside a convenience store, bright lights"),
    (r"地铁|公交|车厢", "inside a subway train"),
    (r"街道|马路|人行道|十字路口", "a modern city street"),
    (r"小区|楼下|单元门", "outside an apartment building"),
    (r"学校|教室|操场", "a school campus"),
    (r"医院|病房|病号服|手术室", "a hospital room, white walls"),
    (r"窗外|窗前|窗台", "looking out a window at the city"),
    # 物品 - 现代
    (r"猫|橘猫|喵", "a cute orange tabby cat"),
    (r"狗|柴犬|汪", "a cute dog"),
    (r"外卖袋|打包", "a takeout food bag"),
    (r"纸巾", "a pack of tissues"),
    (r"钥匙", "holding keys"),
    (r"雨伞|打伞", "holding an umbrella"),
    # 时间氛围 - 现代
    (r"凌晨|深夜|半夜|夜里", "late at night, dark atmosphere, dim indoor lighting"),
    (r"清晨|早上|早晨|闹钟", "early morning, soft morning light"),
    (r"黄昏|傍晚|下班", "evening, warm sunset light through window"),
    (r"周末|休息日", "relaxed weekend atmosphere"),
    # 情绪氛围
    (r"温馨|温暖|幸福|开心", "warm and cozy atmosphere, soft lighting"),
    (r"孤独|一个人|独自", "lonely atmosphere, solitary figure"),
    (r"紧张|心跳加速|不对劲", "tense atmosphere, unsettling mood"),
    (r"恐怖|诡异|毛骨悚然", "subtle psychological horror, ominous shadows, uneasy mood, implied dread, no gore"),
    (r"血[迹腥]?|尸体|腐臭|残肢|断肢|烧焦", "dim unsettling light, shadowy silhouette, off-screen threat, no blood no gore"),
    (r"搞笑|哈哈|莫名其妙", "comedic scene, humorous mood"),
]

# ---- 古风/仙侠场景规则 ----
_CLASSICAL_RULES: list[tuple[str, str]] = [
    # 人物
    (r"[她他].*?走|行走|疾行|加快.*脚步|脚步", "a person walking"),
    (r"[她他].*?站|驻足|停下|伫立|站住", "a person standing still"),
    (r"[她他].*?坐|端坐|盘坐", "a person sitting"),
    (r"[她他].*?跑|奔跑|疾奔|飞奔", "a person running"),
    (r"剑|刀|兵器|武器|剑柄|拔剑|挥剑", "a swordsman, hand on sword hilt"),
    (r"黑衣人|蒙面人|黑袍", "a mysterious figure in black robes"),
    (r"白衣|白袍|素衣", "a figure in flowing white robes"),
    (r"斗篷|披风|长衫|长袍", "a cloaked figure, flowing cloak"),
    (r"斗笠|面纱|面具", "a figure with a bamboo hat hiding face"),
    (r"少女|女子|姑娘|小姐|美人", "a beautiful young woman"),
    (r"少年|少侠|公子|青年", "a young man, handsome"),
    (r"老者|老人|长者|老翁", "an elderly man, wise appearance"),
    (r"将军|武将|甲胄", "a general in armor"),
    (r"书生|文人|儒生", "a scholar in traditional robes"),
    # 动作
    (r"转身|回头|回首", "turning around, looking back"),
    (r"对峙|拦住|拦路|拦在", "two figures facing each other, confrontation"),
    (r"苦笑|微笑|冷笑", "with a melancholy expression"),
    (r"低喝|怒喝|大喊", "dramatic tension, someone shouting"),
    (r"把酒|饮酒|喝酒|酒杯", "drinking wine"),
    # 场景
    (r"小巷|巷子|胡同|街巷", "narrow ancient alley, stone path"),
    (r"酒楼|酒馆|客栈|茶馆", "a traditional Chinese tavern, warm lights"),
    (r"宫殿|皇宫|大殿", "a grand imperial palace"),
    (r"山顶|山巅|峰顶", "a mountaintop"),
    (r"悬崖|峭壁|绝壁", "a dramatic cliff edge"),
    (r"竹林|竹海", "a bamboo forest"),
    (r"桃花|樱花|花海", "a sea of blossoming flowers"),
    (r"江湖|码头|渡口", "a riverside dock"),
    (r"战场|沙场|两军", "a battlefield"),
    (r"书房|案几|书桌", "a traditional study room"),
    (r"青石板|石板路", "cobblestone path"),
    # 自然/天气
    (r"月光|月色|明月|弯月|圆月", "moonlight, luminous moon in sky"),
    (r"星空|繁星|星辰", "starry night sky"),
    (r"日落|夕阳|余晖", "sunset, golden hour"),
    (r"大雨|暴雨|雨中|细雨", "rain falling"),
    (r"大雪|飞雪|雪中|风雪", "snow falling, winter scene"),
    (r"浓雾|大雾|薄雾|迷雾", "misty, foggy atmosphere"),
    (r"灯火|烛光|火光|火把", "warm lantern light, candlelight"),
    # 氛围
    (r"冷冽|寒风|凛冽|刺骨", "cold atmosphere, biting wind"),
    (r"寂静|安静|无声|静谧", "quiet, serene atmosphere"),
    (r"紧张|危险|杀气|杀机", "tense atmosphere, sense of danger"),
    (r"温馨|温暖|和煦", "warm and cozy atmosphere"),
    (r"丝竹|琴声|笛声|箫声", "faint music in the air"),
]

# 现代关键词（用于自动判断时代）
_MODERN_KEYWORDS = (
    r"手机|电脑|电梯|地铁|公交|外卖|快递|公寓|合租|办公室|加班|"
    r"闹钟|微信|视频|APP|网络|WiFi|空调|冰箱|洗衣机|电视|"
    r"出租车|汽车|高铁|飞机|超市|便利店|咖啡|奶茶|"
    r"校服|T恤|牛仔裤|运动鞋|口罩|耳机|眼镜|"
    r"室友|同事|老板|客户|甲方|KPI|"
    r"抖音|朋友圈|点赞|评论|转发|备注"
)

# 古风关键词
_CLASSICAL_KEYWORDS = (
    r"剑|刀|武功|内力|真气|修仙|仙|魔|妖|灵|丹|阵法|"
    r"江湖|武林|门派|掌门|弟子|侠|义|"
    r"大侠|少侠|公子|姑娘|小姐|夫人|"
    r"皇上|陛下|臣|太子|王爷|将军|丞相|"
    r"长袍|汉服|布衣|锦衣|甲胄|斗篷|"
    r"客栈|酒楼|茶馆|青楼|书院|"
    r"马车|轿子|骏马|战马"
)


class PromptGenerator:
    """Stable Diffusion 图片 Prompt 生成器。

    支持两种工作模式:
      1. LLM 模式 (OPENAI_API_KEY 已设置): 使用 GPT 理解场景后生成高质量 prompt
      2. 本地模式 (无 API Key): 提取关键词 + 风格预设，拼接为 prompt

    自动检测文本时代背景（现代 vs 古风），选择对应的场景规则。
    通过 CharacterTracker 维护角色外观一致性。
    """

    def __init__(self, config: dict) -> None:
        self._style_name: str = config.get("style", "chinese_ink")
        self._preset = get_preset(self._style_name)

        llm_cfg = config.get("llm", {})
        self._model: str = llm_cfg.get("model", "gpt-4o-mini")
        self._temperature: float = llm_cfg.get("temperature", 0.7)

        self._use_character_tracking: bool = config.get("character_tracking", True)
        self._tracker = CharacterTracker() if self._use_character_tracking else None
        self._seeded_characters: list[dict] = []
        self._seeded_names: frozenset[str] = frozenset()

        # 检测是否可使用 LLM: 任意已知 API Key 或配置了 provider
        self._llm_config = llm_cfg
        self._use_llm: bool = self._detect_llm_available(llm_cfg)
        self._llm_client_cached = None

        # 缓存全文的时代判断结果
        self._era_cache: str | None = None
        self._horror_style: str = str(config.get("horror_style", "subtle")).lower()
        self._tone: str = str(config.get("tone", "default")).lower()
        self._pov_mode: str = str(config.get("pov_mode", "auto")).lower()
        self._pov_narrator_name: str | None = (
            str(config.get("pov_narrator")).strip()
            if config.get("pov_narrator")
            else None
        )
        self._first_person_work: bool = False
        self._era_override: str | None = normalize_era(config.get("era"))
        self._comfyui_imagegen: bool = config.get("imagegen_backend") == "comfyui"
        self._lora_trigger = str(config.get("lora_trigger") or "").strip()
        if "prompt_prefix" in config:
            self._comfyui_prompt_prefix = str(config.get("prompt_prefix") or "").strip()
        elif self._comfyui_imagegen:
            self._comfyui_prompt_prefix = "beautiful anime illustration of"
        else:
            self._comfyui_prompt_prefix = ""

        if self._comfyui_imagegen:
            log.info(
                "Prompt 生成: ComfyUI/FLUX 模式，跳过负向提示词"
                + (f"，LoRA 唤醒词={self._lora_trigger!r}" if self._lora_trigger else "")
                + (
                    f"，画风前缀={self._comfyui_prompt_prefix!r}"
                    if self._comfyui_prompt_prefix
                    else ""
                )
            )
        if self._era_override:
            log.info("时代背景已锁定: %s", self._era_override)
        if self._use_llm:
            log.info("Prompt 生成: LLM 模式 (model=%s)", self._model)
        else:
            log.info("Prompt 生成: 本地关键词模式 (style=%s)", self._style_name)

    def set_style(self, style_name: str) -> None:
        """运行时切换风格预设（如 ContentAnalyzer 推荐风格）。"""
        self._style_name = style_name
        self._preset = get_preset(style_name)

    def _append_era_llm_note(self, user_msg: str) -> str:
        if self._get_era("") == CLASSICAL:
            user_msg += CLASSICAL_IMAGE_LLM_NOTE
        return user_msg

    def _append_style_llm_note(self, user_msg: str) -> str:
        if self._style_name == "anime":
            user_msg += (
                _ANIME_LLM_STYLE_NOTE_FLUX
                if self._comfyui_imagegen
                else _ANIME_LLM_STYLE_NOTE
            )
        if self._tone == "light":
            if not self._comfyui_imagegen:
                user_msg += _TONE_LIGHT_LLM_NOTE
        return user_msg

    def _append_llm_context_notes(self, user_msg: str) -> str:
        if self._comfyui_imagegen:
            return self._append_era_llm_note(user_msg)
        user_msg = self._append_style_llm_note(user_msg)
        return self._append_era_llm_note(user_msg)

    def _image_system_prompt(self) -> str:
        return _SYSTEM_PROMPT_COMFYUI if self._comfyui_imagegen else _SYSTEM_PROMPT

    @classmethod
    def _is_style_boilerplate_segment(cls, segment: str) -> bool:
        norm = cls._normalize_phrase(segment)
        if not norm:
            return True
        for marker in _STYLE_BOILERPLATE_MARKERS:
            if norm == marker or norm.startswith(marker + ",") or marker in norm:
                return True
        if norm in {
            "detailed",
            "vibrant colors",
            "beautiful scenery",
            "cheerful mood",
            "light storytelling",
        }:
            return True
        return False

    @classmethod
    def _strip_style_boilerplate(cls, prompt: str) -> str:
        if not prompt:
            return prompt
        kept = [
            seg
            for seg in cls._split_prompt_phrases(prompt)
            if not cls._is_style_boilerplate_segment(seg)
        ]
        return ", ".join(kept)

    def _finalize_image_prompt(
        self,
        prompt: str,
        visual_text: str,
        prev_text: str | None = None,
    ) -> str:
        if self._comfyui_imagegen:
            out = self._sanitize_gore_from_prompt(prompt)
            out = self._strip_negative_phrases(out)
            out = self._strip_style_boilerplate(out)
            out = self._compact_prompt_phrases(out)
            if self._comfyui_prompt_prefix or self._lora_trigger:
                from src.imagegen.comfyui_backend import finalize_comfyui_positive_prompt

                out = finalize_comfyui_positive_prompt(
                    out,
                    lora_trigger=self._lora_trigger,
                    prompt_prefix=self._comfyui_prompt_prefix,
                )
            return out

        prompt = self._apply_subtle_horror(prompt, visual_text)
        prompt = self._apply_tone(prompt)
        prompt = self._apply_author_pov(prompt, visual_text)
        prompt = self._apply_peaceful_scene_guard(prompt, visual_text, prev_text)
        return self._compact_prompt_phrases(prompt)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def set_era(self, era: str | None) -> None:
        """锁定时代背景（古代/现代），覆盖自动检测。"""
        self._era_override = normalize_era(era)

    def set_full_text(self, full_text: str) -> None:
        """用全文来判断时代背景，缓存结果供所有片段使用。"""
        if self._era_override:
            self._era_cache = self._era_override
        else:
            self._era_cache = self._detect_era(full_text)
        self._first_person_work = detect_first_person_work(full_text)
        log.info("检测到文本时代: %s", self._era_cache)
        if self._pov_mode == "auto" and self._first_person_work:
            log.info("检测到第一人称叙述为主，启用叙述者有限视角 (pov_mode=auto)")

    def set_pov_narrator(self, name: str | None) -> None:
        """指定本集第一人称叙述者（角色名），覆盖自动检测。"""
        self._pov_narrator_name = str(name).strip() if name else None

    def seed_characters(
        self,
        characters: list[dict],
        *,
        canonical: bool = False,
    ) -> int:
        """用 ContentAnalyzer 等外部角色表预填 CharacterTracker。"""
        self._seeded_characters = [
            entry for entry in (characters or []) if isinstance(entry, dict)
        ]
        self._seeded_names = frozenset(
            str(entry.get("name", "")).strip()
            for entry in self._seeded_characters
            if str(entry.get("name", "")).strip()
        )
        if not self._tracker or not characters:
            return 0
        return self._tracker.seed_characters(characters, canonical=canonical)

    _CAST_DESC_MAX_CHARS = 100

    @staticmethod
    def _truncate_cast_desc(desc: str, max_chars: int) -> str:
        desc = (desc or "").strip()
        if len(desc) <= max_chars:
            return desc
        return desc[: max_chars - 1].rstrip() + "…"

    def _resolve_narrator_visual(self, segment_index: int = 0) -> tuple[str | None, str]:
        return resolve_narrator_visual(
            self._pov_narrator_name,
            self._seeded_characters,
            segment_index=segment_index,
        )

    def _narrator_has_visual_identity(self, segment_index: int = 0) -> bool:
        name, desc = self._resolve_narrator_visual(segment_index)
        return bool(name and desc)

    def _narrator_cast_names(self, text: str, segment_index: int = 0) -> list[str]:
        """第一人称叙述时，仅在有明确身份+外观时注入叙述者。"""
        name, _ = self._resolve_narrator_visual(segment_index)
        if name:
            return [name]
        return []

    def _build_cast_bible(self, text: str, *, segment_index: int = 0) -> str:
        """本段相关角色设定（非全片卡司），避免 prompt 过长触发即梦 InvalidNode。"""
        names: list[str] = []
        seen: set[str] = set()
        for name in self._narrator_cast_names(text, segment_index) + self._resolve_segment_characters(
            text
        ):
            if name and name not in seen:
                names.append(name)
                seen.add(name)

        if not names:
            return ""

        lines: list[str] = []
        seeded_order = [
            str(entry.get("name", "")).strip()
            for entry in self._seeded_characters
            if isinstance(entry, dict)
        ]
        for name in names:
            if name not in seeded_order:
                continue
            for entry in self._seeded_characters:
                if not isinstance(entry, dict):
                    continue
                entry_name = str(entry.get("name", "")).strip()
                if entry_name != name:
                    continue
                desc = self._truncate_cast_desc(
                    resolve_character_desc(entry, segment_index),
                    self._CAST_DESC_MAX_CHARS,
                )
                if desc:
                    lines.append(f"{name}：{desc}")
                break

        if not lines:
            return ""
        return "【本段相关角色，外观保持一致】\n" + "\n".join(lines)

    def count_segment_characters(self, text: str) -> int:
        """本段可见角色人数（与 cast bible 同源，供 ComfyUI 多人场景加步数）。"""
        names: list[str] = []
        seen: set[str] = set()
        for name in self._narrator_cast_names(text) + self._resolve_segment_characters(
            text
        ):
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        return len(names)

    def _resolve_segment_characters(self, text: str) -> list[str]:
        """本段应注入外观的角色名（子串匹配 + 正则，限预填白名单）。"""
        if not self._tracker:
            return []
        seeded_order = [
            str(entry.get("name", "")).strip()
            for entry in self._seeded_characters
            if isinstance(entry, dict) and str(entry.get("name", "")).strip()
        ]
        return self._tracker.resolve_segment_characters(
            text,
            seeded_names=seeded_order or None,
            allowed_names=self._seeded_names or None,
        )

    def _uses_author_pov(self, text: str) -> bool:
        if not self._narrator_has_visual_identity():
            return False
        if self._pov_narrator_name:
            return True
        if self._pov_mode == "off":
            return False
        if self._pov_mode == "author":
            return True
        if self._pov_mode == "auto":
            return self._first_person_work
        return False

    def _build_character_context(
        self,
        text: str,
        prev_text: str | None = None,
        *,
        segment_index: int = 0,
    ) -> tuple[str, str]:
        """组装角色外观描述与叙述者约束说明。"""
        narrator_prompt = ""
        narrator_instruction = ""
        author_pov = self._uses_author_pov(text)

        if self._seeded_characters:
            scene_prompt, scene_instruction = "", ""
            if not author_pov or narrator_physically_present(text):
                scene_prompt, scene_instruction = build_scene_character_context(
                    text, prev_text, self._seeded_characters, segment_index=segment_index
                )
            if scene_instruction:
                narrator_prompt = scene_prompt
                narrator_instruction = scene_instruction
            else:
                narrator_name, narrator_desc = self._resolve_narrator_visual(segment_index)
                if narrator_name and narrator_desc:
                    _, narrator_instruction = build_narrator_instruction_from_identity(
                        narrator_name, narrator_desc
                    )
                    narrator_prompt = narrator_desc
                elif segment_has_first_person(text):
                    narrator_instruction = build_omit_narrator_instruction()
                    narrator_prompt = ""

        if author_pov:
            pov_note = build_author_pov_instruction(text)
            narrator_instruction = (
                f"{pov_note}\n\n{narrator_instruction}"
                if narrator_instruction
                else pov_note
            )

        parts: list[str] = []
        cast_bible = self._build_cast_bible(text, segment_index=segment_index)
        if cast_bible:
            parts.append(cast_bible)

        tracker_prompt = ""
        if self._tracker:
            characters = self._resolve_segment_characters(text)
            tracker_prompt = self._tracker.get_character_prompt(
                characters,
                allowed_names=self._seeded_names or None,
                segment_index=segment_index,
            )
            if tracker_prompt:
                parts.append(f"【本段出场角色】{tracker_prompt}")

        if narrator_prompt and narrator_prompt not in tracker_prompt:
            parts.append(narrator_prompt)

        return "\n\n".join(parts), narrator_instruction

    @staticmethod
    def _visual_source_text(text: str) -> str:
        """生图用文本：去掉系列书名/章节标题，避免「暗黑」等元数据污染画面。"""
        lines = (text or "").splitlines()
        while lines:
            ln = lines[0].strip()
            if not ln:
                lines.pop(0)
                continue
            if re.match(r"^【.+】$", ln):
                lines.pop(0)
                continue
            if (
                "：" in ln
                and len(ln) < 45
                and not ln.startswith("「")
                and not re.search(r"[。！？!?]", ln)
            ):
                lines.pop(0)
                continue
            if re.match(r"^\d+\.\s*.+", ln) and len(ln) < 32 and "。" not in ln:
                lines.pop(0)
                continue
            break
        return "\n".join(lines).strip() or (text or "").strip()

    def generate(
        self,
        text: str,
        segment_index: int,
        prev_text: str | None = None,
    ) -> str:
        """将小说文本片段转换为 SD 图片 prompt。"""
        if not text or not text.strip():
            return self._preset.get("prefix", "")

        visual_text = self._visual_source_text(text)

        # 提取角色信息 + 叙述者绑定
        character_prompt, narrator_instruction = self._build_character_context(
            text, prev_text=prev_text, segment_index=segment_index
        )

        # 根据模式生成 prompt
        if self._use_llm:
            prompt = self._generate_with_llm(
                visual_text,
                character_prompt,
                narrator_instruction,
                prev_text=prev_text,
            )
        else:
            prompt = self._generate_local(
                visual_text, character_prompt, narrator_instruction
            )

        # 更新角色追踪器（仅允许已预填角色名，避免误识别污染）
        if self._tracker:
            self._tracker.update(
                text,
                prompt,
                allowed_names=self._seeded_names or None,
            )

        prompt = self._finalize_image_prompt(prompt, visual_text, prev_text)
        log.debug("段 %d prompt: %s", segment_index, prompt[:80])
        log.info("[ImageGen] 段 %d 生图 prompt:\n%s", segment_index, prompt)
        return prompt

    def generate_alternate(
        self,
        text: str,
        segment_index: int,
        variant: int = 0,
        prev_text: str | None = None,
    ) -> str:
        """拒稿后换角度重生 prompt（variant 0..2 对应不同构图）。"""
        from src.imagegen.moderation import alternate_angle_hint

        if not text or not text.strip():
            return self._preset.get("prefix", "")

        hint_cn, hint_en = alternate_angle_hint(variant)
        visual_text = self._visual_source_text(text)
        character_prompt, narrator_instruction = self._build_character_context(
            text, prev_text=prev_text, segment_index=segment_index
        )

        if self._use_llm:
            user_msg = f"小说文本:\n{visual_text}"
            if character_prompt:
                user_msg += (
                    f"\n\n已知角色描述（必须严格保持一致，尤其是性别和外观）:\n"
                    f"{character_prompt}"
                )
            user_msg = self._append_narrator_instruction(user_msg, narrator_instruction)
            if not self._comfyui_imagegen:
                user_msg += f"\n\n画面风格: {self._style_name}"
            user_msg = self._append_llm_context_notes(user_msg)
            user_msg = self._append_peaceful_llm_note(user_msg, visual_text, prev_text)
            user_msg += (
                f"\n\n【换角度重试 #{variant + 1}】{hint_cn}\n"
                f"Alternate framing (English): {hint_en}."
            )
            if not self._comfyui_imagegen:
                user_msg += " PG-13, no gore, no explicit violence."
            try:
                client = self._get_llm_client()
                response = client.chat(
                    messages=[
                        {"role": "system", "content": self._image_system_prompt()},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self._temperature,
                )
                raw = (response.content or "").strip()
                if not raw:
                    raise PromptGenerationError("LLM 返回空 prompt（换角度模式）")
                prompt = raw
            except PromptGenerationError:
                raise
            except Exception as exc:
                log.error("换角度 LLM prompt 失败: %s", exc)
                raise PromptGenerationError(
                    f"换角度 LLM prompt 生成失败: {exc}"
                ) from exc
        else:
            prompt = self._generate_local(
                visual_text, character_prompt, narrator_instruction
            )

        prompt = f"{prompt}, {hint_en}"
        prompt = self._finalize_image_prompt(prompt, visual_text, prev_text)
        log.info("[ImageGen] 段 %d 换角度生图 prompt #%d:\n%s", segment_index, variant + 1, prompt)
        log.debug("段 %d 换角度 prompt #%d: %s", segment_index, variant + 1, prompt[:80])
        return prompt

    @property
    def character_tracker(self) -> CharacterTracker | None:
        """获取角色追踪器实例（用于外部序列化/恢复）。"""
        return self._tracker

    def generate_video_prompt(
        self,
        segment_text: str,
        segment_index: int,
        prev_text: str | None = None,
    ) -> str:
        """将小说文本片段转换为视频生成 AI 的 prompt。

        视频 prompt 与图片 prompt 的主要区别:
        - 使用自然语言完整句子（而非逗号分隔的关键词）
        - 包含运镜描述（camera movement）
        - 包含角色动作过程（而非静态姿态）
        - 包含场景过渡和氛围描写

        Args:
            segment_text: 中文小说文本片段。
            segment_index: 片段在全文中的序号（从 0 开始）。

        Returns:
            英文视频生成 prompt 字符串。
        """
        if not segment_text or not segment_text.strip():
            return ""

        visual_text = self._visual_source_text(segment_text)

        # 提取角色信息 + 叙述者绑定
        character_prompt, narrator_instruction = self._build_character_context(
            segment_text, prev_text=prev_text, segment_index=segment_index
        )

        # 根据模式生成 prompt
        if self._use_llm:
            prompt = self._generate_video_with_llm(
                visual_text,
                character_prompt,
                narrator_instruction,
                prev_text=prev_text,
            )
        else:
            prompt = self._generate_video_local(
                visual_text, character_prompt, narrator_instruction
            )

        # 更新角色追踪器（仅允许已预填角色名）
        if self._tracker:
            self._tracker.update(
                segment_text,
                prompt,
                allowed_names=self._seeded_names or None,
            )

        prompt = self._apply_subtle_horror(prompt, visual_text)
        prompt = self._apply_tone(prompt)
        prompt = self._apply_author_pov(prompt, visual_text)
        prompt = self._apply_peaceful_scene_guard(prompt, visual_text, prev_text)
        log.debug("段 %d video prompt: %s", segment_index, prompt[:80])
        return prompt

    def _apply_author_pov(self, prompt: str, text: str) -> str:
        if self._uses_author_pov(text):
            suffix = build_author_pov_prompt_suffix(text)
            return self._compact_prompt_phrases(
                self._append_missing_phrases(prompt, suffix)
            )
        if segment_has_first_person(text) and not self._narrator_has_visual_identity():
            suffix = build_omit_narrator_prompt_suffix()
            return self._compact_prompt_phrases(
                self._append_missing_phrases(prompt, suffix)
            )
        return prompt

    def _is_horror_segment(self, text: str) -> bool:
        return bool(_HORROR_SEGMENT_RE.search(text or ""))

    @staticmethod
    def _has_violent_content(text: str) -> bool:
        return bool(_VIOLENT_CONTENT_RE.search(text or ""))

    @staticmethod
    def _is_dining_scene(text: str, prev_text: str | None = None) -> bool:
        combined = f"{prev_text or ''}\n{text or ''}"
        return bool(_DINING_SCENE_RE.search(combined))

    @staticmethod
    def _sanitize_gore_from_prompt(prompt: str) -> str:
        if not prompt:
            return prompt
        cleaned = _GORE_PROMPT_CHUNK_RE.sub("", prompt)
        cleaned = re.sub(r",\s*,+", ", ", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" ,")

    @classmethod
    def _phrase_contains(cls, haystack: str, needle: str) -> bool:
        if not needle:
            return False
        if needle in haystack:
            return True
        needle_core = re.sub(r"^(?:a|an|the)\s+", "", needle)
        haystack_core = re.sub(r"^(?:a|an|the)\s+", "", haystack)
        return bool(needle_core and needle_core in haystack_core)

    @classmethod
    def _normalize_phrase(cls, phrase: str) -> str:
        return re.sub(r"\s+", " ", phrase.strip().lower())

    @classmethod
    def _split_prompt_phrases(cls, text: str) -> list[str]:
        return [part.strip() for part in text.split(",") if part.strip()]

    @classmethod
    def _append_missing_phrases(cls, prompt: str, extra: str) -> str:
        """只追加 base 中尚未出现的 comma 短语（忽略大小写）。"""
        if not extra or not extra.strip():
            return prompt.rstrip(", ").strip()

        base = prompt.rstrip(", ").strip()
        existing = cls._split_prompt_phrases(base)
        existing_norms = [cls._normalize_phrase(seg) for seg in existing]
        missing: list[str] = []

        for phrase in cls._split_prompt_phrases(extra):
            norm = cls._normalize_phrase(phrase)
            if not norm:
                continue
            if norm in (prompt or "").lower():
                continue
            if any(cls._phrase_contains(existing_norm, norm) for existing_norm in existing_norms):
                continue
            if any(cls._phrase_contains(norm, existing_norm) for existing_norm in existing_norms):
                continue
            missing.append(phrase)
            existing_norms.append(norm)

        if not missing:
            return base
        if not base:
            return ", ".join(missing)
        return f"{base}, {', '.join(missing)}"

    @classmethod
    def _compact_prompt_phrases(cls, prompt: str) -> str:
        """合并 comma 短语：去重 + 去掉被更长短语包含的短片段。"""
        segments = cls._split_prompt_phrases(prompt)
        if len(segments) <= 1:
            return prompt.strip(" ,")

        unique: list[str] = []
        seen: set[str] = set()
        for seg in segments:
            key = cls._normalize_phrase(seg)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(seg)

        norms = [cls._normalize_phrase(seg) for seg in unique]
        kept: list[str] = []
        for i, seg in enumerate(unique):
            norm = norms[i]
            if any(
                i != j
                and norm != norms[j]
                and cls._phrase_contains(norms[j], norm)
                for j in range(len(unique))
            ):
                continue
            kept.append(seg)
        return ", ".join(kept)

    @staticmethod
    def _is_negative_phrase(segment: str) -> bool:
        s = segment.strip()
        if not s:
            return True
        if s.upper().startswith("NOT "):
            return True
        return bool(_NEGATIVE_PHRASE_START_RE.match(s))

    @classmethod
    def _strip_negative_phrases(cls, prompt: str) -> str:
        if not prompt:
            return prompt
        kept = [
            part.strip()
            for part in prompt.split(",")
            if part.strip() and not cls._is_negative_phrase(part)
        ]
        return ", ".join(kept)

    @staticmethod
    def _suffix_present(prompt: str, suffix: str) -> bool:
        return suffix.lower() in (prompt or "").lower()

    @classmethod
    def _positive_only_keywords(cls, text: str) -> str:
        if not text:
            return text
        return cls._strip_negative_phrases(text.replace(";", ","))

    def _apply_peaceful_scene_guard(
        self,
        prompt: str,
        text: str,
        prev_text: str | None = None,
    ) -> str:
        """非暴力分镜：剥离 prompt 中的血腥词，并追加干净场景约束。"""
        if self._has_violent_content(text):
            return prompt
        out = self._sanitize_gore_from_prompt(prompt)
        if self._is_dining_scene(text, prev_text):
            out = self._append_missing_phrases(out, _DINING_SCENE_SUFFIX)
        if self._comfyui_imagegen:
            return self._compact_prompt_phrases(out)
        if not self._suffix_present(out, _PEACEFUL_GUARD_SUFFIX):
            out = self._append_missing_phrases(out, _PEACEFUL_GUARD_SUFFIX)
        return self._compact_prompt_phrases(out)

    def _append_peaceful_llm_note(
        self,
        user_msg: str,
        text: str,
        prev_text: str | None = None,
    ) -> str:
        if self._has_violent_content(text):
            return user_msg
        if self._comfyui_imagegen:
            user_msg += _COMFYUI_CONTENT_LLM_NOTE
        else:
            user_msg += _PEACEFUL_LLM_USER_NOTE
        if self._is_dining_scene(text, prev_text):
            if self._comfyui_imagegen:
                user_msg += (
                    "\n场景提示：本段发生在饭局/聚餐，请画餐厅或包间圆桌用餐。"
                )
            else:
                user_msg += (
                    "\n场景提示：本段发生在饭局/聚餐，请画餐厅或包间圆桌用餐，"
                    "不要画客厅、卧室或犯罪现场。"
                )
        return user_msg

    def _apply_subtle_horror(self, prompt: str, text: str) -> str:
        """恐怖分镜追加含蓄恐怖约束，避免模型画出露骨画面。"""
        if (
            self._horror_style == "off"
            or self._tone == "light"
            or not self._is_horror_segment(text)
        ):
            return prompt
        suffix = (
            _FLUX_HORROR_SUBTLE_SUFFIX
            if self._comfyui_imagegen
            else _HORROR_SUBTLE_IMAGE_SUFFIX
        )
        return self._compact_prompt_phrases(self._append_missing_phrases(prompt, suffix))

    def _apply_tone(self, prompt: str) -> str:
        """整体基调修饰（light=更轻松明亮）。"""
        if self._tone != "light":
            return prompt
        out = prompt
        if not self._comfyui_imagegen:
            for heavy, light in _TONE_LIGHT_SCENE_OVERRIDES.items():
                out = out.replace(heavy, light)
        suffix = (
            _FLUX_TONE_LIGHT_SUFFIX
            if self._comfyui_imagegen
            else _TONE_LIGHT_SUFFIX
        )
        return self._compact_prompt_phrases(self._append_missing_phrases(out, suffix))

    # ------------------------------------------------------------------
    # video prompt - LLM mode
    # ------------------------------------------------------------------

    def _append_narrator_instruction(self, user_msg: str, narrator_instruction: str) -> str:
        if narrator_instruction:
            user_msg += f"\n\n{narrator_instruction}"
        return user_msg

    def _generate_video_with_llm(
        self,
        text: str,
        character_prompt: str,
        narrator_instruction: str = "",
        *,
        prev_text: str | None = None,
    ) -> str:
        """使用 LLM 生成视频 prompt。"""
        user_msg = f"小说文本:\n{text}"
        if character_prompt:
            user_msg += f"\n\n已知角色描述（必须严格保持一致，尤其是性别和外观）:\n{character_prompt}"
        user_msg = self._append_narrator_instruction(user_msg, narrator_instruction)
        user_msg += f"\n\n画面风格: {self._style_name}"
        user_msg = self._append_llm_context_notes(user_msg)
        user_msg = self._append_peaceful_llm_note(user_msg, text, prev_text)
        user_msg += "\n\n重要: 仔细分析文中每个角色的性别（他=male, 她=female），确保 prompt 中角色性别正确，不要搞混。"
        if self._horror_style != "off" and self._tone != "light" and self._is_horror_segment(text):
            user_msg += _HORROR_LLM_USER_NOTE

        try:
            client = self._get_llm_client()
            response = client.chat(
                messages=[
                    {"role": "system", "content": _VIDEO_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
            )
        except Exception as e:
            log.error("LLM 视频 prompt 生成失败: %s", e)
            raise PromptGenerationError(f"LLM 视频 prompt 生成失败: {e}") from e

        raw_prompt = (response.content or "").strip()
        if not raw_prompt:
            raise PromptGenerationError("LLM 返回空视频 prompt")

        return self._apply_video_style(raw_prompt)

    # ------------------------------------------------------------------
    # video prompt - local fallback mode
    # ------------------------------------------------------------------

    def _generate_video_local(
        self,
        text: str,
        character_prompt: str,
        narrator_instruction: str = "",
    ) -> str:
        """使用规则匹配生成视频 prompt（无 API 依赖）。

        在图片 prompt 的场景元素基础上，将其改写为自然语言句子，
        并追加运镜描述和视频画质约束。
        """
        era = self._get_era(text)
        scene_parts = self._extract_scene(text, era)

        parts: list[str] = []

        # 1. 角色 + 场景描述（组装为自然语言句子）
        if character_prompt:
            parts.append(character_prompt)
        if scene_parts:
            parts.append(", ".join(scene_parts))

        # 组装主体描述句
        if parts:
            subject_sentence = ". ".join(p.rstrip(".") for p in parts if p) + "."
        else:
            subject_sentence = "A cinematic scene."

        # 2. 运镜描述
        camera = self._select_camera_movement(text)

        # 3. 视频风格和约束
        video_style = self._preset.get("video_style", "cinematic quality, 4K")
        video_constraints = self._preset.get(
            "video_constraints",
            "stable character appearance, natural smooth movements, no distortion",
        )

        # 拼装完整视频 prompt
        prompt = f"{subject_sentence} {camera}. {video_style}, {video_constraints}."
        return prompt

    @staticmethod
    def _select_camera_movement(text: str) -> str:
        """根据文本情绪/场景自动选择运镜描述。"""
        for pattern, camera_desc in _CAMERA_MOVEMENT_RULES:
            if re.search(pattern, text):
                return camera_desc
        return _DEFAULT_CAMERA

    def _apply_video_style(self, raw_prompt: str) -> str:
        """将视频风格预设关键词附加到 prompt 上。"""
        prompt = raw_prompt.rstrip(". ")

        video_style = self._preset.get("video_style", "")
        video_constraints = self._preset.get("video_constraints", "")

        suffix_parts: list[str] = []
        if video_style:
            suffix_parts.append(video_style)
        if video_constraints:
            suffix_parts.append(video_constraints)

        if suffix_parts:
            prompt += ". " + ", ".join(suffix_parts) + "."

        return prompt

    # ------------------------------------------------------------------
    # era detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_era(text: str) -> str:
        """自动判断文本是现代还是古风。"""
        modern_count = len(re.findall(_MODERN_KEYWORDS, text))
        classical_count = len(re.findall(_CLASSICAL_KEYWORDS, text))
        if modern_count > classical_count:
            return "modern"
        if classical_count > modern_count:
            return "classical"
        # 默认：如果有"他/她"但没有古风特征词，倾向现代
        if re.search(r"手机|电梯|公寓|合租|外卖|办公", text):
            return "modern"
        return "classical"

    def _get_era(self, text: str) -> str:
        """获取时代，优先用缓存的全文判断结果。"""
        if self._era_cache:
            return self._era_cache
        return self._detect_era(text)

    # ------------------------------------------------------------------
    # LLM mode
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_llm_available(llm_cfg: dict) -> bool:
        """检测是否有可用的 LLM provider。"""
        from src.llm import is_llm_available

        return is_llm_available(llm_cfg)

    def _get_llm_client(self):
        """创建或返回缓存的 LLM 客户端实例。"""
        if self._llm_client_cached is None:
            from src.llm import create_llm_client

            self._llm_client_cached = create_llm_client(self._llm_config)
        return self._llm_client_cached

    def _generate_with_llm(
        self,
        text: str,
        character_prompt: str,
        narrator_instruction: str = "",
        *,
        prev_text: str | None = None,
    ) -> str:
        """使用 LLM 生成高质量 prompt。"""
        user_msg = f"小说文本:\n{text}"
        if character_prompt:
            user_msg += f"\n\n已知角色描述（必须严格保持一致，尤其是性别和外观）:\n{character_prompt}"
        user_msg = self._append_narrator_instruction(user_msg, narrator_instruction)
        if not self._comfyui_imagegen:
            user_msg += f"\n\n画面风格: {self._style_name}"
        user_msg = self._append_llm_context_notes(user_msg)
        user_msg = self._append_peaceful_llm_note(user_msg, text, prev_text)
        user_msg += "\n\n重要: 仔细分析文中每个角色的性别（他=male, 她=female），确保 prompt 中角色性别正确，不要搞混。"
        if (
            not self._comfyui_imagegen
            and self._horror_style != "off"
            and self._tone != "light"
            and self._is_horror_segment(text)
        ):
            user_msg += _HORROR_LLM_USER_NOTE

        try:
            client = self._get_llm_client()
            response = client.chat(
                messages=[
                    {"role": "system", "content": self._image_system_prompt()},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
            )
        except Exception as e:
            log.error("LLM prompt 生成失败: %s", e)
            raise PromptGenerationError(f"LLM prompt 生成失败: {e}") from e

        raw_prompt = (response.content or "").strip()
        if not raw_prompt:
            raise PromptGenerationError("LLM 返回空 prompt")

        return self._apply_style(raw_prompt)

    # ------------------------------------------------------------------
    # local fallback mode
    # ------------------------------------------------------------------

    def _generate_local(
        self,
        text: str,
        character_prompt: str,
        narrator_instruction: str = "",
    ) -> str:
        """使用场景规则匹配生成 prompt（无 API 依赖）。"""
        era = self._get_era(text)
        scene_parts = self._extract_scene(text, era)

        parts: list[str] = []

        if not self._comfyui_imagegen:
            prefix = self._preset.get("prefix", "")
            if prefix:
                parts.append(prefix)

        if scene_parts:
            parts.append(", ".join(scene_parts))

        if character_prompt:
            parts.append(character_prompt)

        if not self._comfyui_imagegen:
            quality = (
                "soft natural lighting, clear composition, balanced colors"
                if self._tone == "light"
                else "highly detailed, cinematic composition, dramatic lighting"
            )
            parts.append(quality)

            positive = self._preset.get("positive", "")
            if positive:
                parts.append(positive)

        if era == CLASSICAL:
            parts.append(
                "ancient China, traditional Chinese architecture, "
                "historical setting, traditional Chinese costume"
            )

        prompt = ", ".join(parts)
        return prompt

    def _extract_scene(self, text: str, era: str) -> list[str]:
        """从中文文本中提取场景描述，根据时代选择规则集。"""
        found: list[str] = []
        seen: set[str] = set()

        gender = self._detect_gender(text)

        # 选择对应时代的规则
        rules = _MODERN_RULES if era == "modern" else _CLASSICAL_RULES

        for pattern, description in rules:
            if re.search(pattern, text) and description not in seen:
                desc = self._apply_gender(description, gender, era)
                if self._tone == "light":
                    desc = _TONE_LIGHT_SCENE_OVERRIDES.get(desc, desc)
                found.append(desc)
                seen.add(description)

        # 如果没匹配到人物，补一个默认人物
        has_person = any(
            kw in desc for desc in found
            for kw in ("person", "figure", "man", "woman", "boy", "girl",
                       "swordsman", "scholar", "general", "elderly",
                       "delivery", "student")
        )
        if not has_person and re.search(r"[他她]|人|者", text):
            if era == "modern":
                if gender == "female":
                    found.insert(0, "a young woman in modern casual clothes, black hair")
                else:
                    found.insert(0, "a young man in modern casual clothes, black hair")
            else:
                if gender == "female":
                    found.insert(0, "a beautiful young woman in traditional Chinese hanfu")
                else:
                    found.insert(0, "a handsome young man in traditional Chinese robes")

        return found[:10]

    @staticmethod
    def _detect_gender(text: str) -> str:
        """从文本中推断主要人物的性别。"""
        female_cues = len(re.findall(r"她|少女|女子|姑娘|小姐|美人|夫人|娘|妹|女孩", text))
        male_cues = len(re.findall(r"他|少年|少侠|公子|大侠|将军|书生|兄|爷|男", text))
        if female_cues > male_cues:
            return "female"
        return "male"

    @staticmethod
    def _apply_gender(description: str, gender: str, era: str) -> str:
        """将描述中的通用人称替换为具体性别。"""
        if era == "modern":
            if gender == "female":
                description = description.replace("a person walking", "a young woman walking")
                description = description.replace("a person standing still", "a young woman standing")
                description = description.replace("a person sitting", "a young woman sitting")
                description = description.replace("a person running", "a young woman running")
                description = description.replace("a delivery person", "a young woman delivery worker")
            else:
                description = description.replace("a person walking", "a young man walking")
                description = description.replace("a person standing still", "a young man standing")
                description = description.replace("a person sitting", "a young man sitting")
                description = description.replace("a person running", "a young man running")
        else:
            if gender == "female":
                description = description.replace("a person walking", "a beautiful woman walking gracefully")
                description = description.replace("a person standing still", "a beautiful woman standing elegantly")
                description = description.replace("a person sitting", "a beautiful woman sitting gracefully")
                description = description.replace("a person running", "a woman running")
                description = description.replace("a swordsman", "a beautiful swordswoman")
                description = description.replace("a young man, handsome", "a beautiful young woman")
                description = description.replace("a cloaked figure, flowing cloak", "a beautiful woman in flowing cloak")
            else:
                description = description.replace("a person walking", "a handsome man walking")
                description = description.replace("a person standing still", "a man standing")
                description = description.replace("a person sitting", "a man sitting")
                description = description.replace("a person running", "a man running")
                description = description.replace("a cloaked figure, flowing cloak", "a man in flowing cloak, broad shoulders")
        return description

    # ------------------------------------------------------------------
    # style helpers
    # ------------------------------------------------------------------

    def _apply_style(self, raw_prompt: str) -> str:
        """将风格预设关键词附加到 prompt 上（前缀 + 净化 + 正向后缀）。"""
        cleaned = raw_prompt.rstrip(", ").strip()
        if self._style_name == "anime":
            cleaned = _PHOTOREALISTIC_TERMS_RE.sub("", cleaned)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")

        if self._comfyui_imagegen:
            return self._strip_style_boilerplate(cleaned)

        parts: list[str] = []
        prefix = self._preset.get("prefix", "")
        if prefix and prefix.lower() not in cleaned.lower():
            parts.append(prefix)
        if cleaned:
            parts.append(cleaned)

        base = ", ".join(parts)
        positive = self._preset.get("positive", "")
        if positive:
            positive = (
                self._positive_only_keywords(positive)
                if self._comfyui_imagegen
                else positive
            )
            if positive:
                base = self._append_missing_phrases(base, positive)

        return self._compact_prompt_phrases(base)
