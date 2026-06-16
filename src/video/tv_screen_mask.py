"""CRT 电视屏幕蒙版：生成、加载与校准预览。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter

ContentFit = Literal["fill", "contain"]

EdgeBow = dict[str, float]


def _parse_edge_bow(raw: object) -> EdgeBow:
    """解析四边外凸弧度（ref 像素，正数=该边在中点向外鼓出）。"""
    defaults = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        raise ValueError("tv_screen_edge_bow 必须为 {top, bottom, left, right}")
    out = dict(defaults)
    for key in defaults:
        if key in raw:
            out[key] = max(0.0, float(raw[key]))
    return out


def _parabolic_convex_bow(t: float, bow: float) -> float:
    """t∈[0,1]：端点内收 bow、中点贴边，模拟 CRT 玻璃向外凸（∪ 形边）。"""
    if bow <= 0:
        return 0.0
    return bow * (1.0 - 4.0 * t * (1.0 - t))


def generate_crt_screen_mask(
    width: int,
    height: int,
    *,
    corner_radius: float,
    edge_bow: EdgeBow | None = None,
    feather: float = 0.0,
) -> Image.Image:
    """生成 CRT 屏幕蒙版：圆角 + 四边在中点外凸（模拟曲面玻璃）。"""
    if width < 1 or height < 1:
        raise ValueError(f"蒙版尺寸无效: {width}x{height}")
    bows = _parse_edge_bow(edge_bow or {})
    radius = min(max(corner_radius, 0.0), width / 2, height / 2)

    mask = Image.new("L", (width, height), 0)
    px = mask.load()
    w1 = max(width - 1, 1)
    h1 = max(height - 1, 1)

    for y in range(height):
        ty = y / h1
        left = _parabolic_convex_bow(ty, bows["left"])
        right = w1 - _parabolic_convex_bow(ty, bows["right"])
        for x in range(width):
            tx = x / w1
            top = _parabolic_convex_bow(tx, bows["top"])
            bottom = h1 - _parabolic_convex_bow(tx, bows["bottom"])
            if (
                x < math.floor(left + 1e-6)
                or x > math.floor(right + 1e-6)
                or y < math.floor(top + 1e-6)
                or y > math.floor(bottom + 1e-6)
            ):
                continue
            if radius > 0:
                cx = min(x, w1 - x)
                cy = min(y, h1 - y)
                if cx < radius and cy < radius:
                    dx = radius - cx
                    dy = radius - cy
                    if dx * dx + dy * dy > radius * radius:
                        continue
            px[x, y] = 255

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask


def generate_rounded_rect_mask(
    width: int,
    height: int,
    *,
    corner_radius: float,
    feather: float = 0.0,
    edge_bow: EdgeBow | None = None,
) -> Image.Image:
    """生成屏幕区域灰度蒙版（255=可见，0=透明）。"""
    if any(_parse_edge_bow(edge_bow or {}).values()):
        return generate_crt_screen_mask(
            width,
            height,
            corner_radius=corner_radius,
            edge_bow=edge_bow,
            feather=feather,
        )
    if width < 1 or height < 1:
        raise ValueError(f"蒙版尺寸无效: {width}x{height}")
    radius = min(max(corner_radius, 0.0), width / 2, height / 2)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask


def save_screen_mask(
    output_path: Path,
    width: int,
    height: int,
    *,
    corner_radius: float,
    feather: float = 0.0,
    edge_bow: EdgeBow | None = None,
) -> Path:
    """写入屏幕尺寸蒙版 PNG。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = generate_rounded_rect_mask(
        width,
        height,
        corner_radius=corner_radius,
        feather=feather,
        edge_bow=edge_bow,
    )
    mask.save(output_path)
    return output_path


def load_screen_mask(mask_path: Path) -> Image.Image:
    """加载蒙版并转为灰度。"""
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"屏幕蒙版不存在: {mask_path}")
    with Image.open(mask_path) as img:
        return img.convert("L")


def ensure_screen_mask(
    mask_path: Path | None,
    *,
    ref_width: int,
    ref_height: int,
    corner_radius: float,
    feather: float,
    edge_bow: EdgeBow | None = None,
    auto_path: Path | None = None,
) -> Path:
    """返回可用蒙版路径；缺失时按圆角/弧线参数自动生成。"""
    if mask_path is not None:
        mask_path = Path(mask_path)
        if mask_path.exists():
            return mask_path

    target = auto_path or mask_path
    if target is None:
        raise ValueError("未配置 tv_screen_mask 且未提供 auto_path")
    return save_screen_mask(
        target,
        ref_width,
        ref_height,
        corner_radius=corner_radius,
        feather=feather,
        edge_bow=edge_bow,
    )


def render_calibration_preview(
    tv_frame_image: Path,
    *,
    screen_x: int,
    screen_y: int,
    mask_path: Path,
    output_path: Path,
    tint: tuple[int, int, int, int] = (0, 255, 120, 140),
) -> Path:
    """在 tv-frame 上叠加半透明蒙版，用于校准屏幕区域。"""
    tv_frame_image = Path(tv_frame_image)
    mask_path = Path(mask_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(tv_frame_image) as frame:
        base = frame.convert("RGBA")
    mask = load_screen_mask(mask_path)
    overlay = Image.new("RGBA", mask.size, tint)
    overlay.putalpha(mask)
    base.paste(overlay, (screen_x, screen_y), overlay)
    base.convert("RGB").save(output_path)
    return output_path
