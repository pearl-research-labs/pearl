#!/usr/bin/env python3
"""生成应用图标素材（Pillow 绘制珍珠 logo）。

输出（assets/）：
  icon.png         1024x1024  应用图标（iOS 会自动套圆角，此处为直角底图）
  adaptive-icon.png 1024x1024 Android 自适应图标前景（含安全边距）
  splash-icon.png   512x512   启动页居中小图（透明底）
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "assets"
OUT.mkdir(exist_ok=True)

BG_TOP = (16, 20, 48)
BG_BOTTOM = (46, 46, 138)
PEARL_TOP = (255, 248, 238)
PEARL_BOTTOM = (225, 206, 218)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gradient_bg(size, rounded_radius=None):
    img = Image.new("RGBA", (size, size))
    d = ImageDraw.Draw(img)
    for y in range(size):
        d.line([(0, y), (size, y)], fill=lerp(BG_TOP, BG_BOTTOM, y / size) + (255,))
    if rounded_radius is not None:
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, size - 1, size - 1], radius=rounded_radius, fill=255
        )
        img.putalpha(mask)
    return img


def draw_pearl(size, pearl_scale=0.60, with_shadow=True):
    """在透明画布上绘制珍珠（径向渐变 + 高光 + 投影）。"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cx, cy = size * 0.5, size * 0.46
    r = size * pearl_scale / 2

    # 投影
    if with_shadow:
        shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.ellipse(
            [cx - r * 0.9, cy + r * 0.98, cx + r * 0.9, cy + r * 1.3],
            fill=(4, 6, 16, 120),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(size * 0.02))
        img = Image.alpha_composite(img, shadow)

    pearl = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pd = ImageDraw.Draw(pearl)
    steps = 64
    for i in range(steps, 0, -1):
        t = i / steps  # 1→0 由外向内
        rr = r * t
        # 体色：顶部偏白、底部偏粉紫
        v = 1 - t
        col = lerp(PEARL_BOTTOM, PEARL_TOP, v)
        pd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col + (255,))

    # 高光
    hx, hy = cx - r * 0.38, cy - r * 0.42
    for i in range(40, 0, -1):
        t = i / 40
        alpha = int(160 * (1 - t) ** 2)
        hr = r * 0.34 * t
        pd.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(255, 255, 255, alpha))

    # 抗锯齿边缘
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(size / 512))
    pearl.putalpha(Image.composite(pearl.getchannel("A"), Image.new("L", (size, size), 0), mask))

    return Image.alpha_composite(img, pearl)


def main():
    # 应用图标：圆角底 + 珍珠（iOS 上传用 1024 直角，系统自行裁圆角；
    # 本地显示用圆角版本观感一致）
    size = 1024
    icon = gradient_bg(size)
    icon = Image.alpha_composite(icon, draw_pearl(size))
    icon.convert("RGB").save(OUT / "icon.png")

    # Android 自适应图标前景：珍珠居中，四周留白（安全区 66%）
    adaptive = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pearl = draw_pearl(size, pearl_scale=0.42, with_shadow=False)
    adaptive = Image.alpha_composite(adaptive, pearl)
    adaptive.save(OUT / "adaptive-icon.png")

    # 启动页小图（透明底）
    splash = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    splash = Image.alpha_composite(splash, draw_pearl(512, with_shadow=False))
    splash.save(OUT / "splash-icon.png")

    print(f"已生成: {OUT}")


if __name__ == "__main__":
    main()
