"""
Generates docs/assets/banner.png — the README hero banner.

Run any time:
    python docs/make_banner.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT  = Path(__file__).parent / "assets" / "banner.png"
W, H = 1280, 320

# ── palette ───────────────────────────────────────────────────────────────────
BG_L  = (5,   11,  24)    # #050b18  very dark navy (left)
BG_R  = (12,  22,  46)    # #0c162e  slightly lighter navy (right)
CYAN1 = (56,  189, 248)   # #38bdf8  sky-400
CYAN2 = (186, 230, 253)   # #bae6fd  sky-200  (gradient end for title)
AMBER = (245, 158,  11)   # #f59e0b
WHITE = (255, 255, 255)
TEXT2 = (203, 213, 225)   # #cbd5e1  slate-300  tagline
MUTED = (71,   85, 105)   # #475569  slate-600  agency line
RING  = (30,   58,  95)   # #1e3a5f  faint rings

_BOLD   = "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
_MEDIUM = "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf"
_REG    = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


# ── background: diagonal gradient ────────────────────────────────────────────
def _gradient_bg() -> Image.Image:
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for x in range(W):
        t = x / (W - 1)
        arr[:, x, 0] = int(BG_L[0] + (BG_R[0] - BG_L[0]) * t)
        arr[:, x, 1] = int(BG_L[1] + (BG_R[1] - BG_L[1]) * t)
        arr[:, x, 2] = int(BG_L[2] + (BG_R[2] - BG_L[2]) * t)
    # vertical darkening toward top and bottom edges
    for y in range(H):
        vt = 1.0 - 0.18 * (1 - math.sin(math.pi * y / H)) ** 2
        arr[y] = np.clip(arr[y] * vt, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── radar rings (right side decoration) ───────────────────────────────────────
def _radar_rings(draw: ImageDraw.ImageDraw) -> None:
    cx, cy = 1100, H // 2
    for r in range(40, 320, 48):
        alpha = max(0, 55 - r // 6)
        c = tuple(min(255, RING[i] + alpha) for i in range(3))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c, width=1)  # type: ignore[arg-type]


# ── fine horizontal scan lines (very faint) ───────────────────────────────────
def _scan_lines(img: Image.Image) -> Image.Image:
    overlay = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for y in range(0, H, 4):
        d.line([(0, y), (W, y)], fill=(8, 8, 14), width=1)
    return Image.blend(img, overlay, alpha=0.22)


# ── gradient text helper ───────────────────────────────────────────────────────
def _paste_gradient_text(
    base: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    xy: tuple[int, int],
    color_l: tuple[int, int, int],
    color_r: tuple[int, int, int],
) -> None:
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0] + 4
    th = bbox[3] - bbox[1] + 4

    # horizontal gradient slab
    grad = Image.new("RGBA", (tw, th))
    gd   = ImageDraw.Draw(grad)
    for x in range(tw):
        t = x / max(tw - 1, 1)
        r = int(color_l[0] + (color_r[0] - color_l[0]) * t)
        g = int(color_l[1] + (color_r[1] - color_l[1]) * t)
        b = int(color_l[2] + (color_r[2] - color_l[2]) * t)
        gd.line([(x, 0), (x, th)], fill=(r, g, b, 255))

    # text alpha mask
    mask_img = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(mask_img).text(
        (-bbox[0] + 2, -bbox[1] + 2), text, font=font, fill=255
    )

    base.paste(grad, xy, mask_img)


# ── bolt polygon ───────────────────────────────────────────────────────────────
def _bolt(cx: int, cy: int, s: float) -> list[tuple[int, int]]:
    raw = [(9,0),(-3,18),(5,18),(-10,42),(4,42),(-14,62),(20,26),(8,26),(22,0)]
    return [(int(cx + p[0]*s - 4*s), int(cy + p[1]*s - 31*s)) for p in raw]


def main() -> None:
    img  = _gradient_bg()
    img  = _scan_lines(img)
    draw = ImageDraw.Draw(img)

    # ── radar rings ───────────────────────────────────────────────────────────
    _radar_rings(draw)

    # ── bolt glow (blurred amber layer underneath) ────────────────────────────
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    gd.polygon(_bolt(78, H // 2, 2.2), fill=(*AMBER, 180))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=14))
    img.paste(Image.new("RGB", (W, H), (0,0,0)), mask=ImageChops_alpha(glow_layer))

    # simpler glow: just draw a blurred version
    glow2 = Image.new("RGB", (W, H), BG_L)
    gd2   = ImageDraw.Draw(glow2)
    gd2.polygon(_bolt(78, H // 2, 2.2), fill=AMBER)
    glow2 = glow2.filter(ImageFilter.GaussianBlur(radius=18))
    img   = Image.blend(img, glow2, alpha=0.35)
    draw  = ImageDraw.Draw(img)

    # ── bolt sharp ────────────────────────────────────────────────────────────
    draw.polygon(_bolt(78, H // 2, 2.2), fill=AMBER)

    # ── thin vertical separator ────────────────────────────────────────────────
    sep_x = 148
    for y in range(70, H - 70):
        t = (y - 70) / (H - 140)
        alpha = int(255 * math.sin(math.pi * t) * 0.55)
        c = tuple(int(BG_L[i] + (CYAN1[i] - BG_L[i]) * alpha / 255) for i in range(3))
        draw.point((sep_x, y), fill=c)  # type: ignore[arg-type]

    # ── project name (gradient text) ──────────────────────────────────────────
    font_name = _font(_BOLD, 88)
    _paste_gradient_text(img, "climagrid", font_name, (166, 64), CYAN1, CYAN2)
    draw = ImageDraw.Draw(img)

    # ── tagline ───────────────────────────────────────────────────────────────
    font_tag = _font(_MEDIUM, 25)
    draw.text((168, 172), "Climate data, grid-ready", font=font_tag, fill=TEXT2)

    # ── amber accent rule (gradient fade to right) ────────────────────────────
    rule_y = 210
    for x in range(168, 640):
        t = (x - 168) / (640 - 168)
        alpha = int(255 * (1 - t ** 1.6))
        c = tuple(int(BG_L[i] + (AMBER[i] - BG_L[i]) * alpha / 255) for i in range(3))
        draw.line([(x, rule_y), (x, rule_y + 2)], fill=c)  # type: ignore[arg-type]

    # ── agency line ───────────────────────────────────────────────────────────
    font_sm = _font(_REG, 19)
    draw.text((168, 224), "NOAA  ·  NASA  ·  USDA  ·  USFS", font=font_sm, fill=MUTED)

    # ── top border: cyan gradient fade to right ────────────────────────────────
    for x in range(W):
        t = x / (W - 1)
        alpha = int(255 * (1 - t ** 0.6))
        c = tuple(int(BG_L[i] + (CYAN1[i] - BG_L[i]) * alpha / 255) for i in range(3))
        for dy in range(3):
            draw.point((x, dy), fill=c)  # type: ignore[arg-type]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG", optimize=True)
    print(f"Banner saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")


# small shim — avoid importing ImageChops at top level
def ImageChops_alpha(img: Image.Image):  # type: ignore[return]
    return img.split()[3]


if __name__ == "__main__":
    main()
