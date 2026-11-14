"""Render the 90-second Spatial-CSI explainer video.

Frames are drawn procedurally with Pillow (2x supersampled), narration comes from
Windows TTS WAVs in audio/s{i}.wav, and ffmpeg (bundled by imageio-ffmpeg) muxes the MP4.

    python render.py stills   # one PNG per scene into stills/ for review
    python render.py          # full render -> spatial_csi_90s.mp4
"""
import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).parent
W, H, S, FPS = 1280, 720, 2, 30
TARGET = 89.7                                 # total video length (s)
LEADS = [2.6, 0.9, 0.9, 0.9, 0.9, 0.9]        # music before each narration line (s)
TAIL_W = [0.8, 1.0, 1.1, 1.0, 1.5, 1.6]       # how the remaining time is shared out
ALERT_FRAC = 0.20                              # when the emergency fires in the dashboard scene
TITLE_FRAC = 0.42                              # title reveal in the closing scene (on "Spatial-CSI")

# palette
BG = (6, 10, 20)
PANEL = (14, 22, 40)
PANEL_HI = (22, 34, 60)
EDGE = (48, 66, 100)
CYAN = (79, 209, 232)
AMBER = (245, 166, 35)
RED = (235, 64, 52)
GREEN = (72, 199, 116)
WHITE = (232, 238, 246)
MUTED = (128, 143, 168)
SKIN = (224, 188, 152)

FONTS = {
    "ui": "C:/Windows/Fonts/segoeui.ttf",
    "uib": "C:/Windows/Fonts/segoeuib.ttf",
    "head": "C:/Windows/Fonts/bahnschrift.ttf",
    "mono": "C:/Windows/Fonts/consola.ttf",
}
_font_cache = {}


def font(name, size):
    key = (name, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONTS[name], int(size * S))
    return _font_cache[key]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def lerp(c1, c2, a):
    a = clamp(a)
    return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))


def P(*v):
    return [int(round(x * S)) for x in v]


class Canvas:
    def __init__(self, img):
        self.img = img
        self.d = ImageDraw.Draw(img)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, w=1):
        self.d.rectangle(P(x0, y0, x1, y1), fill=fill, outline=outline, width=max(1, int(w * S)))

    def rrect(self, x0, y0, x1, y1, r, fill=None, outline=None, w=1):
        self.d.rounded_rectangle(P(x0, y0, x1, y1), radius=int(r * S), fill=fill,
                                 outline=outline, width=max(1, int(w * S)))

    def ell(self, cx, cy, rx, ry=None, fill=None, outline=None, w=1):
        ry = rx if ry is None else ry
        self.d.ellipse(P(cx - rx, cy - ry, cx + rx, cy + ry), fill=fill, outline=outline,
                       width=max(1, int(w * S)))

    def line(self, pts, fill, w=1):
        self.d.line([tuple(P(x, y)) for x, y in pts], fill=fill, width=max(1, int(w * S)), joint="curve")

    def poly(self, pts, fill=None, outline=None):
        self.d.polygon([tuple(P(x, y)) for x, y in pts], fill=fill, outline=outline)

    def arc(self, cx, cy, r, a0, a1, fill, w=1):
        self.d.arc(P(cx - r, cy - r, cx + r, cy + r), a0, a1, fill=fill, width=max(1, int(w * S)))

    def text(self, x, y, s, size, fill, name="ui", anchor="la"):
        self.d.text(tuple(P(x, y)), s, font=font(name, size), fill=fill, anchor=anchor)

    def text_w(self, s, size, name="ui"):
        return self.d.textlength(s, font=font(name, size)) / S


# ----------------------------------------------------------------- backgrounds
def make_starfield():
    rng = np.random.default_rng(7)
    img = Image.new("RGB", (W * S, H * S), BG)
    c = Canvas(img)
    for _ in range(420):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        b = rng.uniform(0.25, 1.0) ** 2
        r = rng.choice([0.5, 0.7, 1.0, 1.3], p=[0.5, 0.3, 0.15, 0.05])
        col = lerp(BG, (200, 215, 255) if rng.random() < 0.3 else WHITE, b)
        c.ell(x, y, r, fill=col)
    return img


def make_earth(cx, cy, r):
    """RGBA full-frame layer with a lit Earth limb and atmosphere glow."""
    layer = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse(P(cx - r - 14, cy - r - 14, cx + r + 14, cy + r + 14), fill=(70, 150, 255, 150))
    glow = glow.filter(ImageFilter.GaussianBlur(22 * S))
    layer = Image.alpha_composite(layer, glow)
    # vertical gradient disk
    grad = np.zeros((H * S, W * S, 4), np.uint8)
    ys = np.arange(H * S)[:, None] / S
    a = np.clip((ys - (cy - r)) / 260.0, 0, 1)
    top, bot = np.array([70, 150, 225]), np.array([8, 22, 50])
    col = (top[None, :] * (1 - a) + bot[None, :] * a).astype(np.uint8)  # (H*S, 3)
    grad[..., :3] = col[:, None, :]
    grad[..., 3] = 255
    disk = Image.fromarray(grad, "RGBA")
    mask = Image.new("L", layer.size, 0)
    ImageDraw.Draw(mask).ellipse(P(cx - r, cy - r, cx + r, cy + r), fill=255)
    # clouds
    clouds = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    cd = ImageDraw.Draw(clouds)
    rng = np.random.default_rng(3)
    for _ in range(40):
        x = rng.uniform(cx - r, cx + r)
        y = rng.uniform(cy - r, cy - r + 220)
        cd.ellipse(P(x - rng.uniform(40, 120), y - 6, x + rng.uniform(40, 120), y + 6), fill=(230, 240, 255, 45))
    clouds = clouds.filter(ImageFilter.GaussianBlur(4 * S))
    disk = Image.alpha_composite(disk, clouds)
    layer.paste(disk, (0, 0), mask)
    return layer


def make_interior():
    img = Image.new("RGB", (W * S, H * S), BG)
    arr = np.zeros((H * S, W * S, 3), np.float32)
    ys = np.linspace(0, 1, H * S)[:, None, None]
    arr[:] = np.array([16, 24, 42]) * (1 - ys) + np.array([7, 11, 21]) * ys
    img = Image.fromarray(arr.astype(np.uint8), "RGB")
    c = Canvas(img)
    # subtle panel grid (station wall)
    for x in range(0, W, 160):
        c.line([(x, 0), (x, H)], (20, 30, 52), 1)
    for y in range(0, H, 120):
        c.line([(0, y), (W, y)], (20, 30, 52), 1)
    return img


# ------------------------------------------------------------------- props
def draw_station(c, cx, cy, k, t):
    gold, gold_d = (176, 128, 52), (120, 84, 30)
    c.rrect(cx - 270 * k, cy - 5 * k, cx + 270 * k, cy + 5 * k, 2 * k, fill=(165, 170, 182))
    for x in (-235, -160, 160, 235):
        for y0, y1 in ((-120, -12), (12, 120)):
            c.rect(cx + (x - 26) * k, cy + y0 * k, cx + (x + 26) * k, cy + y1 * k, fill=gold, outline=gold_d)
            for gy in range(int(y0) + 18, int(y1), 18):
                c.line([(cx + (x - 26) * k, cy + gy * k), (cx + (x + 26) * k, cy + gy * k)], gold_d, 1)
            c.line([(cx + x * k, cy + y0 * k), (cx + x * k, cy + y1 * k)], gold_d, 1)
    for x in (-95, 95):
        c.rect(cx + (x - 12) * k, cy + 8 * k, cx + (x + 12) * k, cy + 60 * k, fill=(220, 225, 232))
    c.rrect(cx - 80 * k, cy - 16 * k, cx + 80 * k, cy + 16 * k, 14 * k, fill=(205, 210, 220), outline=(150, 155, 168))
    c.rrect(cx - 14 * k, cy - 52 * k, cx + 14 * k, cy + 44 * k, 12 * k, fill=(195, 200, 212), outline=(150, 155, 168))
    blink = (math.sin(t * 6) > 0.6)
    c.ell(cx - 270 * k, cy, 3 * k, fill=RED if blink else (90, 30, 30))
    c.ell(cx + 270 * k, cy, 3 * k, fill=GREEN if not blink else (30, 80, 40))


def draw_sleeper(c, cx, cy, k, t, zzz=True, color=(70, 92, 140)):
    b = 0.5 + 0.5 * math.sin(2 * math.pi * 0.25 * t)
    wdt = 36 * k * (1 + 0.05 * b)
    c.rrect(cx - wdt, cy - 58 * k, cx + wdt, cy + 125 * k, 32 * k, fill=color, outline=lerp(color, WHITE, 0.25), w=2)
    for yy in (10, 70):
        c.line([(cx - wdt + 4, cy + yy * k), (cx + wdt - 4, cy + yy * k)], lerp(color, BG, 0.4), 3 * k)
    c.ell(cx, cy - 86 * k, 27 * k, fill=SKIN)
    c.d.pieslice(P(cx - 28 * k, cy - 115 * k, cx + 28 * k, cy - 62 * k), 180, 360, fill=(58, 44, 38))
    for ex in (-10, 10):
        c.arc(cx + ex * k, cy - 84 * k, 5 * k, 20, 160, (90, 60, 50), 2)
    if zzz:
        for i in range(3):
            ph = (t * 0.45 + i / 3) % 1
            col = lerp(BG, (170, 190, 230), math.sin(ph * math.pi))
            c.text(cx + (40 + ph * 40) * k, cy - (110 + ph * 70) * k, "z", int((14 + ph * 10) * k), col, "uib")


def draw_pod(c, x0, y0, x1, y1, label):
    c.rrect(x0, y0, x1, y1, 26, fill=(20, 30, 52), outline=EDGE, w=2)
    c.rrect(x0 + 14, y0 + 14, x1 - 14, y1 - 14, 18, fill=(26, 38, 64))
    c.rrect(x0 + 40, y0 + 22, x1 - 40, y0 + 28, 3, fill=(120, 160, 200))
    c.text((x0 + x1) / 2, y1 - 30, label, 13, MUTED, "head", "mm")


def draw_router(c, x, y, t):
    c.rrect(x - 26, y - 12, x + 26, y + 12, 5, fill=(40, 52, 78), outline=(90, 110, 145), w=1.5)
    c.line([(x - 14, y - 12), (x - 20, y - 38)], (150, 165, 190), 3)
    c.line([(x + 14, y - 12), (x + 20, y - 38)], (150, 165, 190), 3)
    c.ell(x - 12, y, 2.5, fill=GREEN if math.sin(t * 9) > 0 else (30, 80, 40))
    c.ell(x, y, 2.5, fill=CYAN)


def x_mark(c, cx, cy, r, prog):
    if prog <= 0:
        return
    p1 = clamp(prog * 2)
    p2 = clamp(prog * 2 - 1)
    c.line([(cx - r, cy - r), (cx - r + 2 * r * p1, cy - r + 2 * r * p1)], RED, 9)
    if p2 > 0:
        c.line([(cx + r, cy - r), (cx + r - 2 * r * p2, cy - r + 2 * r * p2)], RED, 9)


def wave_line(c, x0, x1, ymid, amp, t, col, w=2.5, f=0.27, noise=0.12, speed=1.0, glitch=None, win=11.0):
    pts = []
    n = int(x1 - x0) // 3
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        tau = t * speed - win * (x1 - x) / (x1 - x0)
        v = 0.75 * math.sin(2 * math.pi * f * tau) + noise * math.sin(2 * math.pi * 1.9 * tau + 1.3) \
            + noise * 0.6 * math.sin(2 * math.pi * 3.7 * tau)
        if glitch is not None:
            v = glitch(tau, v)
        pts.append((x, ymid - amp * v))
    c.line(pts, col, w)


def card_layer(w, h, fn, alpha, dy=0.0):
    """Draw into an RGBA card-sized layer, return (img, alpha) for pasting."""
    img = Image.new("RGBA", (int(w * S), int(h * S)), (0, 0, 0, 0))
    cc = Canvas(img)
    fn(cc)
    if alpha < 1:
        a = img.getchannel("A").point(lambda v: int(v * alpha))
        img.putalpha(a)
    return img


def paste_layer(base, layer, x, y):
    base.paste(layer, (int(x * S), int(y * S)), layer)


def label_chip(c, x, y, s, col, size=13):
    tw = c.text_w(s, size, "uib")
    c.rrect(x, y, x + tw + 28, y + size + 16, (size + 16) / 2, fill=lerp(BG, col, 0.18), outline=col, w=1.5)
    c.text(x + 14, y + (size + 16) / 2, s, size, col, "uib", "lm")
    return x + tw + 28


# ------------------------------------------------------------------- scenes
STAR = None
EARTH = None
EARTH_BIG = None
INTERIOR = None


def scene_orbit_night(img, t, T):
    split = T * 0.52
    def exterior():
        im = STAR.copy().convert("RGBA")
        e = EARTH
        im.alpha_composite(e, (int(-t * 3 * S), 0))
        im = im.convert("RGB")
        c = Canvas(im)
        draw_station(c, 560 + t * 14, 300 - t * 2, 0.85, t)
        c.text(48, 44, "03:00 GMT", 30, WHITE, "head")
        c.text(48, 84, "LOW EARTH ORBIT  ·  400 KM", 14, MUTED, "mono")
        return im

    def interior():
        im = INTERIOR.copy()
        c = Canvas(im)
        draw_pod(c, 470, 70, 810, 610, "CREW QUARTERS  ·  CM-1")
        draw_sleeper(c, 640, 330, 1.2, t)
        c.text(48, 44, "03:00 GMT", 30, WHITE, "head")
        c.text(48, 84, "CREW SLEEP PERIOD", 14, MUTED, "mono")
        return im

    if t < split - 0.3:
        out = exterior()
    elif t > split + 0.3:
        out = interior()
    else:
        out = Image.blend(exterior(), interior(), ease((t - split + 0.3) / 0.6))
    img.paste(out)


def scene_problem(img, t, T):
    img.paste(INTERIOR)
    c = Canvas(img)
    c.text(640, 70, "AROUND-THE-CLOCK HEALTH MONITORING", 15, CYAN, "mono", "mm")
    c.text(640, 112, "Today's options don't fit life in space", 34, WHITE, "head", "mm")
    # heartbeat strip
    def ecg(tau, v):
        d = tau % 0.9
        return 3.2 * (1 - abs(d - 0.04) / 0.04) if d < 0.08 else v * 0.1
    wave_line(c, 340, 940, 175, 10, t, lerp(BG, CYAN, 0.5), 2, f=1.1, noise=0.0, glitch=ecg, win=4.5)

    def card(cx, title, sub, icon, appear, cross):
        a = ease((t - appear) / 0.5)
        if a <= 0:
            return
        y0 = 230 + (1 - a) * 30

        def body(cc):
            cc.rrect(0, 0, 380, 330, 18, fill=PANEL, outline=EDGE, w=1.5)
            icon(cc, 190, 125)
            cc.text(190, 245, title, 26, WHITE, "head", "mm")
            cc.text(190, 285, sub, 16, MUTED, "ui", "mm")
            x_mark(cc, 190, 125, 70, (t - cross) / 0.6)
        paste_layer(img, card_layer(380, 330, body, a), cx - 190, y0)

    def watch(cc, x, y):
        cc.rrect(x - 26, y - 90, x + 26, y + 90, 14, fill=(44, 56, 84))
        cc.rrect(x - 52, y - 50, x + 52, y + 50, 22, fill=(30, 40, 62), outline=(120, 140, 175), w=3)
        cc.line([(x - 36, y), (x - 14, y), (x - 6, y - 22), (x + 4, y + 20), (x + 12, y), (x + 36, y)], RED, 3)

    def camera(cc, x, y):
        cc.rrect(x - 75, y - 45, x + 55, y + 45, 12, fill=(44, 56, 84), outline=(120, 140, 175), w=3)
        cc.poly([(x + 55, y - 22), (x + 92, y - 42), (x + 92, y + 42), (x + 55, y + 22)], fill=(60, 74, 104))
        cc.ell(x - 10, y, 28, fill=(20, 26, 40), outline=(140, 160, 195), w=3)
        cc.ell(x - 10, y, 12, fill=(60, 110, 160))
        if math.sin(t * 5) > 0:
            cc.ell(x - 58, y - 30, 6, fill=RED)

    card(400, "Wearables", "Uncomfortable to wear 24/7", watch, T * 0.22, T * 0.36)
    card(880, "Cameras", "No privacy in sleeping quarters", camera, T * 0.42, T * 0.56)


def scene_wifi(img, t, T):
    img.paste(INTERIOR)
    c = Canvas(img)
    c.text(48, 44, "THE IDEA", 15, CYAN, "mono")
    c.text(48, 70, "The station's Wi-Fi becomes a sensor", 32, WHITE, "head")
    draw_pod(c, 60, 140, 700, 600, "CREW QUARTERS")
    sx, sy = 470, 360
    rx, ry = 120, 300
    # wifi rings, perturbed where they cross the sleeper
    breath = math.sin(2 * math.pi * 0.25 * t)
    for k in range(9):
        r = (t * 70 + k * 55) % 495
        if r < 8:
            continue
        a = clamp(1 - r / 495) * 0.9
        col = lerp((26, 38, 64), CYAN, a)
        pts = []
        for i in range(61):
            ang = math.radians(-58 + 116 * i / 60)
            px, py = rx + r * math.cos(ang), ry + r * math.sin(ang)
            d2 = ((px - sx) ** 2 / 60 ** 2 + (py - sy) ** 2 / 110 ** 2)
            bump = 9 * math.exp(-d2) * (0.6 + 0.4 * breath) * math.sin(r / 9 + t * 4)
            pts.append((rx + (r + bump) * math.cos(ang), ry + (r + bump) * math.sin(ang)))
        pts = [(x, y) for x, y in pts if 80 < x < 680 and 160 < y < 580]
        if len(pts) > 2:
            c.line(pts, col, 2.5)
    draw_sleeper(c, sx, sy, 1.0, t, zzz=False)
    draw_router(c, rx, ry, t)
    c.text(rx, ry + 34, "Wi-Fi access point", 12, MUTED, "ui", "mm")

    # signal panel
    a = ease((t - T * 0.25) / 0.6)
    if a > 0:
        def panel(cc):
            cc.rrect(0, 0, 500, 300, 18, fill=PANEL, outline=EDGE, w=1.5)
            cc.text(24, 22, "WI-FI SIGNAL (CSI)", 13, MUTED, "mono")
            cc.text(476, 22, "LIVE", 13, GREEN if math.sin(t * 6) > 0 else MUTED, "mono", "ra")
            for gy in (100, 150, 200):
                cc.line([(24, gy), (476, gy)], (26, 38, 62), 1)
            wave_line(cc, 24, 476, 150, 70, t, CYAN, 3)
            hl = ease((t - T * 0.62) / 0.6)
            if hl > 0:
                cc.text(24, 262, "Each rise and fall = one breath", 16, lerp(PANEL, WHITE, hl), "ui", "lm")
        paste_layer(img, card_layer(500, 300, panel, a), 740, 170)
    b = ease((t - T * 0.82) / 0.5)
    if b > 0:
        def chips(cc):
            x = label_chip(cc, 0, 0, "CONTACTLESS", CYAN)
            x = label_chip(cc, x + 10, 0, "NO CAMERA", CYAN)
            label_chip(cc, x + 10, 0, "NO WEARABLE", CYAN)
        paste_layer(img, card_layer(500, 40, chips, b), 740, 500)


def scene_capabilities(img, t, T):
    img.paste(INTERIOR)
    c = Canvas(img)
    c.text(640, 70, "WHAT THE AI SEES", 15, CYAN, "mono", "mm")
    c.text(640, 112, "Three signals, one picture of crew health", 34, WHITE, "head", "mm")

    def breathing(cc):
        x, y = 170, 95
        cc.line([(x, y - 48), (x, y - 8)], CYAN, 5)
        cc.line([(x, y - 8), (x - 18, y + 4)], CYAN, 4)
        cc.line([(x, y - 8), (x + 18, y + 4)], CYAN, 4)
        s = 1 + 0.06 * math.sin(2 * math.pi * 0.25 * t)
        cc.ell(x - 36, y + 18, 24 * s, 40 * s, outline=CYAN, w=4)
        cc.ell(x + 36, y + 18, 24 * s, 40 * s, outline=CYAN, w=4)
        cc.text(170, 190, "Breathing", 26, WHITE, "head", "mm")
        wave_line(cc, 40, 300, 245, 18, t, CYAN, 2.5)
        bpm = 14 + (1 if math.sin(t * 0.7) > 0.6 else 0)
        cc.text(170, 300, f"{bpm} breaths / min", 17, MUTED, "uib", "mm")

    acts = ["Sleeping", "Exercising", "Working", "Eating"]

    def activity(cc):
        x, y = 170, 95
        sw = math.sin(t * 5)
        cc.ell(x, y - 46, 14, outline=GREEN, w=4)
        cc.line([(x, y - 30), (x, y + 22)], GREEN, 5)
        cc.line([(x, y - 18), (x - 26 * sw, y + 8)], GREEN, 4)
        cc.line([(x, y - 18), (x + 26 * sw, y + 8)], GREEN, 4)
        cc.line([(x, y + 22), (x - 24 * sw, y + 62)], GREEN, 4)
        cc.line([(x, y + 22), (x + 24 * sw, y + 62)], GREEN, 4)
        cc.text(170, 190, "Daily activity", 26, WHITE, "head", "mm")
        a = acts[int(t / 1.4) % len(acts)]
        label_chip(cc, 170 - (cc.text_w(a.upper(), 14, "uib") + 28) / 2, 230, a.upper(), GREEN, 14)
        cc.text(170, 300, "recognised automatically", 17, MUTED, "uib", "mm")

    def unusual(cc):
        x, y = 170, 100
        pulse = 0.5 + 0.5 * math.sin(t * 5)
        col = lerp(AMBER, (255, 210, 120), pulse)
        cc.poly([(x, y - 58), (x + 62, y + 50), (x - 62, y + 50)], fill=lerp(PANEL, AMBER, 0.15))
        cc.line([(x, y - 58), (x + 62, y + 50), (x - 62, y + 50), (x, y - 58)], col, 5)
        cc.line([(x, y - 22), (x, y + 14)], col, 7)
        cc.ell(x, y + 32, 5, fill=col)
        cc.text(170, 190, "Unusual events", 26, WHITE, "head", "mm")
        xx = label_chip(cc, 50, 230, "FALL", AMBER, 14)
        label_chip(cc, xx + 10, 230, "SUDDEN STILLNESS", AMBER, 14)
        cc.text(170, 300, "flagged in seconds", 17, MUTED, "uib", "mm")

    for i, (fn, at) in enumerate(((breathing, 0.14), (activity, 0.29), (unusual, 0.42))):
        a = ease((t - T * at) / 0.5)
        if a <= 0:
            continue

        def body(cc, fn=fn):
            cc.rrect(0, 0, 340, 340, 18, fill=PANEL, outline=EDGE, w=1.5)
            fn(cc)
        paste_layer(img, card_layer(340, 340, body, a), 80 + i * 380, 180 + (1 - a) * 30)


CREW = [  # id, module index, x, y, activity, bpm
    ("CM-1", 0, 115, 262, "Sleeping", 13),
    ("CM-2", 0, 190, 262, "Sleeping", 12),
    ("CM-4", 1, 335, 262, "Eating", 16),
    ("CM-3", 2, 520, 262, "Working", 17),
]
MODULES = [("CREW QUARTERS", 50, 170, 250, 350), ("NODE 1 · GALLEY", 270, 190, 400, 330),
           ("US LAB", 420, 170, 620, 350), ("COLUMBUS", 640, 190, 750, 330)]
BRIEF = ("CM-3: sudden motion followed by stillness in US Lab at 03:12. Breathing still detected "
         "(12 bpm, high confidence). Recommend immediate crew check. CM-1, CM-2, CM-4 stable.")


def scene_dashboard(img, t, T):
    img.paste(Image.new("RGB", img.size, (8, 13, 24)))
    c = Canvas(img)
    t_alert = T * ALERT_FRAC
    alert = t >= t_alert
    # header
    c.rect(0, 0, W, 56, fill=(12, 20, 36))
    c.line([(0, 56), (W, 56)], EDGE, 1)
    c.text(30, 28, "SPATIAL-CSI", 20, WHITE, "head", "lm")
    c.text(165, 28, "MISSION CONTROL  ·  CREW HEALTH", 13, MUTED, "mono", "lm")
    secs = 30 + int(t)
    c.text(W - 230, 28, f"GMT 03:12:{secs % 60:02d}", 15, WHITE, "mono", "lm")
    if alert:
        label_chip(c, W - 400, 15, "1 EMERGENCY", RED, 12)
    else:
        label_chip(c, W - 400, 15, "4 CREW · NOMINAL", GREEN, 12)

    # map panel
    c.rrect(30, 76, 770, 440, 14, fill=PANEL, outline=EDGE, w=1.5)
    c.text(50, 98, "STATION MAP", 12, MUTED, "mono", "lm")
    for (name, x0, y0, x1, y1) in MODULES:
        hot = alert and name == "US LAB"
        c.rrect(x0, y0, x1, y1, 16, fill=lerp((20, 32, 56), RED, 0.12 if hot else 0), outline=RED if hot else EDGE, w=2)
        c.text((x0 + x1) / 2, y0 + 20, name, 11, RED if hot else MUTED, "mono", "mm")
    for xa, xb in ((250, 270), (400, 420), (620, 640)):
        c.rect(xa, 245, xb, 275, fill=(30, 44, 72))
    for (cid, mi, x, y, act, bpm) in CREW:
        bad = alert and cid == "CM-3"
        col = RED if bad else GREEN
        ph = (t * (1.4 if bad else 0.5)) % 1
        c.ell(x, y, 10 + ph * (30 if bad else 14), outline=lerp((20, 32, 56), col, 1 - ph), w=2)
        c.ell(x, y, 9, fill=col)
        c.text(x, y + 26, cid, 12, WHITE, "uib", "mm")
    c.text(50, 418, "Simulated scenario · signals replayed from public Wi-Fi sensing datasets", 11, MUTED, "ui", "lm")

    # crew cards
    for i, (cid, mi, x, y, act, bpm) in enumerate(CREW):
        bad = alert and cid == "CM-3"
        y0 = 76 + i * 92
        c.rrect(800, y0, 1250, y0 + 82, 12, fill=lerp(PANEL, RED, 0.12) if bad else PANEL, outline=RED if bad else EDGE, w=2 if bad else 1.5)
        c.ell(824, y0 + 24, 6, fill=RED if bad else GREEN)
        c.text(840, y0 + 24, cid, 17, WHITE, "head", "lm")
        c.text(900, y0 + 24, ["Crew Quarters", "Node 1 · Galley", "US Lab", "Columbus"][mi], 12, MUTED, "ui", "lm")
        c.text(824, y0 + 56, "Possible fall" if bad else act, 15, RED if bad else WHITE, "uib", "lm")
        b = 12 if bad else bpm
        c.text(1230, y0 + 24, f"{b} bpm", 17, WHITE, "mono", "rm")
        wave_line(c, 1000, 1230, y0 + 58, 9, t + i, RED if bad else CYAN, 2,
                  f=0.2 if bad else 0.22 + i * 0.02, noise=0.08)

    # alert banner
    if alert:
        a = ease((t - t_alert) / 0.4)
        flash = 0.5 + 0.5 * math.sin((t - t_alert) * 7)
        x1 = 30 + 1220 * a
        c.rrect(30, 456, max(31, x1), 504, 10, fill=lerp((90, 16, 16), (140, 24, 20), flash), outline=RED, w=2)
        if a > 0.95:
            c.text(54, 480, "EMERGENCY", 17, WHITE, "head", "lm")
            c.text(170, 480, f"Possible fall detected  ·  CM-3  ·  US Lab  ·  03:12:{(30 + int(t_alert)) % 60:02d}",
                   15, WHITE, "uib", "lm")
            c.text(1226, 480, "ACK", 14, WHITE, "mono", "rm")
    # AI brief
    tb = T * 0.45
    if t >= tb:
        a = ease((t - tb) / 0.4)
        c.rrect(30, 518, 1250, 612, 14, fill=lerp((8, 13, 24), PANEL_HI, a), outline=lerp((8, 13, 24), CYAN, a), w=1.5)
        c.text(54, 540, "AI CREW BRIEF", 12, CYAN, "mono", "lm")
        n = int(clamp((t - tb - 0.3) / (T * 0.38)) * len(BRIEF))
        words = BRIEF[:n]
        # wrap into two lines
        line1, line2 = words, ""
        if c.text_w(words, 16) > 1160:
            cut = words.rfind(" ", 0, int(len(words) * 1160 / c.text_w(words, 16)))
            line1, line2 = words[:cut], words[cut + 1:]
        c.text(54, 566, line1, 16, WHITE, "ui", "lm")
        c.text(54, 592, line2, 16, WHITE, "ui", "lm")


def scene_close(img, t, T):
    im = STAR.copy().convert("RGBA")
    im.alpha_composite(EARTH_BIG, (0, int((1 - ease(t / T)) * 40 * S)))
    img.paste(im.convert("RGB"))
    c = Canvas(img)
    draw_station(c, 1040 - t * 5, 120, 0.32, t)
    split = T * TITLE_FRAC
    if t < split:
        fade = clamp((split - t) / 0.4)
        items = ["No new hardware", "No lost privacy", "Uses Wi-Fi already on board"]
        for i, s in enumerate(items):
            a = ease((t - T * (0.05 + i * 0.09)) / 0.4) * fade
            if a <= 0:
                continue
            y = 200 + i * 80
            c.ell(110, y, 18, fill=lerp(BG, GREEN, a * 0.25), outline=lerp(BG, GREEN, a), w=2.5)
            c.line([(101, y), (108, y + 7), (120, y - 7)], lerp(BG, GREEN, a), 3.5)
            c.text(150, y, s, 32, lerp(BG, WHITE, a), "head", "lm")
    else:
        a = ease((t - split) / 0.7)
        c.text(640, 250, "Spatial-CSI", 84, lerp(BG, WHITE, a), "head", "mm")
        c.text(640, 320, "Astronaut Health & Safety Guard", 28, lerp(BG, CYAN, a), "head", "mm")
        b = ease((t - split - 0.6) / 0.7)
        c.text(640, 380, "Invisible protection for the people exploring space", 20, lerp(BG, MUTED, b), "ui", "mm")
        c.text(640, 440, "NASA SPACE APPS CHALLENGE", 15, lerp(BG, (175, 188, 210), b), "mono", "mm")


SCENES = [scene_orbit_night, scene_problem, scene_wifi, scene_capabilities, scene_dashboard, scene_close]


# ------------------------------------------------------------------ captions
def caption_chunks(text, max_words=9):
    words = text.split()
    chunks, cur = [], []
    for w_ in words:
        cur.append(w_)
        if len(cur) >= max_words or (w_[-1] in ".,?:" and len(cur) >= 4):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    # never leave a 1-2 word orphan: rebalance with the previous chunk
    for i in range(1, len(chunks)):
        if len(chunks[i]) <= 2 and len(chunks[i - 1]) > 4:
            move = len(chunks[i - 1]) // 2 - len(chunks[i]) // 2
            chunks[i] = chunks[i - 1][-move:] + chunks[i]
            chunks[i - 1] = chunks[i - 1][:-move]
    return [" ".join(ch) for ch in chunks]


def draw_caption(img, s):
    if not s:
        return
    c = Canvas(img)
    tw = c.text_w(s, 21, "uib")
    x0, x1 = 640 - tw / 2 - 18, 640 + tw / 2 + 18
    box = tuple(P(x0, 638, x1, 682))
    region = img.crop(box)
    region = Image.blend(region, Image.new("RGB", region.size, (0, 0, 0)), 0.6)
    img.paste(region, box[:2])
    c.text(640, 660, s, 21, WHITE, "uib", "mm")


# --------------------------------------------------------------------- main
def load_audio():
    lines = json.loads((ROOT / "narration.json").read_text(encoding="utf-8"))
    clips, rate = [], None
    for i in range(len(lines)):
        with wave.open(str(ROOT / "audio" / f"s{i}.wav")) as w_:
            rate = w_.getframerate()
            assert w_.getsampwidth() == 2 and w_.getnchannels() == 1, "expect 16-bit mono"
            clips.append(np.frombuffer(w_.readframes(w_.getnframes()), np.int16).astype(np.float32) / 32768)
    durs = [len(cl) / rate for cl in clips]
    return lines, clips, durs, rate


def build_timeline(lines, durs):
    spare = max(0.0, TARGET - sum(durs) - sum(LEADS))
    tails = [spare * w_ / sum(TAIL_W) for w_ in TAIL_W]
    scenes, t0 = [], 0.0
    for i, (ln, d) in enumerate(zip(lines, durs)):
        lead = LEADS[i]
        T = lead + d + tails[i]
        chunks = caption_chunks(ln["cap"])
        nw = [len(ch.split()) for ch in chunks]
        tot = sum(nw)
        caps, acc = [], 0
        for ch, k in zip(chunks, nw):
            caps.append((lead + d * acc / tot, lead + d * (acc + k) / tot, ch))
            acc += k
        scenes.append(dict(start=t0, T=T, lead=lead, caps=caps, fn=SCENES[i]))
        t0 += T
    return scenes, t0


def write_audio(scenes, clips, rate, total, path):
    import audio_mix
    assert rate == audio_mix.SR, f"narration must be {audio_mix.SR} Hz"
    voice = np.zeros(int(total * rate) + 1, np.float32)
    for sc, cl in zip(scenes, clips):
        i = int((sc["start"] + sc["lead"]) * rate)
        voice[i:i + len(cl)] += cl[: len(voice) - i]
    t_alert = scenes[4]["start"] + scenes[4]["T"] * ALERT_FRAC
    st = audio_mix.mix(voice.astype(np.float64), scenes, total, t_alert, TITLE_FRAC)
    with wave.open(str(path), "wb") as w_:
        w_.setnchannels(2); w_.setsampwidth(2); w_.setframerate(rate)
        w_.writeframes(st.tobytes())


def render_frame(sc, t, first, last):
    img = Image.new("RGB", (W * S, H * S), BG)
    sc["fn"](img, t, sc["T"])
    cap = next((s for a, b, s in sc["caps"] if a <= t < b + 0.25), "")
    draw_caption(img, cap)
    frame = img.resize((W, H), Image.LANCZOS)
    fi = 0.35 if not first else 0.8
    fo = 0.35 if not last else 1.2
    f = min(1.0, t / fi, (sc["T"] - t) / fo)
    if f < 1:
        frame = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), frame, clamp(f))
    return frame


def main():
    global STAR, EARTH, EARTH_BIG, INTERIOR
    STAR = make_starfield()
    EARTH = make_earth(700, 1250, 760)
    EARTH_BIG = make_earth(640, 1150, 640)
    INTERIOR = make_interior()
    lines, clips, durs, rate = load_audio()
    scenes, total = build_timeline(lines, durs)
    print("scene durations:", [round(s["T"], 2) for s in scenes], "total", round(total, 2))

    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        out = ROOT / "stills"
        out.mkdir(exist_ok=True)
        for i, sc in enumerate(scenes):
            for frac in (0.3, 0.75):
                render_frame(sc, sc["T"] * frac, False, False).save(out / f"s{i}_{int(frac*100)}.png")
        print("stills ->", out)
        return

    audio_path = ROOT / "audio" / "mix_full.wav"
    write_audio(scenes, clips, rate, total, audio_path)
    print("audio mixed ->", audio_path, flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "audio":
        return
    out = ROOT / "spatial_csi_90s.mp4"
    import os
    ff = os.environ.get("FFMPEG")
    if not ff:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(audio_path),
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                             "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)],
                            stdin=subprocess.PIPE)
    n = 0
    for i, sc in enumerate(scenes):
        nf = int(round(sc["T"] * FPS))
        for k in range(nf):
            fr = render_frame(sc, k / FPS, i == 0, i == len(scenes) - 1)
            proc.stdin.write(fr.tobytes())
            n += 1
        print(f"scene {i} done ({n} frames)", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", out, "frames", n, "seconds", round(n / FPS, 2))


if __name__ == "__main__":
    main()
