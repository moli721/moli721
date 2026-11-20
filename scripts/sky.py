"""Dithered-sky banners for the GitHub profile README.

Pure Python (no numpy / PIL). Renders a sky layer and a cloud layer on a coarse
dot grid, applies 4x4 Bayer ordered dither to luminance only (10 levels), and
writes SVGs that embed the two layers as pixelated PNGs plus crisp vector text.

usage: python3 scripts/sky.py img
"""
import base64
import math
import random
import struct
import sys
import zlib

DOT = 2.5  # user units per dot
LEVELS = 10
BAYER4 = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]


def bayer(x, y):
    return (BAYER4[y % 4][x % 4] + 0.5) / 16.0


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def smoothstep(a, b, x):
    t = clamp((x - a) / (b - a))
    return t * t * (3 - 2 * t)


def mix(c1, c2, t):
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


def gradient(stops, t):
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            return mix(c0, c1, smoothstep(0, 1, (t - t0) / (t1 - t0)))
    return stops[-1][1]


# ---------------------------------------------------------------- noise

def _hash(ix, iy, seed):
    n = (ix * 374761393 + iy * 668265263 + seed * 982451653) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    n ^= n >> 16
    return n


def worley(x, y, seed):
    xi, yi = math.floor(x), math.floor(y)
    best = 9.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = xi + dx, yi + dy
            n = _hash(cx, cy, seed)
            px = cx + (n & 0xFFFF) / 65536.0
            py = cy + (n >> 16) / 65536.0
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best:
                best = d
    return math.sqrt(best)


def billow(x, y, seed, octaves=4):
    s, w, f, tot = 0.0, 0.5, 1.0, 0.0
    for i in range(octaves):
        s += w * (1 - worley(x * f, y * f, seed + i * 17))
        tot += w
        f *= 2.07
        w *= 0.5
    return s / tot


# ---------------------------------------------------------------- png

def png(width, height, rgba_rows):
    raw = b"".join(b"\x00" + bytes(row) for row in rgba_rows)

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def dither(rgb, x, y):
    r, g, b = rgb
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    q = math.floor(lum * LEVELS + bayer(x, y)) / LEVELS
    d = (q - lum) * 255.0
    return [int(clamp(c + d, 0, 255) + 0.5) for c in (r, g, b)]


# ---------------------------------------------------------------- scene

def cloud_field(gw, gh, x0, ellipses, seed, scale=17.0):
    """Per-dot density, coverage, shape height, below-centre factor, detail."""
    dens = [[-1.0] * gw for _ in range(gh)]
    shp = [[0.0] * gw for _ in range(gh)]
    low = [[0.0] * gw for _ in range(gh)]
    det = [[0.0] * gw for _ in range(gh)]
    for y in range(gh):
        for i in range(gw):
            x = x0 + i
            base, below = -9.0, 0.0
            for (cx, cy, rx, ry) in ellipses:
                dx, dy = (x - cx) / rx, (y - cy) / ry
                flat = dy * 1.8 if dy > 0 else dy
                v = 1 - math.sqrt(dx * dx + flat * flat)
                if v > base:
                    base, below = v, clamp(dy * 1.6)
            if base < -0.5:
                continue
            s = billow(x / scale, y / scale, seed, 2)
            shp[y][i] = s
            low[y][i] = below
            det[y][i] = billow(x / 6.0, y / 6.0, seed + 5, 2)
            dens[y][i] = base + (s - 0.6) * 0.85 * (1 - 0.8 * below)
    cov = [[smoothstep(0.0, 0.07, d) for d in row] for row in dens]
    return dens, cov, shp, low, det


def render(scene):
    gw, gh, pad = scene["gw"], scene["gh"], scene["pad"]
    sky_rows, cloud_rows = [], []

    # sky layer
    for y in range(gh):
        row = []
        yf = y / (gh - 1)
        for x in range(gw):
            c = gradient(scene["sky"], yf)
            c = mix(c, scene["fog"], smoothstep(scene["fog_from"], 1.0, yf) * scene["fog_amt"])
            for (wx0, wx1, wy, seed) in scene.get("cirrus", []):
                if wx0 - 20 < x < wx1 + 20:
                    env = smoothstep(wx0 - 20, wx0 + 10, x) * (1 - smoothstep(wx1 - 10, wx1 + 20, x))
                    env *= math.exp(-((y - wy) / 3.2) ** 2)
                    if env > 0.02:
                        k = smoothstep(0.55, 0.85, billow(x / 30.0, y / 2.4, seed, 3)) * env
                        c = mix(c, scene["cirrus_color"], k * 0.8)
            moon = scene.get("moon")
            if moon:
                mx, my, mr = moon
                dist = math.hypot(x - mx, y - my)
                c = mix(c, (70, 82, 140), math.exp(-dist / 16.0) * 0.55)
                if dist < mr and math.hypot(x - mx - 2.6, y - my + 1.8) > mr - 0.6:
                    c = (226, 231, 248)
            row.extend(dither(c, x, y) + [255])
        sky_rows.append(row)

    # cloud layer (wider than the card so the sway never shows an edge)
    cw = gw + 2 * pad
    dens, cov, shp, low, det = cloud_field(cw, gh, -pad, scene["clouds"], scene["seed"])
    for y in range(gh):
        row = []
        yf = y / (gh - 1)
        for i in range(cw):
            a = cov[y][i]
            if a <= bayer(i, y):
                row.extend((0, 0, 0, 0))
                continue
            occl = (cov[max(y - 4, 0)][i] + cov[max(y - 9, 0)][i]) * 0.5
            hd = shp[y][i] - shp[max(y - 2, 0)][i]
            light = clamp(1.08 - 0.42 * occl - 0.38 * low[y][i] + 5.0 * hd + (det[y][i] - 0.62) * 0.35)
            c = mix(scene["cloud_shadow"], scene["cloud_lit"], light)
            c = mix(c, scene["fog"], smoothstep(scene["fog_from"], 1.0, yf) * scene["fog_amt"] * 0.8)
            row.extend(dither(c, i, y) + [255])
        cloud_rows.append(row)

    return png(gw, gh, sky_rows), png(cw, gh, cloud_rows)


def stars(scene, n, seed):
    rnd = random.Random(seed)
    out = []
    gw, gh = scene["gw"], scene["gh"]
    while len(out) < n:
        x, y = rnd.randrange(2, gw - 2), rnd.randrange(2, int(gh * scene["star_band"]))
        moon = scene.get("moon")
        if moon and math.hypot(x - moon[0], y - moon[1]) < moon[2] + 6:
            continue
        out.append((x, y, rnd.random()))
    return out


# ---------------------------------------------------------------- svg

def b64(data):
    return "data:image/png;base64," + base64.b64encode(data).decode()


LOCK = ("M-3.5 -1h7v5.5h-7z M-2.2 -1v-2a2.2 2.2 0 0 1 4.4 0v2")


def notch_path(cx, w, h=32, r=15, rr=9):
    x0, x1 = cx - w / 2, cx + w / 2
    return (f"M{x0 - rr} 0 A{rr} {rr} 0 0 1 {x0} {rr} L{x0} {h - r} "
            f"A{r} {r} 0 0 0 {x0 + r} {h} L{x1 - r} {h} A{r} {r} 0 0 0 {x1} {h - r} "
            f"L{x1} {rr} A{rr} {rr} 0 0 1 {x1 + rr} 0 Z")


def svg(scene, sky_png, cloud_png, star_list, kind):
    W, H = scene["gw"] * DOT, scene["gh"] * DOT
    pad_u = scene["pad"] * DOT
    t = scene["text"]
    star_svg = ""
    if star_list:
        groups = {0: [], 1: [], 2: []}
        for (x, y, r) in star_list:
            groups[int(r * 3)].append(
                f'<rect x="{x * DOT}" y="{y * DOT}" width="{DOT}" height="{DOT}"/>')
        star_svg = "".join(
            f'<g class="tw tw{k}" fill="{t["star"]}">{"".join(v)}</g>' for k, v in groups.items())

    if kind == "header":
        cx = W / 2
        fg = f"""
    <path d="{notch_path(cx, 214)}" fill="{t['notch_fill']}" stroke="{t['notch_stroke']}" stroke-width="1"/>
    <g transform="translate({cx - 66} 18.5)" fill="none" stroke="{t['notch_text']}" stroke-width="1.3" stroke-linejoin="round"><path d="{LOCK}"/></g>
    <text class="mono" x="{cx + 6}" y="21" font-size="12" text-anchor="middle" fill="{t['notch_text']}">moli721.xyz</text>
    <text class="sans" x="{cx}" y="{H * 0.53}" font-size="62" font-weight="700" letter-spacing="-1.5" text-anchor="middle" fill="{t['word']}" filter="url(#sh)">moli721</text>
    <text class="mono" x="{cx}" y="{H * 0.53 + 30}" font-size="11" font-weight="600" letter-spacing="6" text-anchor="middle" fill="{t['tag']}" filter="url(#sh)">A LONG WAY TO GO</text>"""
    else:
        fg = f"""
    <text class="sans" x="{W / 2}" y="{H * 0.42}" font-size="20" font-weight="700" text-anchor="middle" fill="{t['word']}" filter="url(#sh)">Thanks for stopping by</text>
    <text class="mono" x="{W / 2}" y="{H * 0.42 + 22}" font-size="10" font-weight="600" letter-spacing="5" text-anchor="middle" fill="{t['tag']}" filter="url(#sh)">SEE YOU AROUND</text>"""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W:g}" height="{H:g}" viewBox="0 0 {W:g} {H:g}" role="img" aria-label="{scene['label']}">
  <style>
    .px {{ image-rendering: optimizeSpeed; image-rendering: crisp-edges; image-rendering: pixelated; }}
    .sans {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Helvetica, Arial, sans-serif; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; }}
    .sway {{ animation: sway 48s steps(12) infinite alternate; }}
    .tw {{ animation: tw 3.6s steps(2) infinite; }}
    .tw1 {{ animation-duration: 5.2s; animation-delay: -1.3s; }}
    .tw2 {{ animation-duration: 7.4s; animation-delay: -2.9s; }}
    @keyframes sway {{ from {{ transform: translateX(-15px); }} to {{ transform: translateX(15px); }} }}
    @keyframes tw {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .25; }} }}
    @media (prefers-reduced-motion: reduce) {{ .sway, .tw {{ animation: none; }} }}
  </style>
  <defs>
    <clipPath id="card"><rect width="{W:g}" height="{H:g}" rx="14"/></clipPath>
    <filter id="sh" x="-20%" y="-50%" width="140%" height="200%"><feDropShadow dx="0" dy="2" stdDeviation="5" flood-color="{t['shadow']}" flood-opacity="{t['shadow_op']}"/></filter>
  </defs>
  <g clip-path="url(#card)">
    <image class="px" width="{W:g}" height="{H:g}" preserveAspectRatio="none" href="{b64(sky_png)}"/>
    {star_svg}
    <g class="sway"><image class="px" x="{-pad_u:g}" width="{W + 2 * pad_u:g}" height="{H:g}" preserveAspectRatio="none" href="{b64(cloud_png)}"/></g>{fg}
  </g>
</svg>
"""


# ---------------------------------------------------------------- scenes

HEADER_CLOUDS = [
    (-6, 70, 34, 16), (30, 60, 38, 22), (66, 54, 30, 22), (96, 68, 26, 13),
    (250, 70, 22, 11), (280, 58, 34, 22), (318, 52, 32, 24), (352, 66, 34, 18),
]
FOOTER_CLOUDS = [
    (-10, 50, 40, 20), (36, 49, 34, 18), (84, 52, 42, 20), (140, 50, 36, 17),
    (196, 52, 44, 20), (252, 49, 36, 18), (300, 52, 40, 20), (352, 50, 38, 18),
]

DAY = dict(
    sky=[(0, (46, 102, 180)), (0.45, (98, 154, 218)), (0.8, (176, 208, 240)), (1, (232, 241, 251))],
    fog=(242, 247, 253), fog_from=0.7, fog_amt=0.9,
    cloud_lit=(255, 255, 255), cloud_shadow=(150, 172, 204),
    cirrus_color=(240, 246, 255),
    text=dict(word="#FFFFFF", tag="#FFFFFF", shadow="#14203C", shadow_op=".35",
              notch_fill="rgb(255 255 255 / 78%)", notch_stroke="rgb(255 255 255 / 60%)",
              notch_text="#454D60", star="#FFFFFF"),
)
NIGHT = dict(
    sky=[(0, (7, 9, 26)), (0.5, (14, 18, 46)), (0.85, (34, 40, 82)), (1, (58, 62, 112))],
    fog=(50, 56, 102), fog_from=0.72, fog_amt=0.7,
    cloud_lit=(118, 132, 180), cloud_shadow=(30, 37, 74),
    cirrus_color=(64, 74, 124),
    text=dict(word="#E8EDF9", tag="#AEB8D6", shadow="#000000", shadow_op=".55",
              notch_fill="rgb(255 255 255 / 9%)", notch_stroke="rgb(255 255 255 / 14%)",
              notch_text="#AEB8D3", star="#E6EBFF"),
)


def build(out_dir):
    jobs = []
    for mode, pal in (("light", DAY), ("dark", NIGHT)):
        jobs.append(("header", mode, dict(pal, gw=344, gh=96, pad=8, seed=7, star_band=0.62,
                                          clouds=HEADER_CLOUDS, label="moli721 · A long way to go",
                                          cirrus=[(14, 120, 15, 101), (226, 330, 13, 103)],
                                          moon=(302, 18, 7) if mode == "dark" else None)))
        jobs.append(("footer", mode, dict(pal, gw=344, gh=44, pad=8, seed=11, star_band=0.7,
                                          clouds=FOOTER_CLOUDS, label="Thanks for stopping by",
                                          fog_from=0.6)))
    for kind, mode, scene in jobs:
        sky_png, cloud_png = render(scene)
        st = stars(scene, 46 if kind == "header" else 22, 3) if mode == "dark" else []
        path = f"{out_dir}/{kind}-{mode}.svg"
        with open(path, "w") as f:
            f.write(svg(scene, sky_png, cloud_png, st, kind))
        print(path, len(sky_png) + len(cloud_png), "bytes png")


if __name__ == "__main__":
    build(sys.argv[1])
