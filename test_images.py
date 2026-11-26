"""Тестовые изображения: мира 640 x 480 и Lena 512 x 512.
Эталон размывается (sigma = 1,5 пикселя) и зашумляется (sigma = 2) - это вход фильтра."""
import os

import numpy as np
from PIL import Image

import quality as q

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "test_images")
BLUR, NOISE, SEED = 1.5, 2.0, 2026

# геометрия миры (в пикселях итогового изображения)
W, H = 640, 480
SQ_C, SQ_HALF, ANG = (180.0, 240.0), 120.0, np.deg2rad(5.0)
PATCHES = {50: (370, 30), 120: (370, 195), 220: (370, 360)}   # уровень: (x0, y0)
PATCH = 90
STAR_C, STAR_R, STAR_N = (555.0, 240.0), 75.0, 24


def _rot(u, v):
    c, s = np.cos(ANG), np.sin(ANG)
    return SQ_C[0] + c * u - s * v, SQ_C[1] + s * u + c * v


def chart(ss=4):
    """Эталон миры (рисуется в 4 раза крупнее и уменьшается)."""
    y, x = np.mgrid[0:H * ss, 0:W * ss]
    x = (x + 0.5) / ss
    y = (y + 0.5) / ss
    img = np.full(x.shape, 180.0)
    dx, dy = x - SQ_C[0], y - SQ_C[1]
    c, s = np.cos(ANG), np.sin(ANG)
    u, v = c * dx + s * dy, -s * dx + c * dy
    img[(np.abs(u) < SQ_HALF) & (np.abs(v) < SQ_HALF)] = 70.0
    for lev, (x0, y0) in PATCHES.items():
        img[(x >= x0) & (x < x0 + PATCH) & (y >= y0) & (y < y0 + PATCH)] = lev
    dx, dy = x - STAR_C[0], y - STAR_C[1]
    r, th = np.hypot(dx, dy), np.arctan2(dy, dx)
    star = (r < STAR_R) & (np.sin(STAR_N / 2 * th) > 0)
    img[star] = 70.0
    return img.reshape(H, ss, W, ss).mean(axis=(1, 3))


def rois():
    """Участки для измерений: два перепада и три однородных поля."""
    xr, yr = _rot(SQ_HALF, 0)            # середина правой стороны квадрата
    xt, yt = _rot(0, -SQ_HALF)           # середина верхней стороны
    xr, yr, xt, yt = map(lambda t: int(round(t)), (xr, yr, xt, yt))
    edge_v = (slice(yr - 50, yr + 50), slice(xr - 45, xr + 45))   # -> MTF по горизонтали
    edge_h = (slice(yt - 45, yt + 45), slice(xt - 50, xt + 50))   # -> MTF по вертикали
    m = 20
    flats = {lev: (slice(y0 + m, y0 + PATCH - m), slice(x0 + m, x0 + PATCH - m))
             for lev, (x0, y0) in PATCHES.items()}
    return edge_v, edge_h, flats


def degrade(ref, rng):
    y = q.gauss_blur(ref, BLUR) + rng.normal(0, NOISE, ref.shape)
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def build():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ref1 = np.clip(np.rint(chart()), 0, 255).astype(np.uint8)
    lena = Image.open(os.path.join(HERE, "..", "lena.png")).convert("L")
    ref2 = np.array(lena)
    out = {}
    for name, ref in (("chart", ref1), ("lena", ref2)):
        deg = degrade(ref.astype(float), rng)
        Image.fromarray(ref).save(os.path.join(OUT, f"{name}_etalon.png"))
        Image.fromarray(deg).save(os.path.join(OUT, f"{name}_input.png"))
        out[name] = (ref, deg)
    return out


def load():
    """Эталон и вход для миры и Lena."""
    files = [os.path.join(OUT, f"{n}_{k}.png") for n in ("chart", "lena") for k in ("etalon", "input")]
    if not all(os.path.exists(f) for f in files):
        return build()
    r = lambda n, k: np.array(Image.open(os.path.join(OUT, f"{n}_{k}.png")))   # noqa: E731
    return {n: (r(n, "etalon"), r(n, "input")) for n in ("chart", "lena")}


if __name__ == "__main__":
    d = build()
    ev, eh, flats = rois()
    ref, deg = d["chart"]
    for nm, img in (("эталон", ref), ("вход", deg)):
        mh = q.slanted_edge(img[ev])["mtf50"]
        mv = q.slanted_edge(img[eh].T)["mtf50"]
        snr = q.snr_db(img[flats[120]])
        print(f"{nm}: MTF50 гор {mh:.3f} ({q.tvl(mh, W):.0f} ТВЛ), "
              f"верт {mv:.3f} ({q.tvl(mv, H):.0f} ТВЛ), ОСШ(120) {snr:.1f} дБ")
