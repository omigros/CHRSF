"""Рисунки для отчёта: каждая часть рисунка - отдельный файл в results/fig."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import crsf_fast as cf  # noqa: E402
import experiment as ex  # noqa: E402
import quality as q  # noqa: E402
import test_images as ti  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "fig")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "font.family": "DejaVu Sans"})
SMALL = (4.2, 3.1)      # часть рисунка (две в ряд)
WIDE = (6.5, 3.8)       # рисунок во всю ширину
APERS = [("7", "7"), ("7", "3"), ("3", "7"), ("11", "5")]
COL = {("7", "7"): "#3C388D", ("7", "3"): "#E39B2D", ("3", "7"): "#4BACC6", ("11", "5"): "#9BBB59"}


def rd(name):
    with open(os.path.join(RES, name), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v.replace(",", "."))
            except (ValueError, AttributeError):
                pass
    return rows


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)


def img_save(a, name, zoom=2):
    im = Image.fromarray(a)
    im.resize((a.shape[1] * zoom, a.shape[0] * zoom), Image.NEAREST).save(os.path.join(FIG, name))


timing = rd("task10_timing.csv")
t11 = rd("task11_all.csv")
t12 = rd("task12_all.csv")
base = rd("task11_baseline.csv")[0]

# ---------- рисунок 1.1: входные изображения
img_save(ex.DATA["chart"][1], "f1_1а.png", 1)
img_save(ex.DATA["lena"][1], "f1_1б.png", 1)

# ---------- рисунок 2.1: время от размера апертуры
for i, size in enumerate(("640x480", "1280x720", "3000x2000")):
    d = [r for r in timing if r["image"] == size and r["Nx"] == r["Ny"]]
    N = [int(r["Nx"]) for r in d]
    fig, ax = plt.subplots(figsize=SMALL)
    ax.semilogy(N, [r["t_kds"] for r in d], "o-", color="#C0504D", label="свёртка")
    ax.semilogy(N, [r["t_orig"] for r in d], "s-", color="#7F7F7F", label="ЧРСФ исходный")
    ax.semilogy(N, [r["t_opt"] for r in d], "^-", color="#3C388D", label="ЧРСФ ускоренный")
    ax.set_xlabel("Размер апертуры N (окно N × N)")
    ax.set_ylabel("Время обработки, с")
    ax.set_xticks(N)
    if i == 0:
        ax.legend(fontsize=9)
    save(fig, f"f2_1{'абв'[i]}.png")

# ---------- рисунок 2.2: неравносторонние апертуры, 3000 x 2000
SH = {3: "#E39B2D", 5: "#9BBB59", 7: "#3C388D"}
for j, (fix, var) in enumerate((("Nx", "Ny"), ("Ny", "Nx"))):
    fig, ax = plt.subplots(figsize=SMALL)
    for v in (3, 5, 7):
        d = sorted([r for r in timing if r["image"] == "3000x2000" and r[fix] == v], key=lambda r: r[var])
        x = [r[var] for r in d]
        ax.semilogy(x, [r["t_kds"] for r in d], "o-", ms=3, color=SH[v], label=f"свёртка, {fix} = {v}")
        ax.semilogy(x, [r["t_opt"] for r in d], "^--", ms=3, color=SH[v], label=f"ЧРСФ уск., {fix} = {v}")
    ax.set_xticks(list(range(3, 26, 4)))
    ax.set_xlabel(var)
    ax.set_ylabel("Время обработки, с")
    ax.legend(fontsize=7, ncol=2)
    save(fig, f"f2_2{'аб'[j]}.png")

# ---------- рисунок 2.3: число операций
N = np.arange(3, 26, 2)
fig, ax = plt.subplots(figsize=WIDE)
ax.plot(N, 2 * N * N - 1, "o-", color="#C0504D", label="свёртка")
ax.plot(N, [2 * cf.n_recirculators(n, n, False) + 2 + 4 for n in N], "s-", color="#7F7F7F",
        label="ЧРСФ исходный")
ax.plot(N, [2 * cf.n_recirculators(n, n) + 2 + 3 for n in N], "^-", color="#3C388D",
        label="ЧРСФ ускоренный")
ax.set_yscale("log")
ax.set_xlabel("Размер апертуры N (окно N × N)")
ax.set_ylabel("Операций на один пиксель")
ax.set_xticks(N)
ax.legend()
save(fig, "f2_3.png")


# ---------- рисунки 3.1, 3.2: серии A1 и A2
def lab(a):
    return f"{a[0]} × {a[1]}"


for ser, key, fig_no in (("S1", "A1", "3_1"), ("S2", "A2", "3_2")):
    for j, (val, ylab, bl) in enumerate((("tvl_h", "Разрешение, ТВЛ", base["tvl_h"]),
                                         ("snr_120", "ОСШ, дБ", base["snr_120"]))):
        fig, ax = plt.subplots(figsize=SMALL)
        for a in APERS:
            d = [r for r in t11 if (r["Nx"], r["Ny"]) == tuple(map(float, a)) and r["series"] == ser]
            ax.plot([r[key] for r in d], [r[val] for r in d], color=COL[a], label=lab(a))
        ax.axhline(bl, ls="--", color="k", lw=1, label="до обработки")
        ax.set_xlabel(f"Коэффициент {key}")
        ax.set_ylabel(ylab)
        if j == 0:
            ax.legend(fontsize=8)
        save(fig, f"f{fig_no}{'аб'[j]}.png")

# ---------- рисунок 3.3: зависимость от k при разных w (7 x 7)
WS = {1.0: "w = 1 (только A2)", 0.5: "w = 0,5", 0.0: "w = 0 (только A1)"}
keys = [("tvl_h", "Разрешение, ТВЛ", base["tvl_h"]), ("snr_120", "ОСШ, дБ", base["snr_120"]),
        ("over_h", "Выброс, %", 0), ("psnr_lena", "PSNR, дБ", base["psnr_lena"]),
        ("ssim_lena", "SSIM", base["ssim_lena"])]
for j, (val, ylab, bl) in enumerate(keys):
    fig, ax = plt.subplots(figsize=SMALL)
    for w, wl in WS.items():
        d = [r for r in t11 if r["Nx"] == 7 and r["Ny"] == 7 and r["series"] == f"S3 w={w}"]
        ax.semilogx([r["k"] for r in d], [r[val] for r in d], "-o", ms=2.5, label=wl)
    ax.axhline(bl, ls="--", color="k", lw=1, label="до обработки")
    ax.set_xlabel("Коэффициент усиления k")
    ax.set_ylabel(ylab)
    if j == 0:
        ax.legend(fontsize=8)
    save(fig, f"f3_3{'абвгд'[j]}.png")

# ---------- рисунок 3.4: сетка A1 x A2
GA1 = [0, 0.5, 1, 2, 3, 5, 10, 20, 50]
GA2 = [0, 1, 5, 21, 50, 100, 200, 500, 1000, 2000]
g = [r for r in t11 if r["series"] == "S4"]
for j, (val, name) in enumerate((("tvl_h", "ТВЛ"), ("snr_120", "ОСШ, дБ"), ("psnr_lena", "PSNR, дБ"))):
    Mx = np.full((len(GA1), len(GA2)), np.nan)
    for r in g:
        Mx[GA1.index(r["A1"]), GA2.index(r["A2"])] = r[val]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(Mx, cmap="viridis", aspect="auto", origin="lower")
    for i in range(Mx.shape[0]):
        for k in range(Mx.shape[1]):
            if not np.isnan(Mx[i, k]):
                ax.text(k, i, f"{Mx[i, k]:.0f}", ha="center", va="center", fontsize=7,
                        color="w" if Mx[i, k] < np.nanmean(Mx) else "k")
    ax.set_xticks(range(len(GA2)), [str(v) for v in GA2], fontsize=8)
    ax.set_yticks(range(len(GA1)), [str(v).replace(".", ",") for v in GA1], fontsize=8)
    ax.set_xlabel("A2")
    ax.set_ylabel("A1")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=name)
    save(fig, f"f3_4{'абв'[j]}.png")

# ---------- рисунок 3.5: фрагменты Lena, окно 7 x 7
lref, linp = ex.DATA["lena"]
cr = (slice(220, 380), slice(220, 380))
variants = [None, (1, 0), (0, 21), (3, 0), (14, 0), (24, 0), "ref"]
for j, v in enumerate(variants):
    a = linp if v is None else lref if v == "ref" else ex.process(linp, 7, 7, *v)
    img_save(a[cr], f"f3_5{'абвгдеж'[j]}.png")

# ---------- рисунок 3.6: MTF
cinp = ex.DATA["chart"][1]
fig, ax = plt.subplots(figsize=WIDE)
e = q.slanted_edge(cinp[ex.EDGE_V])
ax.plot(e["freq"], e["mtf"], "k--", label="до обработки")
for name, v in (("A1 = 1", (1, 0)), ("A2 = 21", (0, 21)), ("A1 = 14", (14, 0)), ("A1 = 24", (24, 0))):
    e = q.slanted_edge(ex.process(cinp, 7, 7, *v)[ex.EDGE_V])
    ax.plot(e["freq"], e["mtf"], label=name)
ax.axhline(0.5, color="gray", lw=0.8)
ax.set_xlim(0, 0.6)
ax.set_xlabel("Пространственная частота, цикл/пиксель")
ax.set_ylabel("MTF")
ax.legend(fontsize=9)
save(fig, "f3_6.png")

# ---------- рисунок 4.1: квадратные окна, режимы A и B
keys = [("tvl_h", "Разрешение, ТВЛ", base["tvl_h"]), ("snr_120", "ОСШ, дБ", base["snr_120"]),
        ("psnr_lena", "PSNR, дБ", base["psnr_lena"]), ("ssim_lena", "SSIM", base["ssim_lena"]),
        ("over_h", "Выброс, %", None), ("halo_h", "Ширина ореола, пикс.", None)]
SIZES = list(range(3, 26, 2))
for j, (val, ylab, bl) in enumerate(keys):
    fig, ax = plt.subplots(figsize=SMALL)
    for mode, st, ml in (("A", "o-", "режим A"), ("B", "s-", "режим B")):
        d = [r for r in t12 if r["mode"].startswith(mode) and r["series"] == "квадратные"]
        ax.plot([r["Nx"] for r in d], [r[val] for r in d], st, ms=3.5, label=ml)
    if bl is not None:
        ax.axhline(bl, ls="--", color="k", lw=1, label="до обработки")
    ax.set_xticks(SIZES[::2])
    ax.set_xlabel("Размер апертуры N")
    ax.set_ylabel(ylab)
    if j == 0:
        ax.legend(fontsize=8)
    save(fig, f"f4_1{'абвгде'[j]}.png")

# ---------- рисунок 4.2: частотные свойства (режим B)
dB = [r for r in t12 if r["mode"].startswith("B") and r["series"] == "квадратные"]
fig, ax = plt.subplots(figsize=SMALL)
for r in dB:
    if r["Nx"] in (3, 7, 11, 15, 25):
        mp = q.mask_props(cf.aperture_fast(int(r["Nx"]), int(r["Ny"]), r["A1"], r["A2"]))
        ax.plot(mp["f"], mp["Hx"], label=f"{int(r['Nx'])} × {int(r['Ny'])}")
ax.axhline(1, color="k", lw=0.8)
ax.set_xlabel("Частота, цикл/пиксель")
ax.set_ylabel("Коэффициент передачи")
ax.legend(fontsize=8)
save(fig, "f4_2а.png")
fig, ax = plt.subplots(figsize=SMALL)
ax.plot([r["Nx"] for r in dB], [r["fpeak_h"] for r in dB], "o-")
ax.set_xticks(SIZES[::2])
ax.set_xlabel("Размер апертуры N")
ax.set_ylabel("Частота макс. усиления,\nцикл/пиксель")
save(fig, "f4_2б.png")

# ---------- рисунок 4.3: профиль перепада
get = {(int(r["Nx"]), int(r["Ny"])): r for r in t12 if r["mode"].startswith("B")}
fig, ax = plt.subplots(figsize=WIDE)
e = q.slanted_edge(ex.CLEAN[ex.EDGE_V])
ax.plot(e["pos"], e["esf"], "k--", label="до обработки")
for N in (3, 7, 15, 25):
    r = get[(N, N)]
    out = np.clip(cf.crsf_fast(ex.CLEAN, N, N, r["A1"], r["A2"]), 0, 255)
    e = q.slanted_edge(out[ex.EDGE_V])
    ax.plot(e["pos"], e["esf"], label=f"{N} × {N}")
ax.set_xlim(-20, 20)
ax.set_xlabel("Расстояние от границы, пикс.")
ax.set_ylabel("Относительная яркость")
ax.legend(fontsize=9)
save(fig, "f4_3.png")

# ---------- рисунок 4.4: неравносторонние окна
for j, s in enumerate(("Nx = 3, меняем Ny", "Nx = 5, меняем Ny", "Nx = 7, меняем Ny",
                       "Ny = 3, меняем Nx", "Ny = 5, меняем Nx", "Ny = 7, меняем Nx")):
    d = [r for r in t12 if r["mode"].startswith("B") and r["series"] == s]
    var = "Nx" if "меняем Nx" in s else "Ny"
    fig, ax = plt.subplots(figsize=SMALL)
    ax.plot([r[var] for r in d], [r["tvl_h"] for r in d], "o-", label="по горизонтали")
    ax.plot([r[var] for r in d], [r["tvl_v"] * ti.W / ti.H for r in d], "s-", label="по вертикали")
    ax.axhline(base["tvl_h"], ls="--", color="k", lw=1, label="до обработки")
    ax.set_xticks(SIZES[::2])
    ax.set_xlabel(var)
    ax.set_ylabel("Разрешение, ТВЛ")
    if j == 0:
        ax.legend(fontsize=8)
    save(fig, f"f4_4{'абвгде'[j]}.png")

# ---------- рисунки 4.5, 4.6: фрагменты при разных апертурах
shapes = [None, (3, 3), (7, 7), (15, 15), (25, 25), (25, 3), (3, 25)]
cinp = ex.DATA["chart"][1]
for j, a in enumerate(shapes):
    for img, crop, fno in ((linp, (slice(220, 380), slice(220, 380)), "4_5"),
                           (cinp, (slice(165, 315), slice(480, 630)), "4_6")):
        out = img if a is None else ex.process(img, *a, get[a]["A1"], get[a]["A2"])
        img_save(out[crop], f"f{fno}{'абвгдеж'[j]}.png")
print("Готово:", len(os.listdir(FIG)), "файлов")
