"""Рисунки раздела 1 отчёта: устройство ЧРСФ (схемы, алгоритм, маски)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

import crsf_fast as cf  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "fig")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavuserif"})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=1.0, color="k",
                                 shrinkA=0, shrinkB=0))


def box(ax, x, y, w, h, text, fs=12):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, fc="w", ec="k", lw=1.0))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs)


# ---------- рисунок 1.1: рециркуляторы
def recirculator(name, idx, letter):
    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.text(0.0, 3.85, f"$x(n_1, n_2)$", fontsize=12)
    ax.plot([0, 1.2], [3.5, 3.5], "k", lw=1); ax.plot(1.2, 3.5, "ko", ms=3)
    arrow(ax, (1.2, 3.5), (5.6, 3.5))
    ax.plot([1.2, 1.2], [3.5, 1.2], "k", lw=1); arrow(ax, (1.2, 1.2), (2.2, 1.2))
    box(ax, 2.2, 1.2, 1.8, 1.0, f"$Z_{idx}^{{-M_{idx}}}$")
    ax.plot([4.0, 6.2], [1.2, 1.2], "k", lw=1); arrow(ax, (6.2, 1.2), (6.2, 2.9))
    ax.add_patch(Rectangle((5.6, 2.9), 1.3, 1.3, fc="w", ec="k", lw=1.0))
    ax.text(6.25, 3.55, "Σ", ha="center", va="center", fontsize=16)
    ax.text(5.15, 3.65, "+", fontsize=12); ax.text(5.8, 2.45, "–", fontsize=13)
    arrow(ax, (6.9, 3.5), (9.9, 3.5)); ax.text(7.3, 3.85, "$y(n_1, n_2)$", fontsize=12)
    ax.plot(8.9, 3.5, "ko", ms=3)
    ax.plot([8.9, 8.9], [3.5, 1.2], "k", lw=1); arrow(ax, (8.9, 1.2), (8.6, 1.2))
    box(ax, 7.0, 1.2, 1.6, 1.0, f"$Z_{idx}^{{-1}}$")
    ax.plot([7.0, 6.65], [1.2, 1.2], "k", lw=1); arrow(ax, (6.65, 1.2), (6.65, 2.9))
    ax.text(6.75, 2.45, "+", fontsize=12)
    save(fig, f"s1_1{letter}.png")


recirculator("СР", 1, "а")
recirculator("КР", 2, "б")


# ---------- структурные схемы
def scheme(name, b1, b2, b3, signs=("+", "–", "+"), norm=True):
    """b1, b2, b3 - подписи блоков ветвей; ветвь 2 самая длинная."""
    w, h, g = 2.15, 1.2, 0.35
    x0 = 1.2
    n2 = len(b2)
    xend = x0 + n2 * w + (n2 - 1) * g
    sx = xend + 0.9                       # левый край сумматора
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    ax.set_xlim(-0.2, sx + 5.2 if norm else sx + 3.2); ax.set_ylim(0, 7.9); ax.axis("off")
    ys = (6.3, 3.8, 1.3)
    ax.text(-0.2, 7.25, "$x(n_1, n_2)$", fontsize=12)
    ax.plot([-0.2, 0.5], [6.3, 6.3], "k", lw=1)
    ax.plot([0.5, 0.5], [6.3, 1.3], "k", lw=1)
    for k, (blocks, y, sg) in enumerate(zip((b1, b2, b3), ys, signs)):
        start = n2 - len(blocks) if k == 2 else 0
        xs = [x0 + (start + i) * (w + g) for i in range(len(blocks))]
        ax.plot(0.5, y, "ko", ms=3) if k == 1 else None
        arrow(ax, (0.5, y), (xs[0], y))
        for i, t in enumerate(blocks):
            box(ax, xs[i], y, w, h, t, fs=11)
            if i < len(blocks) - 1:
                arrow(ax, (xs[i] + w, y), (xs[i + 1], y))
        arrow(ax, (xs[-1] + w, y), (sx, y))
        ax.text(sx - 0.35, y + 0.15, sg, fontsize=13)
    ax.add_patch(Rectangle((sx, 0.5), 1.4, 6.6, fc="w", ec="k", lw=1.0))
    ax.text(sx + 0.7, 3.8, "Σ", ha="center", va="center", fontsize=22)
    if norm:
        arrow(ax, (sx + 1.4, 3.8), (sx + 2.0, 3.8))
        box(ax, sx + 2.0, 3.8, 1.2, h, "1 / S", fs=11)
        arrow(ax, (sx + 3.2, 3.8), (sx + 5.1, 3.8))
        ax.text(sx + 3.3, 4.2, "$y(n_1, n_2)$", fontsize=12)
    else:
        arrow(ax, (sx + 1.4, 3.8), (sx + 3.1, 3.8))
        ax.text(sx + 1.5, 4.2, "$y(n_1, n_2)$", fontsize=12)
    save(fig, name)


CP = "СР\n$M_1 = {}$"
KP = "КР\n$M_2 = {}$"
# конкретная схема 7 x 7
scheme("s1_5.png",
       ["$81 + A_1$", "$Z_1^{-2}$", "$Z_2^{-2}$", CP.format(2), CP.format(2), KP.format(2), KP.format(2)],
       [CP.format(3), CP.format(3), CP.format(2), CP.format(2), KP.format(3), KP.format(3), KP.format(2), KP.format(2)],
       ["$A_2$", "$Z_1^{-3}$", "$Z_2^{-3}$"])
# обобщённая схема с неравносторонней апертурой
scheme("s1_6.png",
       ["$X + A_1$", "$Z_1^{-a_1}$", "$Z_2^{-a_2}$", CP.format("m_1"), CP.format("m_1"), KP.format("m_2"),
        KP.format("m_2")],
       [CP.format("n_1"), CP.format("n_2"), CP.format("n_3"), CP.format("n_4"),
        KP.format("k_1"), KP.format("k_2"), KP.format("k_3"), KP.format("k_4")],
       ["$A_2$", "$Z_1^{-b_1}$", "$Z_2^{-b_2}$"])


# ---------- алгоритм формирования внешней маски (как рис. 2 статьи)
def cells(ax, x, y, vals, vertical=False, size=0.62, fs=11):
    for i, v in enumerate(vals):
        cx, cy = (x, y - i * size) if vertical else (x + i * size, y)
        ax.add_patch(Rectangle((cx, cy), size, size, fc="w", ec="k", lw=0.8))
        ax.text(cx + size / 2, cy + size / 2, f"{v:g}", ha="center", va="center", fontsize=fs)


def algorithm(name, Mx, My):
    d = np.zeros((sum(My), sum(Mx))); d[0, 0] = 1
    Nx, Ny = sum(Mx) - 3, sum(My) - 3
    O = cf.chain(cf.chain(d, Mx, 1), My, 0)[:Ny, :Nx]
    s = 0.62
    fig, ax = plt.subplots(figsize=(12.5, 0.62 * max(Ny, 3) * 0.75 + 0.6))
    x = 0.0
    yc = (Ny * s) / 2 - s / 2
    items = [("h", M) for M in Mx] + [("v", M) for M in My]
    for j, (kind, M) in enumerate(items):
        if kind == "h":
            cells(ax, x, yc, [1] * M); x += M * s
        else:
            top = yc + (M - 1) * s / 2
            cells(ax, x, top, [1] * M, vertical=True); x += s
        ax.text(x + 0.22, yc + s / 2, "*" if j < len(items) - 1 else "=", ha="center", va="center", fontsize=14)
        x += 0.45
    for i in range(Ny):
        cells(ax, x, (Ny - 1 - i) * s, O[i], fs=10)
    ax.set_xlim(-0.1, x + Nx * s + 0.1); ax.set_ylim(-0.1, Ny * s + 0.1)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, name)


algorithm("s1_2.png", [3, 3, 2, 2], [3, 3, 2, 2])
algorithm("s1_7.png", [3, 3, 2, 2], [2, 2, 2, 2])


# ---------- маски числами (как рис. 3 статьи)
def mask_grid(name, K, fs=10, canvas=None):
    Ny, Nx = K.shape
    CW, CH = canvas or (Nx, Ny)              # общий холст: одинаковый размер клеток
    ox, oy = (CW - Nx) / 2, (CH - Ny) / 2
    fig, ax = plt.subplots(figsize=(0.55 * CW + 0.2, 0.42 * CH + 0.2))
    for i in range(Ny):
        for j in range(Nx):
            v = K[i, j]
            fc = "#D9D9D9" if v > 0 else "w"
            ax.add_patch(Rectangle((ox + j, oy + Ny - 1 - i), 1, 1, fc=fc, ec="k", lw=0.8))
            ax.text(ox + j + 0.5, oy + Ny - 0.5 - i, f"{v:g}".replace("-", "–"), ha="center", va="center",
                    fontsize=fs)
    ax.set_xlim(0, CW); ax.set_ylim(0, CH); ax.set_aspect(0.75); ax.axis("off")
    save(fig, name)


mask_grid("s1_3.png", cf.aperture_fast(7, 7, 0, 0, False))
mask_grid("s1_8а.png", cf.aperture_fast(7, 5, 0, 0, False))
mask_grid("s1_9а.png", cf.aperture_fast(7, 3, 0, 0, False), canvas=(7, 7))
mask_grid("s1_9б.png", cf.aperture_fast(3, 7, 0, 0, False), canvas=(7, 7))


# ---------- трёхмерный вид (как рис. 4 статьи)
def surf(name, K):
    fig = plt.figure(figsize=(4.6, 3.6))
    ax = fig.add_subplot(projection="3d")
    Y, X = np.mgrid[0:K.shape[0], 0:K.shape[1]]
    ax.plot_surface(X, Y, K, cmap="coolwarm", edgecolor="k", lw=0.3)
    ax.set_xlabel("$n_1$"); ax.set_ylabel("$n_2$")
    ax.view_init(28, -60)
    save(fig, name)


surf("s1_4.png", cf.aperture_fast(7, 7, 0, 0, False))
surf("s1_8б.png", cf.aperture_fast(7, 5, 0, 0, False))
surf("s1_10а.png", cf.aperture_fast(15, 5, 0, 0, False))
surf("s1_10б.png", cf.aperture_fast(5, 15, 0, 0, False))
print("Готово")
