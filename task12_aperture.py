"""Задача 12: влияние размеров апертуры на качество обработки.
Режим A - A1, A2 постоянны; режим B - постоянны k и w."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import crsf_fast as cf  # noqa: E402
import experiment as ex  # noqa: E402
import quality as q  # noqa: E402
import test_images as ti  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SIZES = list(range(3, 26, 2))


def optimum_7x7():
    """(A1, A2, k, w) оптимума 7 x 7 по сбалансированному критерию из task11_optimum.csv."""
    with open(os.path.join(RES, "task11_optimum.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["aperture"] == "7 × 7" and r["criterion"].startswith("max ТВЛ при ОСШ"):
                g = lambda k: float(r[k].replace(",", "."))   # noqa: E731
                return g("A1"), g("A2"), g("k"), g("w")
    raise RuntimeError("Сначала выполните task11_coefficients.py")


def series():
    out = [("квадратные", (N, N)) for N in SIZES]
    for fix in (3, 5, 7):
        out += [(f"Ny = {fix}, меняем Nx", (N, fix)) for N in SIZES]
        out += [(f"Nx = {fix}, меняем Ny", (fix, N)) for N in SIZES]
    return out


def run(A1, A2, k, w):
    rows = []
    for name, (Nx, Ny) in series():
        r = ex.evaluate(Nx, Ny, A1, A2)
        rows.append(dict(mode="A: A1, A2 постоянны", series=name, **r))
        a1, a2 = cf.from_relative(Nx, Ny, k, w)
        r = ex.evaluate(Nx, Ny, a1, a2)
        rows.append(dict(mode="B: k, w постоянны", series=name, **r))
    return rows


def plots(rows, base):
    def sel(mode, s):
        d = [r for r in rows if r["mode"].startswith(mode) and r["series"] == s]
        return d

    # Рис. 12.1 — квадратные окна, режимы A и B
    keys = [("tvl_h", "ТВЛ по горизонтали", base["tvl_h"]),
            ("snr_120", "ОСШ (поле 120), дБ", base["snr_120"]),
            ("psnr_lena", "PSNR Lena к эталону, дБ", base["psnr_lena"]),
            ("ssim_lena", "SSIM Lena к эталону", base["ssim_lena"]),
            ("over_h", "Выброс на перепаде, %", base["over_h"]),
            ("halo_h", "Ширина ореола, пикс.", None)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, name, b) in zip(axs.flat, keys):
        for mode, st in (("A", "o-"), ("B", "s-")):
            d = sel(mode, "квадратные")
            ax.plot([r["Nx"] for r in d], [r[key] for r in d], st, ms=4,
                    label="A: A1, A2 постоянны" if mode == "A" else "B: k, w постоянны")
        if b is not None:
            ax.axhline(b, ls="--", color="k", lw=1, label="исходное")
        ax.set_xticks(SIZES)
        ax.set_xlabel("Размер апертуры N (N × N)")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task12_square.png"), dpi=150)
    plt.close(fig)

    # Рис. 12.2 — частотные характеристики (режим B)
    d = sel("B", "квадратные")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    for r in d:
        if r["Nx"] in (3, 5, 7, 11, 15, 25):
            K = cf.aperture_fast(r["Nx"], r["Ny"], r["A1"], r["A2"])
            mp = q.mask_props(K)
            axs[0].plot(mp["f"], mp["Hx"], label=f"{r['Nx']} × {r['Ny']}")
    axs[0].axhline(1, color="k", lw=0.8)
    axs[0].set_xlabel("Пространственная частота, цикл/пиксель")
    axs[0].set_ylabel("|H(f, 0)|")
    axs[0].set_title("Частотная характеристика маски (режим B)")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)
    axs[1].plot([r["Nx"] for r in d], [r["fpeak_h"] for r in d], "o-", label="частота максимума усиления")
    axs[1].plot([r["Nx"] for r in d], [1 / r["Nx"] for r in d], "k:", label="1 / N")
    axs[1].set_xlabel("Размер апертуры N")
    axs[1].set_ylabel("цикл/пиксель")
    ax2 = axs[1].twinx()
    ax2.plot([r["Nx"] for r in d], [r["noise_gain"] for r in d], "s-", color="#C0504D",
             label="усиление шума ||K||")
    ax2.set_ylabel("усиление шума, раз", color="#C0504D")
    axs[1].legend(loc="upper center", fontsize=8)
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task12_freq.png"), dpi=150)
    plt.close(fig)

    # Рис. 12.3 — неравносторонние окна: анизотропия (режим B)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for ax, s in zip(axs.flat, ("Ny = 3, меняем Nx", "Ny = 7, меняем Nx",
                                "Nx = 3, меняем Ny", "Nx = 7, меняем Ny")):
        d = sel("B", s)
        var = "Nx" if "меняем Nx" in s else "Ny"
        x = [r[var] for r in d]
        ax.plot(x, [r["tvl_h"] for r in d], "o-", label="ТВЛ по горизонтали (×640)")
        ax.plot(x, [r["tvl_v"] * ti.W / ti.H for r in d], "s-", label="ТВЛ по вертикали (приведено к 640)")
        ax.axhline(base["tvl_h"], ls="--", color="k", lw=1, label="исходное")
        ax2 = ax.twinx()
        ax2.plot(x, [r["snr_120"] for r in d], "^:", color="#C0504D", label="ОСШ")
        ax2.set_ylabel("ОСШ, дБ", color="#C0504D")
        ax.set_xticks(SIZES)
        ax.set_xlabel(var)
        ax.set_title(f"Режим B, {s}")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task12_unequal.png"), dpi=150)
    plt.close(fig)


def pictures(rows):
    lref, linp = ex.DATA["lena"]
    cref, cinp = ex.DATA["chart"]
    get = {(r["Nx"], r["Ny"]): r for r in rows if r["mode"].startswith("B")}
    shapes = [(3, 3), (7, 7), (15, 15), (25, 25), (25, 3), (3, 25)]
    fig, axs = plt.subplots(2, len(shapes) + 1, figsize=(3.0 * (len(shapes) + 1), 6.4))
    crops = ((linp, (slice(220, 380), slice(220, 380))), (cinp, (slice(165, 315), slice(480, 630))))
    for i, (img, crop) in enumerate(crops):
        axs[i, 0].imshow(img[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axs[i, 0].axis("off")
        for j, a in enumerate(shapes, 1):
            r = get[a]
            out = ex.process(img, *a, r["A1"], r["A2"])
            axs[i, j].imshow(out[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axs[i, j].axis("off")
            if i == 0:
                axs[i, j].set_title(f"{a[0]} × {a[1]}", fontsize=10)
    axs[0, 0].set_title("Вход", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task12_images.png"), dpi=150)
    plt.close(fig)

    # профили перепада (ESF) — рост ширины ореола
    fig, ax = plt.subplots(figsize=(7, 4))
    e = q.slanted_edge(ex.CLEAN[ex.EDGE_V])
    ax.plot(e["pos"], e["esf"], "k--", label="вход")
    for N in (3, 7, 15, 25):
        r = get[(N, N)]
        out = np.clip(cf.crsf_fast(ex.CLEAN, N, N, r["A1"], r["A2"]), 0, 255)
        e = q.slanted_edge(out[ex.EDGE_V])
        ax.plot(e["pos"], e["esf"], label=f"{N} × {N}")
    ax.set_xlim(-20, 20)
    ax.set_xlabel("Расстояние от перепада, пикс.")
    ax.set_ylabel("Нормированная яркость (ESF)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task12_esf.png"), dpi=150)
    plt.close(fig)

    # вид масок
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, a in zip(axs, ((7, 7), (15, 15), (25, 7), (7, 25))):
        r = get[a]
        K = cf.aperture_fast(*a, r["A1"], r["A2"])
        Kd = K.copy()
        Kd[a[1] // 2, a[0] // 2] = np.nan            # центр не показываем — он намного больше
        v = np.nanmax(np.abs(Kd))
        im = ax.imshow(Kd, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"{a[0]} × {a[1]}, K0 = {K[a[1] // 2, a[0] // 2]:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Нормированные маски в режиме B (центральный элемент скрыт)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task12_masks.png"), dpi=150)
    plt.close(fig)


def main():
    A1, A2, k, w = optimum_7x7()
    print(f"Оптимум 7x7 из задачи 11: A1 = {A1:.4g}, A2 = {A2:.4g}, k = {k:.3g}, w = {w:.2f}")
    base = ex.baseline()
    rows = run(A1, A2, k, w)
    ex.save_csv(rows, os.path.join(RES, "task12_all.csv"))
    for r in rows:
        if r["series"] in ("квадратные", "Ny = 3, меняем Nx", "Nx = 3, меняем Ny"):
            print(f"{r['mode'][:1]} {r['Nx']:>2}x{r['Ny']:<2} A1={r['A1']:9.4g} A2={r['A2']:10.4g} "
                  f"k={r['k']:9.3g} | ТВЛ {r['tvl_h']:4.0f}/{r['tvl_v'] * ti.W / ti.H:4.0f} | "
                  f"ОСШ {r['snr_120']:5.1f} | выброс {r['over_h']:5.1f}% ореол {r['halo_h']:4.1f} | "
                  f"PSNR {r['psnr_lena']:5.2f} SSIM {r['ssim_lena']:.3f} | fpeak {r['fpeak_h']:.3f}")
    plots(rows, base)
    pictures(rows)
    print("Готово: results/task12_*")


if __name__ == "__main__":
    main()
