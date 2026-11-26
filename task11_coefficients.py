"""Задача 11: влияние коэффициентов A1 и A2 на качество обработки.
Окна 7x7, 7x3, 3x7, 11x5."""
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
APERS = [(7, 7), (7, 3), (3, 7), (11, 5)]
COL = {(7, 7): "#3C388D", (7, 3): "#E39B2D", (3, 7): "#4BACC6", (11, 5): "#9BBB59"}
KS = np.geomspace(0.1, 100, 31)
WS = {1.0: "w = 1 (только A2)", 0.5: "w = 0,5", 0.0: "w = 0 (только A1)"}
GRID_A1 = [0, 0.5, 1, 2, 3, 5, 10, 20, 50]
GRID_A2 = [0, 1, 5, 21, 50, 100, 200, 500, 1000, 2000]


def lab(a):
    return f"{a[0]} × {a[1]}"


def run():
    rows = []
    for a in APERS:
        for A1 in range(1, 51):
            rows.append(dict(series="S1", **ex.evaluate(*a, A1, 0)))
        for A2 in range(1, 51):
            rows.append(dict(series="S2", **ex.evaluate(*a, 0, A2)))
        for w in WS:
            for k in KS:
                A1, A2 = cf.from_relative(*a, k, w)
                rows.append(dict(series=f"S3 w={w}", **ex.evaluate(*a, A1, A2)))
        print("готово окно", lab(a))
    for A1 in GRID_A1:
        for A2 in GRID_A2:
            if A1 == 0 and A2 == 0:
                continue
            rows.append(dict(series="S4", **ex.evaluate(7, 7, A1, A2)))
    return rows


def pick(rows, base):
    """Оптимальные коэффициенты по разным критериям (по каждому окну)."""
    out = []
    crit = {
        "max ТВЛ при MTF50 <= 0,5 (критерий статей)":
            (lambda r: r["mtf50_h"] <= 0.5 and r["mtf50_v"] <= 0.5, "tvl_h"),
        "max PSNR Lena (близость к эталону)": (lambda r: True, "psnr_lena"),
        "max SSIM Lena (структурное сходство)": (lambda r: True, "ssim_lena"),
        "max ТВЛ при ОСШ >= ОСШвх - 6 дБ и выбросе <= 20 %":
            (lambda r: r["snr_120"] >= base["snr_120"] - 6 and r["over_h"] <= 20, "tvl_h"),
    }
    for a in APERS:
        cand = [r for r in rows if (r["Nx"], r["Ny"]) == a and r["series"] != "S4"]
        for name, (ok, key) in crit.items():
            good = [r for r in cand if ok(r)]
            if not good:
                continue
            b = max(good, key=lambda r: r[key])
            out.append(dict(aperture=lab(a), criterion=name, **b))
    return out


def plots(rows, base):
    def sel(a, s):
        return [r for r in rows if (r["Nx"], r["Ny"]) == a and r["series"] == s]

    # Рис. 11.1 / 11.2 — ТВЛ от A1 и от A2 (как рис. 11 и 13 статьи)
    for s, key, name in (("S1", "A1", "task11_tvl_vs_A1.png"), ("S2", "A2", "task11_tvl_vs_A2.png")):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for a in APERS:
            d = sel(a, s)
            ax[0].plot([r[key] for r in d], [r["tvl_h"] for r in d], "-", color=COL[a], label=lab(a))
            ax[1].plot([r[key] for r in d], [r["snr_120"] for r in d], "-", color=COL[a], label=lab(a))
        ax[0].axhline(base["tvl_h"], ls="--", color="k", lw=1, label="исходное")
        ax[0].axhline(ti.W, ls=":", color="r", lw=1, label="MTF50 = 0,5 цикл/пикс")
        ax[1].axhline(base["snr_120"], ls="--", color="k", lw=1, label="исходное")
        ax[0].set_ylabel("Разрешение по горизонтали, ТВЛ")
        ax[1].set_ylabel("ОСШ (поле 120), дБ")
        for x in ax:
            x.set_xlabel(f"Коэффициент {key}")
            x.grid(alpha=0.3)
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, name), dpi=150)
        plt.close(fig)

    # Рис. 11.3 — показатели от k при разных w (окно 7 x 7)
    keys = [("tvl_h", "ТВЛ по горизонтали", base["tvl_h"]),
            ("snr_120", "ОСШ (поле 120), дБ", base["snr_120"]),
            ("over_h", "Выброс на перепаде, %", base["over_h"]),
            ("psnr_lena", "PSNR Lena к эталону, дБ", base["psnr_lena"]),
            ("ssim_lena", "SSIM Lena к эталону", base["ssim_lena"]),
            ("psnr_chart", "PSNR миры к эталону, дБ", base["psnr_chart"])]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, name, b) in zip(axs.flat, keys):
        for w, wl in WS.items():
            d = sel((7, 7), f"S3 w={w}")
            ax.semilogx([r["k"] for r in d], [r[key] for r in d], "-o", ms=3, label=wl)
        ax.axhline(b, ls="--", color="k", lw=1, label="исходное")
        ax.set_xlabel("Коэффициент усиления перепадов k = S_внеш / S")
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Окно 7 × 7: влияние коэффициентов в относительной форме (k, w)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task11_vs_k_7x7.png"), dpi=150)
    plt.close(fig)

    # Рис. 11.4 — то же для всех окон при w = 1
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for a in APERS:
        d = sel(a, "S3 w=1.0")
        ks = [r["k"] for r in d]
        axs[0].semilogx(ks, [r["tvl_h"] for r in d], color=COL[a], label=f"{lab(a)} гор.")
        axs[0].semilogx(ks, [r["tvl_v"] * ti.W / ti.H for r in d], "--", color=COL[a],
                        label=f"{lab(a)} верт.")
        axs[1].semilogx(ks, [r["snr_120"] for r in d], color=COL[a], label=lab(a))
        axs[2].semilogx(ks, [r["psnr_lena"] for r in d], color=COL[a], label=lab(a))
    axs[0].set_ylabel("ТВЛ (верт. приведены к ширине 640)")
    axs[1].set_ylabel("ОСШ (поле 120), дБ")
    axs[2].set_ylabel("PSNR Lena к эталону, дБ")
    for ax in axs:
        ax.set_xlabel("k")
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task11_apertures_vs_k.png"), dpi=150)
    plt.close(fig)

    # Рис. 11.5 — тепловые карты A1 x A2 (7 x 7)
    g = [r for r in rows if r["series"] == "S4"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, name) in zip(axs, (("tvl_h", "ТВЛ по горизонтали"), ("snr_120", "ОСШ, дБ"),
                                     ("psnr_lena", "PSNR Lena, дБ"))):
        M = np.full((len(GRID_A1), len(GRID_A2)), np.nan)
        for r in g:
            M[GRID_A1.index(r["A1"]), GRID_A2.index(r["A2"])] = r[key]
        im = ax.imshow(M, cmap="viridis", aspect="auto", origin="lower")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="w" if M[i, j] < np.nanmean(M) else "k")
        ax.set_xticks(range(len(GRID_A2)), GRID_A2, fontsize=8)
        ax.set_yticks(range(len(GRID_A1)), GRID_A1, fontsize=8)
        ax.set_xlabel("A2")
        ax.set_ylabel("A1")
        ax.set_title(name)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Окно 7 × 7: сетка коэффициентов A1 × A2")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task11_grid_7x7.png"), dpi=150)
    plt.close(fig)


def images(best):
    """Фрагменты Lena и миры: вход, варианты коэффициентов, эталон (окно 7 x 7)."""
    lref, linp = ex.DATA["lena"]
    cref, cinp = ex.DATA["chart"]
    b77 = {r["criterion"]: r for r in best if r["aperture"] == "7 × 7"}
    psnr = b77["max PSNR Lena (близость к эталону)"]
    tvl = b77["max ТВЛ при MTF50 <= 0,5 (критерий статей)"]
    variants = [("Вход (размытие + шум)", None),
                ("A1 = 1, A2 = 0\n(оптимум статьи по A1)", (1, 0)),
                ("A1 = 0, A2 = 21\n(оптимум статьи по A2)", (0, 21)),
                (f"A1 = {tvl['A1']:.3g}, A2 = {tvl['A2']:.4g}\n(max ТВЛ, MTF50 <= 0,5)", (tvl["A1"], tvl["A2"])),
                (f"A1 = {psnr['A1']:.3g}, A2 = {psnr['A2']:.4g}\n(max PSNR)", (psnr["A1"], psnr["A2"])),
                ("Эталон", "ref")]
    fig, axs = plt.subplots(2, len(variants), figsize=(3.1 * len(variants), 6.6))
    for j, (name, v) in enumerate(variants):
        for i, (ref, inp, crop) in enumerate(((lref, linp, (slice(220, 380), slice(220, 380))),
                                              (cref, cinp, (slice(170, 330), slice(220, 380))))):
            if v is None:
                img = inp
            elif v == "ref":
                img = ref
            else:
                img = ex.process(inp, 7, 7, *v)
            axs[i, j].imshow(img[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axs[i, j].axis("off")
        axs[0, j].set_title(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task11_images_7x7.png"), dpi=150)
    plt.close(fig)
    for name, v in (("A2_21", (0, 21)), ("best_psnr", (psnr["A1"], psnr["A2"]))):
        Image.fromarray(ex.process(linp, 7, 7, *v)).save(os.path.join(RES, f"task11_lena_7x7_{name}.png"))


def mtf_plot(best):
    cref, cinp = ex.DATA["chart"]
    b77 = {r["criterion"]: r for r in best if r["aperture"] == "7 × 7"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    e = q.slanted_edge(cinp[ex.EDGE_V])
    ax.plot(e["freq"], e["mtf"], "k--", label=f"вход, MTF50 = {e['mtf50']:.3f}")
    for name, (A1, A2) in (("A2 = 21", (0, 21)),
                           ("max ТВЛ", (b77["max ТВЛ при MTF50 <= 0,5 (критерий статей)"]["A1"],
                                        b77["max ТВЛ при MTF50 <= 0,5 (критерий статей)"]["A2"])),
                           ("max PSNR", (b77["max PSNR Lena (близость к эталону)"]["A1"],
                                         b77["max PSNR Lena (близость к эталону)"]["A2"]))):
        e = q.slanted_edge(ex.process(cinp, 7, 7, A1, A2)[ex.EDGE_V])
        ax.plot(e["freq"], e["mtf"], label=f"{name}: A1 = {A1:.3g}, A2 = {A2:.4g}, MTF50 = {e['mtf50']:.3f}")
    ax.axhline(0.5, color="gray", lw=0.8)
    ax.axvline(0.5, color="r", lw=0.8, ls=":")
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Пространственная частота, цикл/пиксель")
    ax.set_ylabel("MTF (по горизонтали)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task11_mtf_7x7.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RES, exist_ok=True)
    base = ex.baseline()
    print("Исходное:", {k: round(v, 3) for k, v in base.items()})
    rows = run()
    ex.save_csv(rows, os.path.join(RES, "task11_all.csv"))
    best = pick(rows, base)
    ex.save_csv(best, os.path.join(RES, "task11_optimum.csv"))
    for r in best:
        print(f"{r['aperture']:>7} | {r['criterion']:<50} | A1 = {r['A1']:8.3f} A2 = {r['A2']:9.2f} "
              f"(k = {r['k']:.2f}, w = {r['w']:.2f}) | ТВЛ {r['tvl_h']:.0f}/{r['tvl_v']:.0f} | "
              f"ОСШ {r['snr_120']:.1f} | выброс {r['over_h']:.0f}% | PSNR {r['psnr_lena']:.2f} "
              f"SSIM {r['ssim_lena']:.3f}")
    ex.save_csv([dict(name="исходное", **base)], os.path.join(RES, "task11_baseline.csv"))
    plots(rows, base)
    images(best)
    mtf_plot(best)
    print("Готово: results/task11_*")


if __name__ == "__main__":
    main()
