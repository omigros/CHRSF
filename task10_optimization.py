"""Задача 10: проверка ускоренного фильтра и сравнение времени работы
(исходный ЧРСФ, ускоренный ЧРСФ, обычная двумерная свёртка)."""
import csv
import os
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import crsf_fast as cf  # noqa: E402
import rsf_filter as rf  # noqa: E402  исходная версия фильтра

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
A1, A2 = 1, 21                    # коэффициенты статьи (на время не влияют)
SIZES = [(640, 480), (1280, 720), (3000, 2000)]
SQUARE = [3, 5, 7, 9, 11, 15, 21, 25]
SIDES = list(range(3, 26, 2))
FIXED = [3, 5, 7]


def apertures():
    """Квадратные окна; Nx = 3, 5, 7 при Ny = 3..25; Ny = 3, 5, 7 при Nx = 3..25."""
    out = [(N, N) for N in SQUARE]
    out += [(fx, N) for fx in FIXED for N in SIDES]
    out += [(N, fy) for fy in FIXED for N in SIDES]
    seen, res = set(), []
    for a in out:
        if a not in seen:
            seen.add(a)
            res.append(a)
    return res


def kds(img, K):
    """Классическая двумерная свёртка: Nx*Ny умножений и сложений на отсчёт."""
    Ny, Nx = K.shape
    H, W = img.shape
    xp = np.pad(img.astype(np.float64), ((Ny // 2,) * 2, (Nx // 2,) * 2), mode="edge")
    y = np.zeros((H, W))
    for i in range(Ny):
        for j in range(Nx):
            y += K[i, j] * xp[i:i + H, j:j + W]
    return y


def timeit(f, rep):
    t = []
    for _ in range(rep):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def ops(Nx, Ny):
    """Операций на отсчёт: (умножения, сложения) для КДС, ЧРСФ исх., ЧРСФ опт."""
    n0 = cf.n_recirculators(Nx, Ny, shared=False)
    n1 = cf.n_recirculators(Nx, Ny, shared=True)
    return dict(kds_mul=Nx * Ny, kds_add=Nx * Ny - 1,
                orig_mul=3 + 1, orig_add=2 * n0 + 2,   # 3 умножения + деление на S
                opt_mul=3, opt_add=2 * n1 + 2,
                rec_orig=n0, rec_opt=n1)


def check(img):
    rows = []
    for Nx, Ny in apertures():
        K = cf.aperture_fast(Nx, Ny, A1, A2)
        y_ref = rf.crsf(img.astype(float), Nx, Ny, A1, A2)
        y_opt = cf.crsf_fast(img, Nx, Ny, A1, A2)
        y_kds = kds(img, K)
        sc = np.abs(y_ref).max()
        rows.append(dict(Nx=Nx, Ny=Ny,
                         opt_vs_orig=np.abs(y_opt - y_ref).max() / sc,
                         opt_vs_kds=np.abs(y_opt - y_kds).max() / sc,
                         mask_equal=bool(np.allclose(rf.aperture(Nx, Ny, A1, A2), K))))
    return rows


def main():
    os.makedirs(RES, exist_ok=True)
    lena = Image.open(os.path.join(HERE, "..", "lena.png")).convert("L")
    imgs = {s: np.array(lena.resize(s, Image.BICUBIC)) for s in SIZES}

    # 1. корректность
    chk = check(imgs[(640, 480)])
    with open(os.path.join(RES, "task10_correctness.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(chk[0]), delimiter=";")
        w.writeheader()
        w.writerows(chk)
    worst = max(max(r["opt_vs_orig"], r["opt_vs_kds"]) for r in chk)
    print(f"Корректность: макс. относительное расхождение {worst:.1e}, "
          f"маски совпадают: {all(r['mask_equal'] for r in chk)}")

    # 2-3. операции и время
    apers = apertures()
    rows = []
    for (Wd, Ht) in SIZES:
        img = imgs[(Wd, Ht)]
        big = Wd * Ht > 2e6
        for Nx, Ny in apers:
            K = cf.aperture_fast(Nx, Ny, A1, A2)
            t_kds = timeit(lambda: kds(img, K), 2 if big else 3)
            t_orig = timeit(lambda: rf.crsf(img.astype(float), Nx, Ny, A1, A2), 2 if big else 4)
            t_opt = timeit(lambda: cf.crsf_fast(img, Nx, Ny, A1, A2), 4 if big else 8)
            r = dict(image=f"{Wd}x{Ht}", Nx=Nx, Ny=Ny, t_kds=t_kds, t_orig=t_orig, t_opt=t_opt,
                     speedup_opt_vs_orig=t_orig / t_opt, speedup_opt_vs_kds=t_kds / t_opt,
                     speedup_orig_vs_kds=t_kds / t_orig, **ops(Nx, Ny))
            rows.append(r)
            print(f"{Wd}x{Ht} {Nx:>2}x{Ny:<2}  КДС {t_kds:7.3f} c  ЧРСФ {t_orig:6.3f} c  "
                  f"ЧРСФ-опт {t_opt:6.3f} c  (x{t_orig / t_opt:.1f} к исх., x{t_kds / t_opt:.0f} к КДС)")
    with open(os.path.join(RES, "task10_timing.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}".replace(".", ",") if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(RES, "task10_platform.txt"), "w", encoding="utf-8") as f:
        f.write(f"{platform.platform()}\n{platform.processor()}\nPython {platform.python_version()}, "
                f"numpy {np.__version__}\n")

    # графики: время от размера квадратной апертуры
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (Wd, Ht) in zip(axs, SIZES):
        sel = [r for r in rows if r["image"] == f"{Wd}x{Ht}" and r["Nx"] == r["Ny"]]
        N = [r["Nx"] for r in sel]
        ax.semilogy(N, [r["t_kds"] for r in sel], "o-", color="#C0504D", label="КДС")
        ax.semilogy(N, [r["t_orig"] for r in sel], "s-", color="#7F7F7F", label="ЧРСФ (исходный)")
        ax.semilogy(N, [r["t_opt"] for r in sel], "^-", color="#3C388D", label="ЧРСФ (оптимизированный)")
        ax.set_title(f"Изображение {Wd} × {Ht}")
        ax.set_xlabel("Размер апертуры N (N × N)")
        ax.set_ylabel("Время обработки, с")
        ax.set_xticks(N)
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task10_time_vs_aperture.png"), dpi=150)

    # число операций на отсчёт
    N = np.arange(3, 26, 2)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(N, [2 * n * n - 1 for n in N], "o-", color="#C0504D", label="КДС (умн. + слож.)")
    ax.plot(N, [ops(n, n)["orig_mul"] + ops(n, n)["orig_add"] for n in N], "s-", color="#7F7F7F",
            label="ЧРСФ исходный")
    ax.plot(N, [ops(n, n)["opt_mul"] + ops(n, n)["opt_add"] for n in N], "^-", color="#3C388D",
            label="ЧРСФ оптимизированный")
    ax.set_yscale("log")
    ax.set_xlabel("Размер апертуры N (N × N)")
    ax.set_ylabel("Операций на один отсчёт")
    ax.set_xticks(N)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "task10_ops.png"), dpi=150)
    print("Готово: results/task10_*")


if __name__ == "__main__":
    main()
