"""Показатели качества изображения: ТВЛ, ОСШ, СКО, PSNR, SSIM, выброс, ореол."""
import numpy as np


# ---------- размытие
def gauss_kernel(sigma, radius=None):
    radius = int(np.ceil(3.5 * sigma)) if radius is None else radius
    t = np.arange(-radius, radius + 1)
    g = np.exp(-t ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def sep_filter(img, k, mode="reflect"):
    """Свёртка с ядром k по строкам, затем по столбцам."""
    r = len(k) // 2
    x = np.pad(np.asarray(img, float), r, mode=mode)
    H, W = img.shape
    t = np.zeros((H + 2 * r, W))
    for j, kj in enumerate(k):
        t += kj * x[:, j:j + W]
    y = np.zeros((H, W))
    for i, ki in enumerate(k):
        y += ki * t[i:i + H, :]
    return y


def gauss_blur(img, sigma):
    return sep_filter(img, gauss_kernel(sigma))


# ---------- разрешение (ТВЛ)
def slanted_edge(roi, oversample=4):
    """Частотно-контрастная характеристика (MTF) по наклонному перепаду.
    roi - фрагмент с почти вертикальной границей тёмного и светлого."""
    roi = np.asarray(roi, float)
    rows, cols = roi.shape
    # 1) находим положение границы в каждой строке и проводим через них прямую
    rs = np.ones(5) / 5
    sm = np.apply_along_axis(lambda c: np.convolve(c, rs, mode="same"), 0, roi)
    d = np.abs(np.diff(sm, axis=1))
    xs = np.arange(cols - 1) + 0.5
    yy = np.arange(rows)
    guess = np.full(rows, float(np.argmax(d.sum(0))) + 0.5)
    for _ in range(2):
        w = (np.abs(xs[None, :] - guess[:, None]) <= 8).astype(float)
        dw = d * w
        cent = (dw * xs).sum(1) / np.maximum(dw.sum(1), 1e-12)
        a, b = np.polyfit(yy[2:-2], cent[2:-2], 1)
        guess = a * yy + b
    # 2) профиль перепада: яркость в зависимости от расстояния до прямой (шаг 1/4 пикселя)
    X, Y = np.meshgrid(np.arange(cols) + 0.5, yy)
    dist = X - (a * Y + b)
    half = cols // 2 - 2
    keep = np.abs(dist) < half
    idx = np.floor((dist[keep] + half) * oversample).astype(int)
    nb = 2 * half * oversample
    cnt = np.bincount(idx, minlength=nb)[:nb]
    s = np.bincount(idx, weights=roi[keep], minlength=nb)[:nb]
    good = cnt > 0
    pos = (np.arange(nb) + 0.5) / oversample - half
    esf = np.interp(pos, pos[good], s[good] / cnt[good])
    if esf[-nb // 8:].mean() < esf[:nb // 8].mean():   # профиль: от тёмного к светлому
        esf = esf[::-1]
    # 3) производная профиля -> спектр (БПФ) -> MTF
    lsf = np.gradient(esf)
    c = np.argmax(np.abs(lsf))
    n = np.arange(nb)
    win = 0.54 + 0.46 * np.cos(np.pi * np.clip((n - c) / (nb / 2), -1, 1))
    spec = np.abs(np.fft.rfft(lsf * win))
    mtf = spec / spec[0]
    freq = np.fft.rfftfreq(nb, d=1.0 / oversample)
    corr = np.sinc(2 * freq / oversample)               # поправка на численную производную
    mtf = mtf / np.where(np.abs(corr) > 0.1, corr, 0.1)
    sel = freq <= 1.0
    freq, mtf = freq[sel], mtf[sel]
    below = np.nonzero(mtf < 0.5)[0]
    if len(below):
        i = below[0]
        f0, f1, m0, m1 = freq[i - 1], freq[i], mtf[i - 1], mtf[i]
        mtf50 = f0 + (m0 - 0.5) * (f1 - f0) / (m0 - m1)
    else:
        mtf50 = float(freq[-1])
    # 4) выброс и ширина ореола по профилю
    q = nb // 6
    lo, hi = esf[:q].mean(), esf[-q:].mean()
    e = (esf - lo) / (hi - lo)
    over = max(0.0, e.max() - 1) * 100
    under = max(0.0, -e.min()) * 100
    edge = pos[np.argmin(np.abs(e - 0.5))]
    dev = np.abs(np.where(pos > edge, e - 1, e)) > 0.02    # отклонение больше 2 %
    far = np.abs(pos - edge) > 1.5
    halo = float(np.abs(pos[dev & far] - edge).max()) if np.any(dev & far) else 0.0
    return dict(freq=freq, mtf=mtf, pos=pos, esf=e, mtf50=float(mtf50),
                overshoot=over, undershoot=under, halo=halo)


def tvl(mtf50, limit_px):
    """ТВЛ = (CPP / 0.5) * предельное разрешение (число пикселей)."""
    return mtf50 / 0.5 * limit_px


# ---------- шум и отличие от эталона
def snr_db(region):
    """ОСШ = 20 lg(M / sigma) по однородной области."""
    r = np.asarray(region, float)
    return 20 * np.log10(r.mean() / max(r.std(), 1e-9))


def sko(x, y):
    """СКО между входным x и выходным y изображениями."""
    return float(np.sqrt(np.mean((np.asarray(x, float) - np.asarray(y, float)) ** 2)))


def posh_paper(x, y):
    """ПОСШ = 20 lg(M / СКО), M - средняя яркость выхода."""
    return 20 * np.log10(np.mean(y) / max(sko(x, y), 1e-9))


def psnr(ref, y):
    mse = np.mean((np.asarray(ref, float) - np.asarray(y, float)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def ssim(ref, y):
    """Индекс структурного сходства SSIM."""
    x, y = np.asarray(ref, float), np.asarray(y, float)
    k = gauss_kernel(1.5, 5)
    f = lambda a: sep_filter(a, k)                      # noqa: E731
    mx, my = f(x), f(y)
    sxx = f(x * x) - mx ** 2
    syy = f(y * y) - my ** 2
    sxy = f(x * y) - mx * my
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return float(m.mean())


# ---------- свойства маски
def mask_props(K):
    """Частотная характеристика маски вдоль осей и усиление шума."""
    K = np.asarray(K, float)
    f = np.linspace(0, 0.5, 201)
    kx = K.sum(0)                                   # сечение H(fx, 0)
    ky = K.sum(1)                                   # сечение H(0, fy)
    def resp(k):                                    # noqa: E306
        n = np.arange(len(k)) - len(k) // 2
        return np.abs(np.cos(2 * np.pi * np.outer(f, n)) @ k)
    Hx, Hy = resp(kx), resp(ky)
    return dict(f=f, Hx=Hx, Hy=Hy,
                gain_noise=float(np.sqrt((K ** 2).sum())),
                peak_x=float(Hx.max()), fpeak_x=float(f[Hx.argmax()]),
                peak_y=float(Hy.max()), fpeak_y=float(f[Hy.argmax()]),
                nyq_x=float(Hx[-1]), nyq_y=float(Hy[-1]))
