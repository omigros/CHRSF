"""Обработка тестовых изображений и измерение всех показателей."""
import numpy as np

import crsf_fast as cf
import quality as q
import test_images as ti

DATA = ti.load()
EDGE_V, EDGE_H, FLATS = ti.rois()
# мира без шума - для измерения выброса и ореола
CLEAN = q.gauss_blur(DATA["chart"][0].astype(float), ti.BLUR)


def process(img, Nx, Ny, A1, A2):
    return cf.to_uint8(cf.crsf_fast(img, Nx, Ny, A1, A2))


def evaluate(Nx, Ny, A1, A2, with_lena=True):
    """Все показатели для окна Nx x Ny и коэффициентов A1, A2."""
    ref, inp = DATA["chart"]
    out = process(inp, Nx, Ny, A1, A2)
    eh = q.slanted_edge(out[EDGE_V])          # вертикальный перепад -> MTF по горизонтали
    ev = q.slanted_edge(out[EDGE_H].T)        # горизонтальный перепад -> MTF по вертикали
    clean = np.clip(cf.crsf_fast(CLEAN, Nx, Ny, A1, A2), 0, 255)
    sh = q.slanted_edge(clean[EDGE_V])
    sv = q.slanted_edge(clean[EDGE_H].T)
    K = cf.aperture_fast(Nx, Ny, A1, A2)
    mp = q.mask_props(K)
    k, w = cf.to_relative(Nx, Ny, A1, A2)
    r = dict(Nx=Nx, Ny=Ny, A1=A1, A2=A2, k=k, w=w,
             K0=float(K[Ny // 2, Nx // 2]),
             mtf50_h=eh["mtf50"], mtf50_v=ev["mtf50"],
             tvl_h=q.tvl(eh["mtf50"], ti.W), tvl_v=q.tvl(ev["mtf50"], ti.H),
             over_h=sh["overshoot"], over_v=sv["overshoot"],
             halo_h=sh["halo"], halo_v=sv["halo"],
             snr_50=q.snr_db(out[FLATS[50]]), snr_120=q.snr_db(out[FLATS[120]]),
             snr_220=q.snr_db(out[FLATS[220]]),
             sko_chart=q.sko(inp, out), posh_chart=q.posh_paper(inp, out),
             psnr_chart=q.psnr(ref, out), ssim_chart=q.ssim(ref, out),
             mean_shift=float(out.mean() - inp.astype(float).mean()),
             noise_gain=mp["gain_noise"], peak_h=mp["peak_x"], fpeak_h=mp["fpeak_x"],
             peak_v=mp["peak_y"], fpeak_v=mp["fpeak_y"])
    if with_lena:
        lref, linp = DATA["lena"]
        lout = process(linp, Nx, Ny, A1, A2)
        r.update(psnr_lena=q.psnr(lref, lout), ssim_lena=q.ssim(lref, lout),
                 sko_lena=q.sko(linp, lout), posh_lena=q.posh_paper(linp, lout))
    return r


def baseline():
    """Показатели входного (необработанного) изображения."""
    ref, inp = DATA["chart"]
    lref, linp = DATA["lena"]
    eh = q.slanted_edge(inp[EDGE_V])
    ev = q.slanted_edge(inp[EDGE_H].T)
    eh["overshoot"] = q.slanted_edge(CLEAN[EDGE_V])["overshoot"]
    return dict(mtf50_h=eh["mtf50"], mtf50_v=ev["mtf50"],
                tvl_h=q.tvl(eh["mtf50"], ti.W), tvl_v=q.tvl(ev["mtf50"], ti.H),
                over_h=eh["overshoot"], snr_50=q.snr_db(inp[FLATS[50]]),
                snr_120=q.snr_db(inp[FLATS[120]]), snr_220=q.snr_db(inp[FLATS[220]]),
                psnr_chart=q.psnr(ref, inp), ssim_chart=q.ssim(ref, inp),
                psnr_lena=q.psnr(lref, linp), ssim_lena=q.ssim(lref, linp))


def save_csv(rows, path):
    import csv
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}".replace(".", ",") if isinstance(v, (float, np.floating)) else v)
                        for k, v in r.items()})
