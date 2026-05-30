import numpy as np
import cv2
import sys
import time
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Системные настройки
# ------------------------------------------------------------
sys.stdout.reconfigure(encoding='utf-8')
np.set_printoptions(precision=4, suppress=True, linewidth=140)

# ------------------------------------------------------------
# Быстрые рециркуляторы
# ------------------------------------------------------------
def SR_fast(img, M):
    cs = np.cumsum(img, axis=1)
    out = cs.copy()
    out[:, M:] -= cs[:, :-M]
    return out

def KR_fast(img, M):
    cs = np.cumsum(img, axis=0)
    out = cs.copy()
    out[M:, :] -= cs[:-M, :]
    return out

def cascade(img, Mx, My):
    return KR_fast(SR_fast(img, Mx), My)

def CHRSF_lowpass(img, Mx, My):
    out = img.astype(np.float64)
    for _ in range(4):
        out = cascade(out, Mx, My)
    return out

# ------------------------------------------------------------
# 1D маска (4 каскада)
# ------------------------------------------------------------
def build_1d_mask(M):
    k = np.array([1.0])
    for _ in range(4):
        k = np.convolve(k, np.ones(M))
    return k

# ------------------------------------------------------------
# АДАПТИВНАЯ корректирующая маска
# ------------------------------------------------------------
def build_correction_mask(Ax_real, Ay_real):

    # уменьшаем примерно в 2 раза
    Ax_c = max(3, (Ax_real // 2) | 1)
    Ay_c = max(3, (Ay_real // 2) | 1)

    Mx_c = (Ax_c - 1) // 4 + 1
    My_c = (Ay_c - 1) // 4 + 1

    Rx_c = build_1d_mask(Mx_c)
    Ry_c = build_1d_mask(My_c)

    corr = np.outer(Ry_c, Rx_c)

    return corr

# ------------------------------------------------------------
# Формирование масок
# ------------------------------------------------------------
def build_masks(Ax_req, Ay_req):

    Mx = (Ax_req - 1) // 4 + 1
    My = (Ay_req - 1) // 4 + 1

    Ax_real = 4 * (Mx - 1) + 1
    Ay_real = 4 * (My - 1) + 1

    Rx = build_1d_mask(Mx)
    Ry = build_1d_mask(My)

    base = np.outer(Ry, Rx)

    # -------- корректирующая маска --------
    corr = build_correction_mask(Ax_real, Ay_real)

    # масштабирование
    k_corr = np.sum(base) / np.sum(corr)

    # -------- финальная маска --------
    final = -base.copy()

    cy = final.shape[0] // 2
    cx = final.shape[1] // 2

    hc, wc = corr.shape
    dy = hc // 2
    dx = wc // 2

    final[cy-dy:cy+dy+1, cx-dx:cx+dx+1] += k_corr * corr

    return base, corr, final, k_corr, Ax_real, Ay_real

# ------------------------------------------------------------
# Повышение чёткости (НЕ ТРОГАЕМ — ОНО ПРАВИЛЬНОЕ)
# ------------------------------------------------------------
def CHRSF_sharpen(img, Ax_req, Ay_req, alpha):

    Mx = (Ax_req - 1) // 4 + 1
    My = (Ay_req - 1) // 4 + 1

    low = CHRSF_lowpass(img, Mx, My)

    low *= np.mean(img) / (np.mean(low) + 1e-9)

    high = img - low
    result = img + alpha * high

    return np.clip(result, 0, 255).astype(np.uint8)

# ------------------------------------------------------------
# Основная программа
# ------------------------------------------------------------
print("=== МОДЕРНИЗИРОВАННЫЙ ЧРСФ С АДАПТИВНОЙ КОРРЕКЦИЕЙ ===")

Ax_req = int(input("Введите Ax (нечётная): "))
Ay_req = int(input("Введите Ay (нечётная): "))
alpha  = float(input("Введите коэффициент усиления: "))

base, corr, final, k_corr, Ax_real, Ay_real = build_masks(Ax_req, Ay_req)

print("\nЗаданная апертура: {} x {}".format(Ax_req, Ay_req))
print("Реальная апертура: {} x {}".format(Ax_real, Ay_real))
print("Размер корректирующей маски: {} x {}".format(corr.shape[1], corr.shape[0]))
print("Коэффициент k = {:.6f}".format(k_corr))
print("Сумма финальной маски = {:.6f}".format(np.sum(final)))

# ------------------------------------------------------------
# Загрузка изображения
# ------------------------------------------------------------
img = cv2.imread("lena.png", cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("Файл lena.png не найден")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ------------------------------------------------------------
# Фильтрация
# ------------------------------------------------------------
start = time.perf_counter()

result = np.zeros_like(img)
for c in range(3):
    result[:,:,c] = CHRSF_sharpen(
        img[:,:,c],
        Ax_req,
        Ay_req,
        alpha
    )

end = time.perf_counter()

print("Время обработки: {:.4f} сек".format(end - start))

# ------------------------------------------------------------
# Сохранение
# ------------------------------------------------------------
filename = f"result_{Ax_req}_{Ay_req}.png"
cv2.imwrite(filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("Сохранено как:", filename)

# ------------------------------------------------------------
# Визуализация
# ------------------------------------------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Исходное")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(result)
plt.title("После ЧРСФ")
plt.axis("off")

plt.show()