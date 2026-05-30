import numpy as np
import cv2
import sys
import time
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

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
    corr = build_correction_mask(Ax_real, Ay_real)
    k_corr = np.sum(base) / np.sum(corr)
    final = -base.copy()
    cy = final.shape[0] // 2
    cx = final.shape[1] // 2
    hc, wc = corr.shape
    dy = hc // 2
    dx = wc // 2
    final[cy-dy:cy+dy+1, cx-dx:cx+dx+1] += k_corr * corr
    return base, corr, final, k_corr, Ax_real, Ay_real

# ------------------------------------------------------------
# Повышение чёткости
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
# Экспорт масок в Excel
# ------------------------------------------------------------
def export_excel(Ax_req, Ay_req, Mx, My, Ax_real, Ay_real,
                 base, corr, final, k_corr):

    def st(color):
        return PatternFill("solid", start_color=color, fgColor=color)

    def brd():
        s = Side(style="thin", color="AAAAAA")
        return Border(left=s, right=s, top=s, bottom=s)

    def write_matrix(ws, matrix, start_row, start_col, fmt="0"):
        rows, cols = matrix.shape
        for c in range(cols):
            max_len = max(len(str(int(abs(matrix[r, c])))) for r in range(rows))
            ws.column_dimensions[get_column_letter(start_col + c)].width = max(max_len + 3, 7)
        for r in range(rows):
            for c in range(cols):
                val = float(matrix[r, c])
                cell = ws.cell(row=start_row + r, column=start_col + c, value=val)
                cell.number_format = fmt
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = brd()
                cell.font = Font(name="Arial", size=9)
                if val > 0:
                    cell.fill = st("D5F5E3")
                elif val < 0:
                    cell.fill = st("FADBD8")
                else:
                    cell.fill = st("FDFEFE")

    def title(ws, row, col, text, bg="1A5276", span=1):
        cell = ws.cell(row=row, column=col, value=text)
        cell.font = Font(name="Arial", bold=True, size=11, color="FFFFFF")
        cell.fill = st(bg)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        if span > 1:
            ws.merge_cells(start_row=row, start_column=col,
                           end_row=row, end_column=col + span - 1)

    def info(ws, row, col, label, value):
        c1 = ws.cell(row=row, column=col, value=label)
        c1.font = Font(name="Arial", size=10)
        c1.fill = st("EBF5FB")
        c1.border = brd()
        c2 = ws.cell(row=row, column=col + 1, value=value)
        c2.font = Font(name="Arial", bold=True, size=10)
        c2.alignment = Alignment(horizontal="center")
        c2.border = brd()

    # --- 1D шаги свёртки ---
    def write_1d_steps(ws, M, axis_label, start_row, bg_fill):
        steps = [np.array([1.0])]
        k = np.array([1.0])
        for _ in range(4):
            k = np.convolve(k, np.ones(M))
            steps.append(k.copy())

        max_len = len(steps[-1])
        title(ws, start_row, 1, axis_label, bg="2980B9", span=max_len + 2)
        start_row += 1

        step_labels = [
            "Входной импульс [1]",
            f"Каскад 1  (*[1]*{M})",
            "Каскад 2",
            "Каскад 3",
            f"Каскад 4  (итог, {4*(M-1)+1} эл.)",
        ]
        ws.column_dimensions["A"].width = 26
        ws.column_dimensions[get_column_letter(max_len + 2)].width = 10

        for i, (step, lbl) in enumerate(zip(steps, step_labels)):
            bold = (i == 4)
            c = ws.cell(row=start_row + i, column=1, value=lbl)
            c.font = Font(name="Arial", bold=bold, size=10)
            c.fill = st("EBF5FB")
            c.border = brd()
            offset = (max_len - len(step)) // 2
            for j, v in enumerate(step):
                cell = ws.cell(row=start_row + i, column=2 + offset + j, value=int(v))
                cell.font = Font(name="Arial", bold=bold, size=10)
                cell.alignment = Alignment(horizontal="center")
                cell.border = brd()
                cell.fill = st(bg_fill)
            sum_cell = ws.cell(row=start_row + i, column=max_len + 2, value=int(np.sum(step)))
            sum_cell.font = Font(name="Arial", bold=True, size=10)
            sum_cell.alignment = Alignment(horizontal="center")
            sum_cell.border = brd()
            sum_cell.fill = st("FEF9E7")

        return start_row + len(steps) + 1

    wb = Workbook()

    # ── Лист 1: Параметры ──────────────────────────────────
    ws1 = wb.active
    ws1.title = "Параметры"
    ws1.column_dimensions["A"].width = 35
    ws1.column_dimensions["B"].width = 20

    title(ws1, 1, 1, "ЧРСФ — Параметры и формулы расчёта", span=2)
    r = 3
    title(ws1, r, 1, "Входные данные", bg="2980B9", span=2); r += 1
    info(ws1, r, 1, "Ax_req  (запрошенная апертура X)", Ax_req); r += 1
    info(ws1, r, 1, "Ay_req  (запрошенная апертура Y)", Ay_req); r += 1

    r += 1
    title(ws1, r, 1, "Шаг 1 — Расчёт M (размер окна рециркулятора)", bg="1E8449", span=2); r += 1
    info(ws1, r, 1, f"Mx = (Ax_req − 1) / 4 + 1 = ({Ax_req}−1)/4+1", Mx); r += 1
    info(ws1, r, 1, f"My = (Ay_req − 1) / 4 + 1 = ({Ay_req}−1)/4+1", My); r += 1

    r += 1
    title(ws1, r, 1, "Шаг 2 — Реальная апертура (длина ядра после 4 свёрток)", bg="1E8449", span=2); r += 1
    info(ws1, r, 1, f"Ax_real = 4·(Mx−1)+1 = 4·({Mx}−1)+1", Ax_real); r += 1
    info(ws1, r, 1, f"Ay_real = 4·(My−1)+1 = 4·({My}−1)+1", Ay_real); r += 1

    r += 1
    Ax_c = max(3, (Ax_real // 2) | 1)
    Ay_c = max(3, (Ay_real // 2) | 1)
    Mx_c = (Ax_c - 1) // 4 + 1
    My_c = (Ay_c - 1) // 4 + 1
    title(ws1, r, 1, "Шаг 3 — Корректирующая маска", bg="7D3C98", span=2); r += 1
    info(ws1, r, 1, f"Ax_c = (Ax_real//2)|1 = ({Ax_real}//2)|1", Ax_c); r += 1
    info(ws1, r, 1, f"Ay_c = (Ay_real//2)|1 = ({Ay_real}//2)|1", Ay_c); r += 1
    info(ws1, r, 1, f"Mx_c = (Ax_c−1)/4+1 = ({Ax_c}−1)/4+1", Mx_c); r += 1
    info(ws1, r, 1, f"My_c = (Ay_c−1)/4+1 = ({Ay_c}−1)/4+1", My_c); r += 1

    r += 1
    title(ws1, r, 1, "Шаг 4 — Коэффициент нормировки k", bg="922B21", span=2); r += 1
    info(ws1, r, 1, "Σ (основная маска)", int(np.sum(base))); r += 1
    info(ws1, r, 1, "Σ (корректирующая маска)", int(np.sum(corr))); r += 1
    info(ws1, r, 1, f"k = Σ_base / Σ_corr = {int(np.sum(base))}/{int(np.sum(corr))}", round(k_corr, 6)); r += 1
    info(ws1, r, 1, "Σ финальной маски (должно быть ≈ 0)", round(float(np.sum(final)), 8)); r += 1

    r += 1
    title(ws1, r, 1, "Шаг 5 — Повышение чёткости (нерезкое маскирование)", bg="1A5276", span=2); r += 1
    for lbl in [
        ("low = ЧРСФ_НЧ(img, Mx, My)", "размытая копия изображения"),
        ("high = img − low",            "высокочастотные детали"),
        ("result = img + α·high",       "итог: усиленные края"),
    ]:
        info(ws1, r, 1, lbl[0], lbl[1]); r += 1

    # ── Лист 2: 1D свёртки ─────────────────────────────────
    ws2 = wb.create_sheet("1D свёртки")
    r2 = write_1d_steps(ws2, Mx, f"Ось X  (Mx={Mx})", 1, "D6EAF8")
    r2 += 1
    write_1d_steps(ws2, My, f"Ось Y  (My={My})", r2, "D5F5E3")

    # ── Лист 3: Основная маска ─────────────────────────────
    ws3 = wb.create_sheet("Основная маска")
    title(ws3, 1, 1, f"Основная маска {Ay_real}×{Ax_real}  (B-сплайн 4-го порядка, все значения положительные)",
          span=Ax_real + 1)
    write_matrix(ws3, base, 2, 1, fmt="0")
    ws3.cell(row=Ay_real + 3, column=1,
             value=f"Σ = {int(np.sum(base))}").font = Font(name="Arial", bold=True)

    # ── Лист 4: Инвертированная маска ─────────────────────
    ws4 = wb.create_sheet("Инвертированная маска")
    title(ws4, 1, 1, f"Инвертированная маска {Ay_real}×{Ax_real}  (−base, все значения отрицательные)",
          bg="922B21", span=Ax_real + 1)
    write_matrix(ws4, -base, 2, 1, fmt="0")
    ws4.cell(row=Ay_real + 3, column=1,
             value=f"Σ = {int(-np.sum(base))}").font = Font(name="Arial", bold=True)

    # ── Лист 5: Корректирующая маска ──────────────────────
    ws5 = wb.create_sheet("Корректирующая маска")
    rows_c, cols_c = corr.shape
    title(ws5, 1, 1, f"Корректирующая маска {rows_c}×{cols_c}  (k = {k_corr:.4f})",
          bg="1E8449", span=cols_c + 1)
    write_matrix(ws5, corr, 2, 1, fmt="0")
    ws5.cell(row=rows_c + 3, column=1,
             value=f"Σ = {int(np.sum(corr))}   |   k·Σ = {k_corr*np.sum(corr):.1f}").font = Font(name="Arial", bold=True)

    # ── Лист 6: Финальная маска ───────────────────────────
    ws6 = wb.create_sheet("Финальная маска")
    title(ws6, 1, 1,
          f"Финальная маска {Ay_real}×{Ax_real}  (−base + k·corr в центре)  |  Σ ≈ 0",
          bg="2C3E50", span=Ax_real + 1)
    write_matrix(ws6, final, 2, 1, fmt="0.0")
    ws6.cell(row=Ay_real + 3, column=1,
             value=f"Σ = {np.sum(final):.6f}").font = Font(name="Arial", bold=True)

    out = f"CHRSF_masks_{Ax_req}x{Ay_req}.xlsx"
    wb.save(out)
    print("Excel сохранён:", out)


# ------------------------------------------------------------
# Основная программа
# ------------------------------------------------------------
print("=== МОДЕРНИЗИРОВАННЫЙ ЧРСФ С АДАПТИВНОЙ КОРРЕКЦИЕЙ ===")

Ax_req = int(input("Введите Ax (нечётная): "))
Ay_req = int(input("Введите Ay (нечётная): "))
alpha  = float(input("Введите коэффициент усиления: "))

base, corr, final, k_corr, Ax_real, Ay_real = build_masks(Ax_req, Ay_req)

Mx = (Ax_req - 1) // 4 + 1
My = (Ay_req - 1) // 4 + 1

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
    result[:,:,c] = CHRSF_sharpen(img[:,:,c], Ax_req, Ay_req, alpha)

end = time.perf_counter()
print("Время обработки: {:.4f} сек".format(end - start))

# ------------------------------------------------------------
# Сохранение изображения
# ------------------------------------------------------------
filename = f"result_{Ax_req}_{Ay_req}.png"
cv2.imwrite(filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("Сохранено как:", filename)

# ------------------------------------------------------------
# Экспорт масок в Excel
# ------------------------------------------------------------
export_excel(Ax_req, Ay_req, Mx, My, Ax_real, Ay_real,
             base, corr, final, k_corr)

# ------------------------------------------------------------
# Визуализация масок
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Маски фильтра", fontsize=13)

im0 = axes[0].imshow(base, cmap="Blues")
axes[0].set_title(f"Основная маска {Ay_real}×{Ax_real}")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(corr, cmap="Greens")
axes[1].set_title(f"Корректирующая маска {corr.shape[0]}×{corr.shape[1]}")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(final, cmap="RdBu")
axes[2].set_title(f"Финальная маска {Ay_real}×{Ax_real}")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Визуализация изображений
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Исходное")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("После ЧРСФ")
plt.axis("off")

plt.show()
