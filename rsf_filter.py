"""ЧРСФ: 4-каскадный рекурсивно-сепарабельный фильтр."""
import numpy as np
from PIL import Image


# ===== Этап 1. Параметры фильтра
def outer_orders(N):
    """Длины 4 рециркуляторов внешней маски."""
    s = N + 3                     # сумма длин
    M = [s // 4] * 4              # поровну на 4 звена
    for i in range(s % 4):        # остаток -- первым
        M[i] += 1
    return M


def inner_size(N):
    """Размер внутренней маски: 3->1, 5->1, 7->3, 9->3."""
    return 2 * ((N - 3) // 4) + 1


def inner_orders(n):
    """Длины 2 рециркуляторов внутренней маски."""
    m = (n + 1) // 2
    return [m, m]


def params(Nx, Ny, A1, A2):
    """Все числа фильтра для окна Nx x Ny."""
    Mx, My = outer_orders(Nx), outer_orders(Ny)
    nx, ny = inner_size(Nx), inner_size(Ny)
    mx, my = inner_orders(nx), inner_orders(ny)
    ax, ay = (Nx - nx) // 2, (Ny - ny) // 2
    bx, by = (Nx - 1) // 2, (Ny - 1) // 2
    S_out = np.prod(Mx) * np.prod(My)   # сумма внешней
    S_in = np.prod(mx) * np.prod(my)    # сумма внутренней
    X = S_out / S_in                    # уравнивание
    S = A1 * S_in + A2                  # сумма итоговой
    return Mx, My, mx, my, ax, ay, bx, by, X, S


# ===== Этап 2. Рециркуляторы
def sr(x, M):
    """Строчный рециркулятор: сумма M соседей."""
    H, W = x.shape                # высота и ширина
    y = np.zeros((H, W))          # пустой результат
    acc = np.zeros(H)             # сумма окна в строках
    for n in range(W):            # столбцы слева направо
        acc = acc + x[:, n]       # + вошедший
        if n >= M:
            acc = acc - x[:, n - M]   # - вышедший
        y[:, n] = acc             # записали сумму
    return y


def kr(x, M):
    """Кадровый рециркулятор: то же по столбцам."""
    return sr(x.T, M).T


# ===== Этап 3. Внешняя маска (ветвь 2)
def outer_mask(x, Mx, My):
    """4 строчных + 4 кадровых рециркулятора."""
    for M in Mx:
        x = sr(x, M)
    for M in My:
        x = kr(x, M)
    return x


# ===== Этап 4. Внутренняя маска (ветвь 1)
def shift(x, kx, ky):
    """Задержка: сдвиг на kx вправо и ky вниз."""
    H, W = x.shape
    y = np.zeros((H, W))
    y[ky:, kx:] = x[:H - ky, :W - kx]
    return y


def inner_mask(x, mx, my, k, ax, ay):
    """2 СР + 2 КР, умножение на k, сдвиг в центр."""
    for m in mx:
        x = sr(x, m)
    for m in my:
        x = kr(x, m)
    return shift(k * x, ax, ay)


# ===== Этап 5. Центральный пиксель (ветвь 3)
def center_pixel(x, A2, bx, by):
    """Пиксель с весом A2, сдвинутый в центр."""
    return shift(A2 * x, bx, by)


# ===== Этап 6. Фильтр целиком: сумматор и деление
def crsf(x, Nx, Ny, A1, A2):
    """Фильтр для одного канала изображения x."""
    Mx, My, mx, my, ax, ay, bx, by, X, S = \
        params(Nx, Ny, A1, A2)
    H, W = x.shape
    x = np.pad(x, ((by, by), (bx, bx)), mode="edge")
    b1 = inner_mask(x, mx, my, X + A1, ax, ay)
    b2 = outer_mask(x, Mx, My)
    b3 = center_pixel(x, A2, bx, by)
    y = b1 - b2 + b3                       # сумматор
    y = y[Ny - 1:Ny - 1 + H, Nx - 1:Nx - 1 + W]
    return y / S if S != 0 else y


# ===== Этап 7. Апертура и самопроверка
def aperture(Nx, Ny, A1, A2):
    """Подаём единицу в центр окна -- получаем маску."""
    d = np.zeros((Ny, Nx))
    d[Ny // 2, Nx // 2] = 1
    return crsf(d, Nx, Ny, A1, A2)


def check():
    """Без A1 и A2 сумма маски должна быть 0."""
    for Nx, Ny in [(3, 3), (7, 5), (9, 13)]:
        k = aperture(Nx, Ny, 0, 0)
        assert abs(k.sum()) < 1e-9, "сумма не 0"
    print("Проверка пройдена: сумма маски = 0")


# ===== Этап 8. Обработка изображения
def main():
    check()
    path = input("Путь к изображению: ")
    Nx = int(input("Nx (нечётное, >= 3) [7]: ") or 7)
    Ny = int(input("Ny (нечётное, >= 3) [7]: ") or 7)
    if Nx % 2 == 0 or Ny % 2 == 0 or min(Nx, Ny) < 3:
        return print("Nx и Ny: нечётные и не меньше 3")
    A1 = float(input("A1: "))
    A2 = float(input("A2: "))
    print(np.round(aperture(Nx, Ny, A1, A2), 3))
    img = Image.open(path).convert("RGB")
    img = np.array(img, dtype=float)
    out = np.zeros_like(img)
    for c in range(3):                   # каналы R, G, B
        out[:, :, c] = crsf(img[:, :, c], Nx, Ny, A1, A2)
    out = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(out).save("result.png")
    print("Готово: result.png")


if __name__ == "__main__":
    main()
