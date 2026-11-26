"""Четырёхкаскадный рекурсивно-сепарабельный фильтр (ЧРСФ), ускоренная версия."""
import numpy as np


# ---------- параметры фильтра
def outer_orders(N):
    """Длины 4 рециркуляторов внешней маски (сумма длин N + 3)."""
    s = N + 3
    M = [s // 4] * 4
    for i in range(s % 4):
        M[i] += 1
    return M


def inner_size(N):
    """Размер внутренней маски: наибольшее нечётное число, не большее N / 2."""
    return 2 * ((N - 3) // 4) + 1


def params(Nx, Ny, A1, A2):
    """Все параметры фильтра для окна Nx x Ny."""
    Mx, My = outer_orders(Nx), outer_orders(Ny)
    nx, ny = inner_size(Nx), inner_size(Ny)
    mx, my = [(nx + 1) // 2] * 2, [(ny + 1) // 2] * 2
    ax, ay = (Nx - nx) // 2, (Ny - ny) // 2  # сдвиг внутренней маски
    bx, by = (Nx - 1) // 2, (Ny - 1) // 2  # сдвиг центрального элемента
    S_out = np.prod(Mx) * np.prod(My)  # сумма внешней маски
    S_in = np.prod(mx) * np.prod(my)  # сумма внутренней маски
    X = S_out / S_in  # коэффициент уравнивания
    S = A1 * S_in + A2  # сумма итоговой маски
    return Mx, My, mx, my, ax, ay, bx, by, X, S


# ---------- рециркуляторы
def rec(x, M, axis):
    """Рециркулятор длины M: сумма M соседних отсчётов.
    axis = 1 - строчный (по строке), axis = 0 - кадровый (по столбцу)."""
    if M <= 1:  # длина 1 ничего не меняет
        return x
    if axis == 1:
        c = np.cumsum(x, axis=1, dtype=x.dtype)  # накопленная сумма по строке
        y = np.empty_like(c)
        y[:, :M] = c[:, :M]
        np.subtract(c[:, M:], c[:, :-M], out=y[:, M:])  # сумма окна = разность накопленных сумм
        return y
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.shape[0]):  # сразу целыми строками
        np.add(y[i - 1], x[i], out=y[i])  # + вошедшая строка
        if i >= M:
            np.subtract(y[i], x[i - M], out=y[i])  # - вышедшая строка
    return y


def chain(x, lengths, axis):
    """Несколько рециркуляторов подряд."""
    for M in lengths:
        x = rec(x, M, axis)
    return x


def remainder(outer, inner):
    """Звенья внешней маски, которых нет во внутренней (None, если общих звеньев нет)."""
    rest = list(outer)
    for m in inner:
        if m not in rest:
            return None
        rest.remove(m)
    return rest


def n_recirculators(Nx, Ny, shared=True):
    """Число рециркуляторов: исходная схема (shared=False) или ускоренная."""
    Mx, My, mx, my = params(Nx, Ny, 0, 0)[:4]
    if not shared:
        return len(Mx) + len(My) + len(mx) + len(my)
    count = lambda L: sum(1 for M in L if M > 1)  # noqa: E731
    rx, ry = remainder(Mx, mx), remainder(My, my)
    return (count(mx) + count(my) + count(Mx if rx is None else rx)
            + count(My if ry is None else ry))


def branches(xp, Mx, My, mx, my):
    """Ветвь 1 (внутренняя маска) и ветвь 2 (внешняя маска).
    Если звенья внутренней маски есть во внешней, внешняя достраивается из внутренней."""
    rx, ry = remainder(Mx, mx), remainder(My, my)
    if rx is not None and ry is not None:
        inner = chain(chain(xp, mx, 1), my, 0)
        outer = chain(chain(inner, rx, 1), ry, 0)
    elif rx is not None:
        u = chain(xp, mx, 1)
        inner = chain(u, my, 0)
        outer = chain(chain(u, rx, 1), My, 0)
    elif ry is not None:
        v = chain(xp, my, 0)
        inner = chain(v, mx, 1)
        outer = chain(chain(v, ry, 0), Mx, 1)
    else:
        inner = chain(chain(xp, mx, 1), my, 0)
        outer = chain(chain(xp, Mx, 1), My, 0)
    return inner, outer


# ---------- фильтр
def crsf_fast(img, Nx, Ny, A1, A2, normalize=True):
    """Обработка изображения (полутонового или цветного)."""
    Mx, My, mx, my, ax, ay, bx, by, X, S = params(Nx, Ny, A1, A2)
    x = np.asarray(img)
    H, W = x.shape[:2]
    if np.issubdtype(x.dtype, np.integer):
        # целые числа: int32, если сумма внешней маски не переполнит его
        big = int(np.abs(x).max()) * np.prod(Mx) * np.prod(My) >= 2**31
        work = np.int64 if big else np.int32
    else:
        work = np.float64
    pad = ((by, by), (bx, bx)) + ((0, 0),) * (x.ndim - 2)
    xp = np.pad(x.astype(work), pad, mode="edge")  # края дополняем крайними пикселями
    inner, outer = branches(xp, Mx, My, mx, my)
    r0, c0 = Ny - 1, Nx - 1
    I = inner[r0 - ay:r0 - ay + H, c0 - ax:c0 - ax + W]  # сдвиг = смещение вырезки
    O = outer[r0:r0 + H, c0:c0 + W]
    if normalize and S != 0:
        k1, k2, k3 = (X + A1) / S, 1 / S, A2 / S
    else:
        k1, k2, k3 = X + A1, 1, A2
    return k1 * I - k2 * O + k3 * x  # сумматор трёх ветвей


def aperture_fast(Nx, Ny, A1, A2, normalize=True):
    """Маска фильтра: отклик на единицу в центре окна."""
    d = np.zeros((Ny, Nx))
    d[Ny // 2, Nx // 2] = 1.0
    return crsf_fast(d, Nx, Ny, A1, A2, normalize)


def to_uint8(y):
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


# ---------- коэффициенты k и w
def sums(Nx, Ny):
    """S_внеш, S_вн и X для окна."""
    Mx, My, mx, my = params(Nx, Ny, 0, 0)[:4]
    s_out = float(np.prod(Mx) * np.prod(My))
    s_in = float(np.prod(mx) * np.prod(my))
    return s_out, s_in, s_out / s_in


def to_relative(Nx, Ny, A1, A2):
    """A1, A2 -> k (усиление перепадов) и w (доля исходного пикселя)."""
    s_out, s_in, _ = sums(Nx, Ny)
    S = A1 * s_in + A2
    return s_out / S, A2 / S


def from_relative(Nx, Ny, k, w=1.0):
    """k, w -> A1, A2 для заданного окна."""
    s_out, s_in, X = sums(Nx, Ny)
    return (1 - w) * X / k, w * s_out / k
