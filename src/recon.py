"""Reconstrucción 2D a partir de las mediciones del single pixel.

Dos métodos:

1. Directa: x = Phi.T y. Rellena con ceros los coeficientes no medidos y
   aplica la transformada inversa. Como Phi tiene filas ortonormales, es la
   solución de mínimos cuadrados de norma mínima.

2. Variación total (TV) con FISTA: resuelve

       min_x  0.5 * ||Phi x - y||^2  +  lam * TV(x)

   donde TV(x) = suma de |gradiente de x| (TV isotrópica, por canal).
   FISTA alterna un paso de gradiente del término de datos con el operador
   proximal de la TV. Como Phi Phi.T = I, la constante de Lipschitz del
   gradiente es 1 y el paso puede ser 1. Con paso 1, el paso de gradiente
   equivale a reemplazar los coeficientes medidos por y y dejar los demás.
"""

import math

import torch


def reconstruir_directa(y, operador):
    """Reconstrucción directa (ceros en lo no medido)."""
    return operador.adjunta(y)


# --- Operadores de diferencias finitas para la TV ---------------------------
def gradiente(x):
    """Diferencias hacia adelante (condición de Neumann en el borde).
    x: (..., h, w) -> (2, ..., h, w)."""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return torch.stack([gx, gy])


def divergencia(p):
    """Divergencia discreta, definida como -gradiente^T (diferencias hacia atrás)."""
    px, py = p[0], p[1]
    dx = torch.zeros_like(px)
    dy = torch.zeros_like(py)
    dx[..., :, 0] = px[..., :, 0]
    dx[..., :, 1:-1] = px[..., :, 1:-1] - px[..., :, :-2]
    dx[..., :, -1] = -px[..., :, -2]
    dy[..., 0, :] = py[..., 0, :]
    dy[..., 1:-1, :] = py[..., 1:-1, :] - py[..., :-2, :]
    dy[..., -1, :] = -py[..., -2, :]
    return dx + dy


def prox_tv(v, lam, iteraciones=20, p=None):
    """Operador proximal de lam*TV mediante el algoritmo de Chambolle (2004):

        argmin_x 0.5 * ||x - v||^2 + lam * TV(x)

    Se resuelve el problema dual con un descenso proyectado sobre el campo p
    (|p| <= 1). `p` permite reutilizar la solución dual previa (arranque en
    caliente), lo que reduce las iteraciones necesarias dentro de FISTA.
    Devuelve (x, p).
    """
    if lam <= 0:
        return v, p
    if p is None:
        p = torch.zeros((2,) + tuple(v.shape), device=v.device, dtype=v.dtype)
    tau = 0.125  # paso estable: tau <= 1/8
    for _ in range(iteraciones):
        g = gradiente(divergencia(p) - v / lam)
        norma = torch.sqrt(g[0] ** 2 + g[1] ** 2)
        p = (p + tau * g) / (1 + tau * norma)
    return v - lam * divergencia(p), p


def reconstruir_tv(y, operador, lam=0.003, iteraciones=100, iter_prox=20):
    """Reconstrucción con variación total mediante FISTA.

    Parámetros
    ----------
    lam : peso de la TV (imágenes en [0, 1]). Más grande = más suave.
    iteraciones : iteraciones de FISTA.
    iter_prox : iteraciones internas de Chambolle por cada paso proximal.
    """
    x = operador.adjunta(y)  # inicio: reconstrucción directa
    z = x.clone()
    t = 1.0
    p = None
    for _ in range(iteraciones):
        # 1) Paso de gradiente sobre 0.5*||Phi z - y||^2 (paso 1 = 1/L).
        v = z - operador.adjunta(operador.medir(z) - y)
        # 2) Paso proximal de la TV.
        x_nuevo, p = prox_tv(v, lam, iter_prox, p)
        # 3) Extrapolación de Nesterov (lo que distingue FISTA de ISTA).
        t_nuevo = (1 + math.sqrt(1 + 4 * t * t)) / 2
        z = x_nuevo + ((t - 1) / t_nuevo) * (x_nuevo - x)
        x, t = x_nuevo, t_nuevo
    return x


METODOS = ("directa", "tv")


def reconstruir(y, operador, metodo, **kwargs):
    """Punto de entrada común: metodo = "directa" o "tv"."""
    if metodo == "directa":
        return reconstruir_directa(y, operador)
    if metodo == "tv":
        return reconstruir_tv(y, operador, **kwargs)
    raise ValueError(f"Método desconocido: {metodo}. Opciones: {METODOS}")
