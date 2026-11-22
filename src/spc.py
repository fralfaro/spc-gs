"""Simulación de la cámara de un píxel: medición y = Phi x por canal de color.

Phi (K x N, con N = h*w) está formada por K filas de una transformada
ortonormal separable (ver patterns.py). Por eso:

    Phi @ Phi.T = I_K          (filas ortonormales)
    Phi.T @ y                  = reconstrucción con ceros en lo no medido

Las operaciones se implementan con multiplicaciones de matrices pequeñas
(h x h y w x w) en lugar de construir Phi explícitamente, que para una
rejilla de 256 x 512 tendría 131072 columnas.

Convención de formas: una imagen es un tensor (..., h, w), por ejemplo
(3, h, w) para RGB; cada canal se mide por separado con el mismo Phi.
"""

import numpy as np
import torch

from .patterns import SEMILLA, indices_medidos, matrices_1d, permutacion_pixeles


class OperadorSPC:
    """Operador de medición Phi para un patrón, una rejilla y una razón."""

    def __init__(self, patron, shape, razon, semilla=SEMILLA, device="cpu",
                 dtype=torch.float32):
        self.patron = patron
        self.shape = tuple(shape)
        self.razon = float(razon)
        self.semilla = semilla
        h, w = self.shape
        A, B = matrices_1d(patron, h, w)
        self.A = torch.tensor(A, device=device, dtype=dtype)
        self.B = torch.tensor(B, device=device, dtype=dtype)
        self.idx = torch.tensor(indices_medidos(patron, self.shape, razon, semilla),
                                device=device, dtype=torch.long)
        self.perm = None
        if patron == "random":
            perm = permutacion_pixeles(self.shape, semilla)
            self.perm = torch.tensor(perm, device=device, dtype=torch.long)
            self.perm_inv = torch.tensor(np.argsort(perm), device=device,
                                         dtype=torch.long)

    @property
    def num_mediciones(self):
        return int(self.idx.numel())

    # --- Transformada completa (ortonormal) ---------------------------------
    def transformada(self, x):
        """Coeficientes de todos los patrones: C = A @ P(x) @ B.T."""
        h, w = self.shape
        if self.perm is not None:  # permutación de píxeles del patrón aleatorio
            x = x.reshape(*x.shape[:-2], h * w)[..., self.perm].reshape(x.shape)
        return self.A @ x @ self.B.T

    def transformada_inversa(self, c):
        """Inversa (= transpuesta, por ortonormalidad): x = P^-1(A.T @ C @ B)."""
        h, w = self.shape
        x = self.A.T @ c @ self.B
        if self.perm is not None:
            x = x.reshape(*x.shape[:-2], h * w)[..., self.perm_inv].reshape(x.shape)
        return x

    # --- Medición y su adjunta ----------------------------------------------
    def medir(self, x):
        """y = Phi x: (..., h, w) -> (..., K)."""
        c = self.transformada(x)
        return c.reshape(*c.shape[:-2], -1)[..., self.idx]

    def adjunta(self, y):
        """Phi.T y: rellena con ceros lo no medido y aplica la inversa."""
        h, w = self.shape
        c = torch.zeros(*y.shape[:-1], h * w, device=y.device, dtype=y.dtype)
        c[..., self.idx] = y
        return self.transformada_inversa(c.reshape(*y.shape[:-1], h, w))


def agregar_ruido(y, fotones=None, sigma_lectura=0.0, bits=None,
                  semilla=SEMILLA):
    """Ruido opcional del detector (apagado por defecto).

    Parámetros
    ----------
    fotones : float o None
        Si se indica, se aplica ruido de Poisson (disparo). La señal se escala
        para que el valor absoluto máximo de y equivalga a `fotones` cuentas.
        Aproximación: como los patrones +-1 dan valores negativos, se aplica
        Poisson sobre |y| y se restaura el signo (varianza proporcional a la
        magnitud de la medición).
    sigma_lectura : float
        Desviación estándar del ruido gaussiano de lectura, como fracción de
        max|y|.
    bits : int o None
        Si se indica (por ejemplo 16), cuantiza uniformemente y en
        [-max|y|, max|y|] con 2**bits niveles, como un conversor A/D.
    """
    gen = torch.Generator(device=y.device).manual_seed(semilla)
    escala = y.abs().max().clamp_min(1e-12)
    if fotones is not None:
        k = fotones / escala
        y = torch.sign(y) * torch.poisson(y.abs() * k, generator=gen) / k
    if sigma_lectura > 0:
        y = y + sigma_lectura * escala * torch.randn(y.shape, generator=gen,
                                                     device=y.device, dtype=y.dtype)
    if bits is not None:
        paso = 2 * escala / (2 ** bits - 1)
        y = torch.round((y + escala) / paso) * paso - escala
    return y


def medir_dmd(x, operador, ruido=None, repeticiones_dc=16):
    """Medición con un DMD real: máscaras {0, 1} y una medición DC (todos los espejos encendidos).

    Un DMD no puede mostrar patrones +-1. Con p = (1 + h)/2 (h = patrón +-1, ya normalizado a
    sqrt(N) * fila de Phi) se mide y = <p, x> = (S + sqrt(N) c)/2, con S = suma de x (medición DC)
    y c el coeficiente +-1. Se recupera c = (2 y - S)/sqrt(N): el ruido de y se duplica y el del DC
    (promediado `repeticiones_dc` veces) se suma a todos los coeficientes. Sin ruido devuelve
    exactamente `operador.medir(x)`. Solo vale para patrones +-1 (hadamard, hadamard_cc, random).
    """
    if operador.patron == "dct":
        raise ValueError("El modelo DMD {0,1} no aplica a la DCT (no es binaria)")
    h, w = operador.shape
    raiz = float(np.sqrt(h * w))
    c = operador.medir(x)                                  # (..., K), coeficientes +-1 normalizados
    s = raiz * operador.transformada(x)[..., 0, 0]         # suma de la imagen (medición DC)
    y = (s[..., None] + raiz * c) / 2                      # señal medida con máscaras {0, 1} (>= 0)
    if ruido:
        r = int(repeticiones_dc)
        junto = torch.cat([y, s[..., None].expand(*y.shape[:-1], r)], dim=-1)  # misma escala de ruido
        junto = agregar_ruido(junto, **ruido)
        y, s = junto[..., :-r], junto[..., -r:].mean(dim=-1)
    return (2 * y - s[..., None]) / raiz


def medir_imagen(x, operador, ruido=None):
    """Mide una imagen (3, h, w) con el operador; `ruido` es un dict opcional
    con los argumentos de `agregar_ruido` (None = sin ruido). Si incluye `dmd: True`
    se usa el modelo de DMD {0, 1} de `medir_dmd` (clave opcional `repeticiones_dc`)."""
    if ruido and ruido.get("dmd"):
        resto = {k: v for k, v in ruido.items() if k not in ("dmd", "repeticiones_dc")}
        return medir_dmd(x, operador, resto or None, ruido.get("repeticiones_dc", 16))
    y = operador.medir(x)
    if ruido:
        y = agregar_ruido(y, **ruido)
    return y
