"""Patrones de medición de la cámara de un píxel (SPC).

Cada patrón se describe como una transformada ortonormal y separable sobre una
rejilla de h x w píxeles:

    coeficientes C = A @ X @ B.T          (A: h x h, B: w x w, ortonormales)

Medir una fracción r de los coeficientes equivale a proyectar la imagen sobre
r*h*w patrones (las filas de la matriz de medición Phi). Los patrones son:

- "hadamard": Hadamard 2D separable con filas en orden de secuencia (número de
  cambios de signo), y coeficientes 2D ordenados de bajas a altas frecuencias.
  Se miden los primeros r*h*w coeficientes.
- "hadamard_cc": la misma base de Hadamard, pero con el orden "cake-cutting"
  (el de inf449-proyecto): los coeficientes (u, v) se ordenan por el número de
  regiones conexas del patrón, (u + 1) * (v + 1), en vez de u/h + v/w.
- "random": binario aleatorio. Se permutan los píxeles con una semilla y se
  aplica Hadamard 2D; luego se submuestrea un subconjunto aleatorio (con la
  misma semilla) de coeficientes, incluyendo siempre el patrón constante
  (intensidad total). Los patrones resultantes son +-1 y su estructura
  espacial es aleatoria.
- "dct": DCT-II 2D ortonormal con orden en zigzag (de bajas a altas
  frecuencias). No es binario; se incluye como referencia.

La función común `indices_medidos` devuelve, para una rejilla y una razón,
los índices (en el arreglo aplanado de coeficientes) que se miden.
"""

import numpy as np
from scipy.linalg import hadamard

PATRONES = ("hadamard", "hadamard_cc", "random", "dct")
SEMILLA = 449


def es_potencia_de_2(n):
    return n > 0 and (n & (n - 1)) == 0


def hadamard_secuencia(n):
    """Matriz de Hadamard n x n ortonormal, con filas en orden de secuencia.

    La secuencia de una fila es su número de cambios de signo: es el análogo
    de la frecuencia para funciones de Walsh. Ordenar por secuencia deja
    primero los patrones "suaves" y al final los más oscilantes.
    """
    if not es_potencia_de_2(n):
        raise ValueError(f"Hadamard requiere un tamaño potencia de 2 (recibido {n})")
    H = hadamard(n).astype(np.float64)
    cambios = (np.diff(H, axis=1) != 0).sum(axis=1)  # cambios de signo por fila
    H = H[np.argsort(cambios, kind="stable")]
    return H / np.sqrt(n)  # normalización: H @ H.T = I


def dct_matriz(n):
    """Matriz de la DCT-II ortonormal de tamaño n x n (fila k = frecuencia k)."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    C = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    C[0] *= np.sqrt(1.0 / n)
    C[1:] *= np.sqrt(2.0 / n)
    return C


def matrices_1d(patron, h, w):
    """Devuelve (A, B), las matrices 1D por filas y por columnas del patrón."""
    if patron in ("hadamard", "hadamard_cc", "random"):
        return hadamard_secuencia(h), hadamard_secuencia(w)
    if patron == "dct":
        return dct_matriz(h), dct_matriz(w)
    raise ValueError(f"Patrón desconocido: {patron}. Opciones: {PATRONES}")


def orden_bajas_a_altas(h, w):
    """Orden de los coeficientes 2D de bajas a altas frecuencias (zigzag).

    Los coeficientes (u, v) se recorren por diagonales de frecuencia
    normalizada u/h + v/w; dentro de una diagonal, por u creciente. En una
    rejilla cuadrada esto coincide con las diagonales del zigzag clásico de
    JPEG; normalizar por h y w trata igual a ambos ejes en rejillas
    rectangulares (por ejemplo 256 x 512).
    Devuelve índices planos (u * w + v).
    """
    u, v = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    frecuencia = u / h + v / w
    # lexsort ordena por la última clave primero: frecuencia, luego u.
    return np.lexsort((u.ravel(), frecuencia.ravel()))


def orden_cake_cutting(h, w):
    """Orden "cake-cutting" de los coeficientes de Hadamard (de menos a más regiones).

    Con las filas en orden de secuencia, la fila u tiene u cambios de signo, así que el
    patrón separable (u, v) tiene exactamente (u + 1) * (v + 1) regiones conexas. Se ordena
    por ese número; los empates, por índice plano (u * w + v).
    """
    u, v = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.lexsort((np.arange(h * w), ((u + 1) * (v + 1)).ravel()))


def numero_mediciones(shape, razon):
    """Número de mediciones K = round(razon * h * w), al menos 1."""
    h, w = shape
    if not 0 < razon <= 1:
        raise ValueError("La razón de muestreo debe estar en (0, 1]")
    return max(1, int(round(razon * h * w)))


def permutacion_pixeles(shape, semilla=SEMILLA):
    """Permutación aleatoria de los píxeles (solo para el patrón "random")."""
    h, w = shape
    rng = np.random.default_rng(semilla)
    return rng.permutation(h * w)


def orden_medicion(patron, shape, semilla=SEMILLA):
    """Orden completo en que se toman las mediciones (índices planos).

    - hadamard, hadamard_cc y dct: de bajas a altas frecuencias (determinista; la semilla
      no se usa).
    - random: primero el coeficiente 0 y luego los demás en orden aleatorio.
      El coeficiente 0 (patrón constante = intensidad total de la escena) se
      mide siempre: ese patrón no cambia al permutar los píxeles y, si no se
      mide, el nivel medio de la imagen queda indeterminado (la TV no puede
      recuperarlo porque TV(x + c) = TV(x)). Se usa un generador distinto al
      de la permutación de píxeles (semilla + 1) para que sean independientes.
    """
    h, w = shape
    if patron in ("hadamard", "dct"):
        return orden_bajas_a_altas(h, w)
    if patron == "hadamard_cc":
        return orden_cake_cutting(h, w)
    if patron == "random":
        rng = np.random.default_rng(semilla + 1)
        return np.concatenate([[0], 1 + rng.permutation(h * w - 1)])
    raise ValueError(f"Patrón desconocido: {patron}. Opciones: {PATRONES}")


def indices_medidos(patron, shape, razon, semilla=SEMILLA):
    """Índices planos de los coeficientes medidos para (patrón, rejilla, razón):
    los primeros K = round(razon * h * w) del orden de medición.

    Como el orden es siempre el mismo, los conjuntos quedan anidados: las
    mediciones al 5 % están contenidas en las del 10 %, etc.
    """
    k = numero_mediciones(shape, razon)
    return np.sort(orden_medicion(patron, shape, semilla)[:k])


def patron_2d(patron, shape, indice, semilla=SEMILLA):
    """Imagen h x w del patrón asociado a un coeficiente (para visualizar).

    Es la fila `indice` de Phi reordenada como imagen: lo que mostraría el
    modulador espacial (DMD) para tomar esa medición.
    """
    h, w = shape
    A, B = matrices_1d(patron, h, w)
    u, v = divmod(int(indice), w)
    P = np.outer(A[u], B[v])  # patrón separable = fila u de A x fila v de B
    if patron == "random":
        perm = permutacion_pixeles(shape, semilla)
        # El coeficiente se calcula sobre la imagen permutada x[perm]; en la
        # rejilla original, el valor del patrón en el píxel perm[j] es P[j].
        Q = np.empty(h * w)
        Q[perm] = P.ravel()
        P = Q.reshape(h, w)
    return P
