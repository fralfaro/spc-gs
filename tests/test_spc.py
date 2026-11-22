"""Pruebas básicas de patrones, medición y reconstrucción.

Se usa una rejilla pequeña (16 x 32) y precisión doble para que las pruebas
sean rápidas y exactas. Ejecutar con:  pytest -q
"""

import numpy as np
import pytest
import torch

from src.patterns import (PATRONES, indices_medidos, matrices_1d, numero_mediciones,
                          orden_bajas_a_altas, patron_2d)
from src.recon import divergencia, gradiente, reconstruir_directa, reconstruir_tv
from src.spc import OperadorSPC, agregar_ruido

FORMA = (16, 32)


def imagen_prueba(forma=FORMA, semilla=0):
    """Imagen RGB suave + bordes, en [0, 1], forma (3, h, w)."""
    h, w = forma
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    base = 0.5 + 0.3 * np.sin(4 * xx) * np.cos(3 * yy)
    base[: h // 2, : w // 3] += 0.2  # un bloque con bordes
    ruido = np.random.default_rng(semilla).normal(0, 0.02, (3, h, w))
    return torch.tensor(np.clip(base[None] + ruido, 0, 1))


def matriz_phi(op):
    """Construye Phi explícitamente aplicando el operador a la base canónica."""
    h, w = op.shape
    base = torch.eye(h * w, dtype=torch.float64).reshape(h * w, h, w)
    return op.medir(base).T  # (K, N)


@pytest.mark.parametrize("patron", PATRONES)
def test_matrices_1d_ortonormales(patron):
    A, B = matrices_1d(patron, *FORMA)
    assert np.allclose(A @ A.T, np.eye(FORMA[0]), atol=1e-10)
    assert np.allclose(B @ B.T, np.eye(FORMA[1]), atol=1e-10)


@pytest.mark.parametrize("patron", PATRONES)
@pytest.mark.parametrize("razon", [0.1, 0.5, 1.0])
def test_phi_ortogonal(patron, razon):
    """Tras normalizar, las filas de Phi son ortonormales: Phi Phi^T = I."""
    op = OperadorSPC(patron, FORMA, razon, dtype=torch.float64)
    phi = matriz_phi(op)
    k = numero_mediciones(FORMA, razon)
    assert phi.shape == (k, FORMA[0] * FORMA[1])
    assert torch.allclose(phi @ phi.T, torch.eye(k, dtype=torch.float64), atol=1e-10)


@pytest.mark.parametrize("patron", ["hadamard", "random"])
def test_patrones_binarios(patron):
    """Hadamard y aleatorio son binarios: valores +-1/sqrt(N)."""
    n = FORMA[0] * FORMA[1]
    for i in [0, 5, 123]:
        p = patron_2d(patron, FORMA, i)
        assert np.allclose(np.abs(p), 1 / np.sqrt(n))


def test_patron_2d_coincide_con_phi():
    """La imagen del patrón es la fila correspondiente de Phi completa."""
    for patron in PATRONES:
        op = OperadorSPC(patron, FORMA, 1.0, dtype=torch.float64)
        phi = matriz_phi(op).numpy()
        for i in [0, 7, 300]:
            assert np.allclose(patron_2d(patron, FORMA, i).ravel(), phi[i], atol=1e-10)


def test_orden_empieza_en_dc():
    """El primer coeficiente medido en orden de frecuencias es la componente
    continua (u = v = 0)."""
    assert orden_bajas_a_altas(*FORMA)[0] == 0


@pytest.mark.parametrize("patron", PATRONES)
def test_reconstruccion_completa_recupera_imagen(patron):
    """Con 100 % de mediciones y sin ruido, la reconstrucción directa es exacta."""
    x = imagen_prueba()
    op = OperadorSPC(patron, FORMA, 1.0, dtype=torch.float64)
    x_rec = reconstruir_directa(op.medir(x), op)
    assert torch.allclose(x_rec, x, atol=1e-10)


@pytest.mark.parametrize("patron", PATRONES)
def test_tv_completa_recupera_imagen(patron):
    """Con 100 % de mediciones y una TV muy débil, TV-FISTA también recupera
    la imagen (error máximo pequeño)."""
    x = imagen_prueba()
    op = OperadorSPC(patron, FORMA, 1.0, dtype=torch.float64)
    x_rec = reconstruir_tv(op.medir(x), op, lam=1e-5, iteraciones=20)
    assert (x_rec - x).abs().max() < 1e-3


def test_tv_mejora_a_la_directa_con_pocas_mediciones():
    """Al 25 % con patrón aleatorio, TV debe tener menor error que la directa."""
    x = imagen_prueba(forma=(32, 64))
    op = OperadorSPC("random", (32, 64), 0.25, dtype=torch.float64)
    y = op.medir(x)
    err_dir = (reconstruir_directa(y, op) - x).pow(2).mean()
    err_tv = (reconstruir_tv(y, op, lam=0.02, iteraciones=100) - x).pow(2).mean()
    assert err_tv < err_dir


def test_divergencia_es_menos_adjunta_del_gradiente():
    """<grad x, p> = -<x, div p>, necesario para que el prox de TV sea correcto."""
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(3, *FORMA, generator=gen, dtype=torch.float64)
    p = torch.randn(2, 3, *FORMA, generator=gen, dtype=torch.float64)
    assert torch.isclose((gradiente(x) * p).sum(), -(x * divergencia(p)).sum())


def test_misma_semilla_mismo_resultado():
    """La misma semilla produce los mismos índices, patrones y ruido."""
    a = indices_medidos("random", FORMA, 0.1, semilla=449)
    b = indices_medidos("random", FORMA, 0.1, semilla=449)
    c = indices_medidos("random", FORMA, 0.1, semilla=450)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    x = imagen_prueba()
    y1 = OperadorSPC("random", FORMA, 0.1, semilla=449, dtype=torch.float64).medir(x)
    y2 = OperadorSPC("random", FORMA, 0.1, semilla=449, dtype=torch.float64).medir(x)
    assert torch.equal(y1, y2)
    r1 = agregar_ruido(y1, fotones=1e4, sigma_lectura=0.01, bits=16, semilla=449)
    r2 = agregar_ruido(y1, fotones=1e4, sigma_lectura=0.01, bits=16, semilla=449)
    assert torch.equal(r1, r2)
    assert not torch.equal(r1, y1)  # el ruido sí cambia la medición


def test_conjuntos_anidados():
    """Las mediciones a menor razón están contenidas en las de mayor razón."""
    for patron in PATRONES:
        bajo = set(indices_medidos(patron, FORMA, 0.05))
        alto = set(indices_medidos(patron, FORMA, 0.25))
        assert bajo <= alto


def test_ruido_apagado_por_defecto():
    x = imagen_prueba()
    op = OperadorSPC("hadamard", FORMA, 0.5, dtype=torch.float64)
    y = op.medir(x)
    assert torch.equal(agregar_ruido(y), y)


def test_seleccion_de_vistas():
    from src.make_dataset import seleccionar_vistas
    nombres = [f"{i:06d}.jpg" for i in range(1, 252)]
    ent, prueba = seleccionar_vistas(nombres, 24, 3)
    assert len(ent) == 21 and len(prueba) == 3
    assert not set(ent) & set(prueba)
    assert seleccionar_vistas(nombres, 24, 3) == (ent, prueba)  # determinista


def test_remuestreo_mixto_casi_sin_perdida():
    """Una imagen suave de 54x97 llevada a 64x128 (ampliación) o a 48x128 (caso
    mixto: el alto se reduce y el ancho se amplía) y de vuelta debe cambiar poco.
    Antes, el caso mixto ampliaba el ancho con 'area'."""
    from src.make_dataset import remuestrear
    h, w = 54, 97
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    x = (0.5 + 0.3 * torch.sin(3 * xx) * torch.cos(2 * yy)).expand(3, h, w).contiguous()
    # Umbrales medidos: ampliación 8.6e-4; mixto 6.8e-5 (con la versión anterior, 1.3e-3).
    for rejilla, umbral in [((64, 128), 2e-3), ((48, 128), 5e-4)]:
        vuelta = remuestrear(remuestrear(x, rejilla), (h, w))
        assert (vuelta - x).abs().mean() < umbral, rejilla


def test_csv_agrega_columnas_nuevas(tmp_path):
    """Una fila con columnas nuevas no corrompe el CSV: se unen las columnas."""
    import csv
    from src.make_dataset import agregar_fila_csv
    ruta = tmp_path / "m.csv"
    agregar_fila_csv(ruta, {"a": 1, "b": 2})
    agregar_fila_csv(ruta, {"a": 3, "b": 4})
    agregar_fila_csv(ruta, {"a": 5, "b": 6, "c": 7})
    with open(ruta, newline="") as f:
        filas = list(csv.DictReader(f))
    assert [r["a"] for r in filas] == ["1", "3", "5"]
    assert filas[0]["c"] == "" and filas[2]["c"] == "7"


@pytest.mark.parametrize("patron", ["hadamard", "hadamard_cc", "random"])
def test_dmd_sin_ruido_igual_a_medicion_directa(patron):
    """El modelo DMD {0,1} + DC recupera exactamente los coeficientes +-1 si no hay ruido."""
    from src.spc import medir_dmd
    x = imagen_prueba()
    op = OperadorSPC(patron, FORMA, 0.25, dtype=torch.float64)
    assert torch.allclose(medir_dmd(x, op), op.medir(x), atol=1e-9)


def test_dmd_ruido_es_reproducible_y_cambia_la_medicion():
    from src.spc import medir_imagen
    x = imagen_prueba()
    op = OperadorSPC("hadamard", FORMA, 0.25, dtype=torch.float64)
    r = {"dmd": True, "fotones": 1e4, "sigma_lectura": 0.01, "bits": 16}
    a, b = medir_imagen(x, op, r), medir_imagen(x, op, r)
    assert torch.equal(a, b) and not torch.allclose(a, op.medir(x))
