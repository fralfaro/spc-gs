"""Pruebas de la escritura protegida de resultados (src/salidas.py)."""

import pytest

from src.salidas import proteger


def test_proteger_detiene_si_existe(tmp_path):
    ruta = tmp_path / "tabla.csv"
    ruta.write_text("a\n1\n")
    with pytest.raises(SystemExit):
        proteger(ruta)
    assert ruta.read_text() == "a\n1\n"  # no se tocó


def test_proteger_permite_si_no_existe_o_se_fuerza(tmp_path):
    proteger(tmp_path / "nuevo.csv")  # no existe: no hace nada
    ruta = tmp_path / "tabla.csv"
    ruta.write_text("x")
    proteger(ruta, forzar=True)  # existe, pero se pidió forzar
