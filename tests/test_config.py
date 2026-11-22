"""Pruebas de la configuración por archivo (src/config.py y configs/*.yaml)."""

from pathlib import Path

import pytest
import yaml

from src.config import (POR_DEFECTO, cargar_config, condiciones_2d, condiciones_3d, ruta,
                        vistas_guardadas)
from src.make_dataset import seleccionar_vistas

RAIZ = Path(__file__).resolve().parents[1]
CONFIGS = sorted((RAIZ / "configs").glob("*.yaml"))


def escribir(tmp_path, datos):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(datos, allow_unicode=True))
    return p


@pytest.mark.parametrize("archivo", CONFIGS, ids=[c.stem for c in CONFIGS])
def test_configs_del_repositorio_son_validas(archivo):
    cfg = cargar_config(archivo)
    assert cfg["experimento"] == archivo.stem  # el nombre del archivo es el del experimento
    assert cfg["semilla"] == 449


def test_valores_por_defecto(tmp_path):
    cfg = cargar_config(escribir(tmp_path, {"experimento": "x"}))
    assert cfg["rejilla"] == [256, 512] and cfg["tv"]["lam"] == 0.003
    assert cfg["semilla"] == 449 and cfg["3dgs"]["semillas"] == [449]
    assert ruta(cfg, "resultados") == RAIZ / "results" / "x"
    assert ruta(cfg, "condiciones") == RAIZ / "data" / "conditions" / "truck"
    assert POR_DEFECTO["experimento"] is None  # cargar no modifica los valores por defecto


@pytest.mark.parametrize("datos, mensaje", [
    ({}, "experimento"),
    ({"experimento": "x", "tv": {"lamda": 0.1}}, "tv.lamda"),
    ({"experimento": "x", "rejilla": [100, 200]}, "potencias de 2"),
    ({"experimento": "x", "spc": {"razones": [0, 0.5]}}, "Razón"),
    ({"experimento": "x", "spc": {"patrones_3d": ["walsh"]}}, "Patrón desconocido"),
    ({"experimento": "x", "3dgs": {"condiciones": ["no_existe"]}}, "no define"),
    ({"experimento": "Con Espacios"}, "experimento"),
])
def test_configuraciones_invalidas(tmp_path, datos, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        cargar_config(escribir(tmp_path, datos))


def test_condiciones_del_experimento_principal():
    """25 condiciones 2D (control + 3 patrones x 2 métodos x 4 razones) y 18 en 3D
    (original + control + 2 patrones x 2 métodos x 4 razones), como en results/."""
    cfg = cargar_config(RAIZ / "configs" / "truck_256x512.yaml")
    c2, c3 = condiciones_2d(cfg), condiciones_3d(cfg)
    assert len(c2) == 25 and len(c3) == 18
    nombres = {c["nombre"] for c in c2}
    assert {"control_256x512", "hadamard_tv_05_256x512", "dct_directa_50_256x512"} <= nombres
    assert not any(c["patron"] == "dct" for c in c3)


def test_vistas_guardadas_fijas_y_dependientes_de_la_semilla(tmp_path):
    nombres = [f"{i:06d}.jpg" for i in range(1, 252)]
    ent, pru = seleccionar_vistas(nombres, 24, 3)
    cfg = cargar_config(escribir(tmp_path, {"experimento": "x"}))
    a = vistas_guardadas(cfg, ent, pru)
    assert a == vistas_guardadas(cfg, ent, pru)
    assert len(a) == 4 and set(a[:2]) <= set(ent) and set(a[2:]) <= set(pru)
    cfg["semilla"] = 450
    assert vistas_guardadas(cfg, ent, pru) != a
