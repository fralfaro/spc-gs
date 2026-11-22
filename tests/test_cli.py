"""Pruebas de la CLI (src/cli.py) con una escena sintética pequeña en una carpeta temporal."""

import filecmp

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from src import cli

COLUMNAS_POR_VISTA = ["experimento", "escena", "condicion", "tipo", "patron", "metodo", "razon",
                      "rejilla", "mediciones", "semilla", "lam_tv", "iter_tv", "iter_prox",
                      "ruido", "origen", "conjunto", "vista", "PSNR", "SSIM", "LPIPS",
                      "PSNR_rejilla", "SSIM_rejilla"]


@pytest.fixture
def escena(tmp_path):
    """8 imágenes RGB suaves de 48x96 y una carpeta sparse vacía."""
    raiz = tmp_path / "escena"
    (raiz / "images").mkdir(parents=True)
    (raiz / "sparse" / "0").mkdir(parents=True)
    rng = np.random.default_rng(0)
    yy, xx = np.meshgrid(np.linspace(0, 1, 48), np.linspace(0, 1, 96), indexing="ij")
    for i in range(8):
        f = 0.5 + 0.3 * np.sin((3 + i) * xx) * np.cos((2 + i) * yy)
        img = np.clip(np.stack([f, f[::-1], 1 - f], -1) + rng.normal(0, 0.02, (48, 96, 3)), 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(raiz / "images" / f"{i:06d}.jpg",
                                                           quality=95)
    return raiz


def config(tmp_path, escena, nombre="exp", **cambios):
    datos = {"experimento": nombre, "escena": "sintetica",
             "rutas": {"imagenes": str(escena), "condiciones": str(tmp_path / "condiciones"),
                       "modelos": str(tmp_path / "modelos"),
                       "resultados": str(tmp_path / "resultados" / "{experimento}")},
             "vistas": {"total": 6, "prueba": 2, "guardadas": {"entrenamiento": 1, "prueba": 1}},
             "rejilla": [32, 64],
             "spc": {"patrones_3d": ["hadamard"], "patrones_solo_2d": ["random"],
                     "metodos": ["directa", "tv"], "razones": [0.5]},
             "tv": {"lam": 0.003, "iteraciones": 5, "iter_prox": 5},
             "3dgs": {"iteraciones": []}, "evaluacion": {"extendida": False}}
    for clave, valor in cambios.items():
        datos[clave] = {**datos[clave], **valor} if isinstance(valor, dict) else valor
    p = tmp_path / f"{nombre}.yaml"
    p.write_text(yaml.safe_dump(datos, allow_unicode=True))
    return p


def test_reconstruir_esquema_y_archivos(tmp_path, escena):
    cli.main(["reconstruir", "--config", str(config(tmp_path, escena))])
    res = tmp_path / "resultados" / "exp"
    pv = pd.read_csv(res / "metrics_2d_por_vista.csv")
    assert list(pv.columns) == COLUMNAS_POR_VISTA
    # control + 2 patrones x 2 métodos x 1 razón = 5 condiciones; 6 vistas cada una.
    assert len(pv) == 30 and pv.condicion.nunique() == 5
    assert set(pv.origen) == {"generada"} and (pv.semilla == 449).all()
    assert pv.groupby("condicion").conjunto.value_counts().unstack().eq([4, 2]).all().all()
    resumen = pd.read_csv(res / "metrics_2d.csv")
    assert len(resumen) == 15 and set(resumen.conjunto) == {"todas", "entrenamiento", "prueba"}
    todas = resumen[resumen.conjunto == "todas"].set_index("condicion").PSNR
    assert np.allclose(todas, pv.groupby("condicion").PSNR.mean()[todas.index])
    assert (res / "config_usada.yaml").exists() and (res / "registro_corridas.csv").exists()
    usada = yaml.safe_load((res / "config_usada.yaml").read_text())
    assert usada["configuracion"]["semilla"] == 449 and len(usada["commit"]) >= 7
    # Subconjunto guardado: 2 vistas; referencia + una carpeta por condición.
    guardadas = sorted(p.name for p in (res / "recon" / "referencia").glob("*.jpg"))
    assert len(guardadas) == 2
    assert (res / "recon" / "hadamard_tv_50_32x64" / guardadas[0].replace(".jpg", "_rejilla.png")).exists()
    assert len(pd.read_csv(res / "metrics_2d_guardadas.csv")) == 4 * 2


def test_no_sobrescribe_y_reutiliza(tmp_path, escena):
    cfg = config(tmp_path, escena)
    cli.main(["reconstruir", "--config", str(cfg)])
    imagen = tmp_path / "condiciones" / "hadamard_tv_50_32x64" / "images" / "000000.jpg"
    antes = imagen.stat().st_mtime_ns
    previo = pd.read_csv(tmp_path / "resultados" / "exp" / "metrics_2d_por_vista.csv")
    with pytest.raises(SystemExit):  # results/<experimento> ya existe
        cli.main(["reconstruir", "--config", str(cfg)])
    cli.main(["reconstruir", "--config", str(cfg), "--forzar"])
    nuevo = pd.read_csv(tmp_path / "resultados" / "exp" / "metrics_2d_por_vista.csv")
    assert set(nuevo.origen) == {"reutilizada"}
    assert imagen.stat().st_mtime_ns == antes  # data/ no se tocó
    pd.testing.assert_frame_equal(nuevo.drop(columns="origen"), previo.drop(columns="origen"))


def test_otra_configuracion_con_el_mismo_nombre_se_detiene(tmp_path, escena):
    cli.main(["dataset", "--config", str(config(tmp_path, escena))])
    with pytest.raises(SystemExit, match="otra configuración"):
        cli.main(["dataset", "--config", str(config(tmp_path, escena, tv={"lam": 0.01}))])


def test_condicion_existente_con_otros_parametros_se_detiene(tmp_path, escena):
    cli.main(["reconstruir", "--config", str(config(tmp_path, escena))])
    otra = config(tmp_path, escena, nombre="otro", tv={"lam": 0.01})
    with pytest.raises(SystemExit, match="otros parámetros"):
        cli.main(["reconstruir", "--config", str(otra)])


def test_misma_semilla_mismas_imagenes(tmp_path, escena):
    cli.main(["reconstruir", "--config", str(config(tmp_path, escena))])
    otra = tmp_path / "otra"
    otra.mkdir()
    cli.main(["reconstruir", "--config", str(config(otra, escena))])
    for c in ["random_directa_50_32x64", "random_tv_50_32x64"]:
        a = tmp_path / "condiciones" / c / "images"
        b = otra / "condiciones" / c / "images"
        assert all(filecmp.cmp(p, b / p.name, shallow=False) for p in a.iterdir())


@pytest.mark.parametrize("nombre, n", [("truck_256x512", 36), ("truck_256x512_semillas", 10),
                                       ("truck_256x512_30k", 2), ("truck_512x1024", 0)])
def test_corridas_3d_de_las_configuraciones(nombre, n):
    """Las corridas siguen el esquema de carpetas de scripts/train_gsplat.sh."""
    from src.config import RAIZ, cargar_config
    lista = list(cli.corridas_3d(cargar_config(RAIZ / "configs" / f"{nombre}.yaml")))
    assert len(lista) == n
    for c, it, s, corrida in lista:
        assert corrida == RAIZ / "outputs" / "truck" / c["nombre"] / f"it{it}_s{s}_f1"
