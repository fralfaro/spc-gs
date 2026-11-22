"""Configuración de un experimento (archivos configs/*.yaml).

Un archivo YAML describe un experimento completo: escena, vistas, rejilla, patrones, métodos,
razones, parámetros de la TV, entrenamiento 3DGS y semilla. Todo lo que no se indica toma el
valor por defecto de POR_DEFECTO, que reproduce el experimento principal (Truck, 256x512).
Las claves desconocidas son un error (así un error de tipeo no pasa inadvertido).

Uso:
    from src.config import cargar_config
    cfg = cargar_config("configs/truck_256x512.yaml")
"""

import copy
import re
from pathlib import Path

import numpy as np
import yaml

from .make_dataset import nombre_condicion
from .patterns import PATRONES, SEMILLA, es_potencia_de_2
from .recon import METODOS

RAIZ = Path(__file__).resolve().parents[1]

POR_DEFECTO = {
    "experimento": None,            # obligatorio: nombre de la carpeta results/<experimento>/
    "descripcion": "",
    "escena": "truck",
    "semilla": SEMILLA,             # patrones aleatorios, selección de vistas guardadas, ruido
    "rutas": {
        "imagenes": "data/{escena}",                  # images/ y sparse/0 de la escena
        "condiciones": "data/conditions/{escena}",    # imágenes de cada condición (para 3DGS)
        "modelos": "outputs/{escena}",                # modelos 3DGS entrenados
        "resultados": "results/{experimento}",        # CSV, configuración usada y figuras
    },
    "vistas": {
        "total": 24,                # vistas equiespaciadas que se usan (entrenamiento + prueba)
        "prueba": 3,
        # Subconjunto fijo cuyas imágenes se guardan en results/<experimento>/recon y render_3d,
        # elegido con la semilla entre las de entrenamiento y las de prueba.
        "guardadas": {"entrenamiento": 2, "prueba": 2},
    },
    "rejilla": [256, 512],          # alto x ancho de la rejilla del SPC
    "spc": {
        "patrones_3d": ["hadamard", "random"],    # pasan a 3DGS
        "patrones_solo_2d": ["dct"],              # solo métricas 2D (no realizables en DMD binario)
        "metodos": ["directa", "tv"],
        "razones": [0.05, 0.10, 0.25, 0.50],
        "ruido": None,              # None o dict(fotones=..., sigma_lectura=..., bits=...)
    },
    "tv": {"lam": 0.003, "iteraciones": 100, "iter_prox": 20},
    "3dgs": {
        "iteraciones": [3000, 7000],
        "semillas": [SEMILLA],
        "factor": 1,
        "grado_sh": 3,
        "condiciones": "todas",     # "todas" o lista de nombres de condición
    },
    "evaluacion": {"extendida": True},   # también las vistas no usadas para entrenar
}

_NOMBRE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")


def _fusionar(base, cambios, ruta=""):
    """Copia de `base` con los valores de `cambios`; error si una clave no existe en `base`."""
    salida = copy.deepcopy(base)
    for clave, valor in (cambios or {}).items():
        if clave not in base:
            raise ValueError(f"Clave desconocida en la configuración: {ruta}{clave}")
        if isinstance(base[clave], dict) and isinstance(valor, dict) and clave != "ruido":
            salida[clave] = _fusionar(base[clave], valor, f"{ruta}{clave}.")
        else:
            salida[clave] = valor
    return salida


def validar(cfg):
    """Comprueba tipos y rangos; lanza ValueError con un mensaje claro."""
    if not cfg["experimento"] or not _NOMBRE.match(str(cfg["experimento"])):
        raise ValueError("'experimento' es obligatorio y solo admite a-z, 0-9, '_' y '-'")
    h, w = cfg["rejilla"]
    spc = cfg["spc"]
    for p in spc["patrones_3d"] + spc["patrones_solo_2d"]:
        if p not in PATRONES:
            raise ValueError(f"Patrón desconocido: {p}. Opciones: {PATRONES}")
    if set(spc["patrones_3d"]) & set(spc["patrones_solo_2d"]):
        raise ValueError("Un patrón no puede estar a la vez en patrones_3d y patrones_solo_2d")
    usa_hadamard = {"hadamard", "hadamard_cc", "random"} & set(spc["patrones_3d"] + spc["patrones_solo_2d"])
    if usa_hadamard and not (es_potencia_de_2(h) and es_potencia_de_2(w)):
        raise ValueError(f"Hadamard requiere una rejilla con potencias de 2 (recibido {h}x{w})")
    for m in spc["metodos"]:
        if m not in METODOS:
            raise ValueError(f"Método desconocido: {m}. Opciones: {METODOS}")
    for r in spc["razones"]:
        if not 0 < float(r) <= 1:
            raise ValueError(f"Razón fuera de (0, 1]: {r}")
    v = cfg["vistas"]
    if not 0 < v["prueba"] < v["total"]:
        raise ValueError("Se requiere 0 < vistas.prueba < vistas.total")
    g = v["guardadas"]
    if g["entrenamiento"] > v["total"] - v["prueba"] or g["prueba"] > v["prueba"]:
        raise ValueError("vistas.guardadas pide más vistas de las que hay en ese conjunto")
    if cfg["tv"]["lam"] < 0 or cfg["tv"]["iteraciones"] < 1:
        raise ValueError("tv.lam debe ser >= 0 y tv.iteraciones >= 1")
    c = cfg["3dgs"]["condiciones"]
    if c != "todas" and not isinstance(c, list):
        raise ValueError("3dgs.condiciones debe ser 'todas' o una lista de nombres")
    if c != "todas":
        conocidas = {d["nombre"] for d in _condiciones_3d_todas(cfg)}
        faltan = [n for n in c if n not in conocidas]
        if faltan:
            raise ValueError(f"3dgs.condiciones incluye condiciones que no define la configuración: {faltan}")


def cargar_config(ruta):
    """Lee un YAML, completa con POR_DEFECTO, valida y resuelve las rutas."""
    with open(ruta, encoding="utf-8") as f:
        crudo = yaml.safe_load(f) or {}
    cfg = _fusionar(POR_DEFECTO, crudo)
    cfg["rejilla"] = [int(x) for x in cfg["rejilla"]]
    cfg["spc"]["razones"] = [float(r) for r in cfg["spc"]["razones"]]
    validar(cfg)
    cfg["archivo"] = str(Path(ruta))
    return cfg


def ruta(cfg, clave):
    """Ruta absoluta de cfg['rutas'][clave], con {escena} y {experimento} sustituidos."""
    texto = cfg["rutas"][clave].format(escena=cfg["escena"], experimento=cfg["experimento"])
    p = Path(texto)
    return p if p.is_absolute() else RAIZ / p


def texto_rejilla(cfg):
    h, w = cfg["rejilla"]
    return f"{h}x{w}"


def condiciones_2d(cfg):
    """Control y todas las condiciones SPC (patrones 3D y solo 2D), en orden fijo."""
    rej = cfg["rejilla"]
    salida = [dict(nombre=nombre_condicion("control", rej), tipo="control", patron="control",
                   metodo="-", razon=1.0, en_3d=True)]
    spc = cfg["spc"]
    for p in spc["patrones_3d"] + spc["patrones_solo_2d"]:
        for m in spc["metodos"]:
            for r in spc["razones"]:
                salida.append(dict(nombre=nombre_condicion("spc", rej, p, m, r), tipo="spc",
                                   patron=p, metodo=m, razon=r, en_3d=p in spc["patrones_3d"]))
    return salida


def _condiciones_3d_todas(cfg):
    original = dict(nombre="original", tipo="original", patron="original", metodo="-",
                    razon=1.0, en_3d=True)
    return [original] + [c for c in condiciones_2d(cfg) if c["en_3d"]]


def condiciones_3d(cfg):
    """Condiciones que se entrenan con 3DGS: original, control y patrones_3d (o la lista dada)."""
    todas = _condiciones_3d_todas(cfg)
    elegidas = cfg["3dgs"]["condiciones"]
    return todas if elegidas == "todas" else [c for c in todas if c["nombre"] in elegidas]


def vistas_guardadas(cfg, entrenamiento, prueba):
    """Subconjunto fijo de vistas cuyas imágenes se guardan, elegido con la semilla.

    Devuelve una lista ordenada de nombres: primero las de entrenamiento, luego las de prueba.
    """
    rng = np.random.default_rng(cfg["semilla"])
    g = cfg["vistas"]["guardadas"]
    ent = sorted(rng.choice(sorted(entrenamiento), g["entrenamiento"], replace=False).tolist())
    pru = sorted(rng.choice(sorted(prueba), g["prueba"], replace=False).tolist())
    return ent + pru


def a_yaml(cfg):
    """Texto YAML de la configuración resuelta (para config_usada.yaml)."""
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
