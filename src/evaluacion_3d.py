"""Evaluación de modelos 3DGS contra las imágenes ORIGINALES.

Núcleo común de scripts/eval_3d.py (interfaz heredada, escribe en results/metrics_3d.csv) y de
`python -m src.cli evaluar` (escribe en results/<experimento>/). Los renders se comparan siempre
con las imágenes originales de la escena, nunca con las degradadas con que se entrenó.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .metrics import calcular_metricas
from .train_3dgs import cargar_vistas, renderizar

RAIZ = Path(__file__).resolve().parents[1]


def cargar_corrida(corrida):
    """(info.json, parámetros de las gaussianas en la GPU) de una carpeta de entrenamiento."""
    corrida = Path(corrida)
    info = json.loads((corrida / "info.json").read_text())
    splats = {k: v.cuda() for k, v in torch.load(corrida / "ckpt.pt").items()}
    return info, splats


def evaluar_vistas(splats, info, nombres, carpeta_renders=None, mostrar=False,
                   carpeta_imagenes=None):
    """Renderiza `nombres` y los compara con las imágenes ORIGINALES.
    Devuelve (promedio de PSNR, SSIM y LPIPS, lista de métricas por vista).

    `carpeta_imagenes` es la carpeta de la escena (con images/ y sparse/0); por defecto
    data/<escena>."""
    originales = Path(carpeta_imagenes) if carpeta_imagenes else RAIZ / "data" / info["escena"]
    filas = []
    # Se cargan de a 25 vistas para no llenar la memoria de la GPU con muchas imágenes.
    for i in range(0, len(nombres), 25):
        filas += _evaluar_lote(splats, info, originales, nombres[i:i + 25],
                               carpeta_renders, mostrar)
    media = {k: float(np.mean([f[k] for f in filas])) for k in ("psnr", "ssim", "lpips")}
    return media, filas


def _evaluar_lote(splats, info, originales, nombres, carpeta_renders, mostrar):
    vistas = cargar_vistas(originales, nombres, info["factor"], "cuda")
    filas = []
    for v in vistas:
        alto, ancho = v["imagen"].shape[:2]
        with torch.no_grad():
            render, _ = renderizar(splats, v["viewmat"], v["K"], ancho, alto, info["grado_sh"])
        render = render.clamp(0, 1)
        if carpeta_renders is not None:
            Image.fromarray((render.cpu().numpy() * 255).round().astype(np.uint8)).save(
                carpeta_renders / (Path(v["nombre"]).stem + ".png"))
        m = calcular_metricas(render, v["imagen"])
        filas.append({"vista": v["nombre"], **m})
        if mostrar:
            print(f"  {v['nombre']}: PSNR {m['psnr']:.2f} | SSIM {m['ssim']:.4f} | "
                  f"LPIPS {m['lpips']:.4f}")
    return filas


def vistas_no_usadas(info, carpeta_imagenes=None):
    """Todas las imágenes de la escena que no se usaron para entrenar."""
    base = Path(carpeta_imagenes) if carpeta_imagenes else RAIZ / "data" / info["escena"]
    usadas = set(info["entrenamiento"])
    return sorted(p.name for p in (base / "images").iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.name not in usadas)
