"""Crea la carpeta de imágenes de una condición experimental.

Para cada condición se genera:

    data/conditions/<escena>/<condicion>/
        images/      imágenes con LOS MISMOS NOMBRES que en data/<escena>/images
        sparse  ->   enlace simbólico a data/<escena>/sparse (mismas cámaras)
        vistas.json  vistas de entrenamiento y de prueba, y la configuración
        metricas_2d.csv  métricas 2D por vista (contra la imagen original)

Condiciones:
- original: enlaces simbólicos a las imágenes originales (baseline).
- control:  imagen -> rejilla (p. ej. 256x512) -> tamaño original, sin SPC.
            Mide cuánto se pierde solo por el remuestreo.
- spc:      imagen -> rejilla -> medición y = Phi x (razón r) ->
            reconstrucción (directa o TV) -> tamaño original.

Además, cada ejecución (salvo "original") agrega una fila con el promedio de
las métricas 2D a results/metrics_2d.csv.

Ejemplos:
    python -m src.make_dataset --condicion original
    python -m src.make_dataset --condicion control --rejilla 256x512
    python -m src.make_dataset --condicion spc --patron hadamard --metodo tv --razon 0.10
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .metrics import calcular_metricas
from .patterns import SEMILLA
from .recon import reconstruir
from .spc import OperadorSPC, medir_imagen

RAIZ = Path(__file__).resolve().parents[1]
CALIDAD_JPEG = 100  # se guarda en JPEG (mismo nombre .jpg) con calidad máxima


# --- Selección de vistas ----------------------------------------------------
def seleccionar_vistas(nombres, n_vistas=24, n_prueba=3):
    """Elige n_vistas equiespaciadas (en orden de nombre) y, entre ellas,
    n_prueba vistas de prueba también equiespaciadas y lejos de los extremos.

    Con 251 imágenes, 24 vistas y 3 de prueba: las de prueba son las vistas
    en las posiciones 4, 12 y 20 de la selección; las otras 21 entrenan.
    """
    nombres = sorted(nombres)
    n_vistas = min(n_vistas, len(nombres))
    idx = np.round(np.linspace(0, len(nombres) - 1, n_vistas)).astype(int)
    vistas = [nombres[i] for i in idx]
    paso = n_vistas // n_prueba
    prueba = [vistas[paso // 2 + k * paso] for k in range(n_prueba)]
    entrenamiento = [v for v in vistas if v not in prueba]
    return entrenamiento, prueba


def nombre_condicion(condicion, rejilla, patron=None, metodo=None, razon=None):
    h, w = rejilla
    if condicion == "original":
        return "original"
    if condicion == "control":
        return f"control_{h}x{w}"
    return f"{patron}_{metodo}_{int(round(razon * 100)):02d}_{h}x{w}"


# --- Entrada / salida de imágenes --------------------------------------------
def cargar_imagen(ruta):
    """Lee una imagen como tensor float (3, H, W) en [0, 1]."""
    a = np.asarray(Image.open(ruta).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def guardar_imagen(x, ruta):
    """Guarda un tensor (3, H, W) en [0, 1] como JPEG de calidad máxima (4:4:4)."""
    a = (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(a).save(ruta, quality=CALIDAD_JPEG, subsampling=0)


def remuestrear(x, tamano):
    """Cambia el tamaño de (3, H, W).

    - Si se reducen ambas dimensiones: promedio por área (evita aliasing).
    - Si se amplían ambas: bicúbica.
    - Caso mixto (una se reduce y la otra se amplía, como 979x546 -> 1024x512):
      bicúbica con antialias. Antes se usaba "area" según el alto, que amplía
      mal el ancho; con la rejilla 512x1024 eso bajaba el control de 38.0 a 30.6 dB.
    """
    reduce_alto, reduce_ancho = tamano[0] <= x.shape[-2], tamano[1] <= x.shape[-1]
    if reduce_alto and reduce_ancho:
        y = F.interpolate(x[None], size=tamano, mode="area")
    else:
        # antialias solo en el caso mixto: en PyTorch también cambia el núcleo bicúbico
        # al ampliar, y así la ampliación pura queda igual que en los resultados ya medidos.
        y = F.interpolate(x[None], size=tamano, mode="bicubic", align_corners=False,
                          antialias=reduce_alto or reduce_ancho)
    return y[0].clamp(0, 1)


# --- Procesamiento de una imagen ---------------------------------------------
def degradar(x, condicion, rejilla, operador=None, metodo=None, opciones_tv=None,
             ruido=None):
    """Aplica la condición a una imagen (3, H, W) y devuelve (3, H, W)."""
    tamano_original = x.shape[-2:]
    g = remuestrear(x, rejilla)  # a la rejilla del single pixel
    if condicion == "spc":
        y = medir_imagen(g, operador, ruido)               # medición y = Phi x
        g = reconstruir(y, operador, metodo, **(opciones_tv or {}))
        g = g.clamp(0, 1)
    return remuestrear(g, tamano_original)  # de vuelta al tamaño original


def generar_condicion(escena, tipo, rejilla, entrenamiento, prueba, origen, destino,
                      patron="hadamard", metodo="directa", razon=0.10, semilla=SEMILLA,
                      lam=0.003, iter_tv=100, iter_prox=None, ruido=None, device=None):
    """Genera la carpeta de una condición (núcleo común de este script y de src/cli.py).

    Escribe en `destino`: images/ (o enlaces, si tipo = "original"), el enlace sparse al modelo
    COLMAP de `origen`, vistas.json y metricas_2d.csv (métricas por vista contra la original).
    Sobrescribe lo que haya en `destino`: quien llama decide si se puede (src/cli.py no lo
    llama nunca sobre una condición existente).

    `iter_prox` (iteraciones de Chambolle por paso de FISTA) se registra solo si se indica;
    None usa el valor por defecto de recon.reconstruir_tv (20).
    Devuelve dict(info, filas, tiempo_s, mediciones, operador).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    vistas = entrenamiento + prueba
    cond = nombre_condicion(tipo, rejilla, patron, metodo, razon)
    (destino / "images").mkdir(parents=True, exist_ok=True)

    # Enlace simbólico al mismo modelo COLMAP: todas las condiciones comparten
    # exactamente las mismas cámaras.
    enlace = destino / "sparse"
    if not enlace.exists():
        os.symlink((origen / "sparse").resolve(), enlace, target_is_directory=True)

    info = {"escena": escena, "condicion": cond, "tipo": tipo,
            "patron": patron if tipo == "spc" else tipo,
            "metodo": metodo if tipo == "spc" else "-",
            "razon": razon if tipo == "spc" else 1.0,
            "rejilla": "-" if tipo == "original" else f"{rejilla[0]}x{rejilla[1]}",
            "semilla": semilla, "lam_tv": lam if metodo == "tv" else None,
            "iter_tv": iter_tv if metodo == "tv" else None}
    if iter_prox is not None and metodo == "tv":
        info["iter_prox"] = iter_prox
    if ruido and tipo == "spc":
        info["ruido"] = ruido  # solo se registra si está activo (apagado en los experimentos)
    info.update({"entrenamiento": entrenamiento, "prueba": prueba})
    with open(destino / "vistas.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    if tipo == "original":
        for n in vistas:
            ruta = destino / "images" / n
            if not ruta.exists():
                os.symlink((origen / "images" / n).resolve(), ruta)
        return {"info": info, "filas": [], "tiempo_s": 0.0, "mediciones": 0, "operador": None}

    operador = None
    if tipo == "spc":
        operador = OperadorSPC(patron, rejilla, razon, semilla, device)
    opciones_tv = {"lam": lam, "iteraciones": iter_tv}
    if iter_prox is not None:
        opciones_tv["iter_prox"] = iter_prox

    filas = []
    t0 = time.time()
    for n in vistas:
        x = cargar_imagen(origen / "images" / n).to(device)
        z = degradar(x, tipo, rejilla, operador, metodo, opciones_tv, ruido)
        guardar_imagen(z, destino / "images" / n)
        # Métricas 2D sobre la imagen guardada (lo que verá 3DGS) vs la original.
        z_guardada = cargar_imagen(destino / "images" / n)
        m = calcular_metricas(z_guardada, x.cpu(), device)
        filas.append({"vista": n, "conjunto": "prueba" if n in prueba else "entrenamiento", **m})
    dt = time.time() - t0

    with open(destino / "metricas_2d.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    return {"info": info, "filas": filas, "tiempo_s": dt,
            "mediciones": operador.num_mediciones if operador else 0, "operador": operador}


def crear_condicion(args):
    """Interfaz de línea de comandos heredada: escribe en data/conditions/<escena>/ y agrega una
    fila a results/metrics_2d.csv (la CLI nueva, src/cli.py, escribe en results/<experimento>/)."""
    rejilla = tuple(int(v) for v in args.rejilla.lower().split("x"))
    origen = RAIZ / "data" / args.escena
    nombres = sorted(p.name for p in (origen / "images").iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    entrenamiento, prueba = seleccionar_vistas(nombres, args.vistas, args.prueba)
    vistas = entrenamiento + prueba

    cond = nombre_condicion(args.condicion, rejilla, args.patron, args.metodo, args.razon)
    destino = RAIZ / "data" / "conditions" / args.escena / cond
    ruido = None
    if args.fotones or args.sigma_lectura or args.bits:
        ruido = {"fotones": args.fotones, "sigma_lectura": args.sigma_lectura,
                 "bits": args.bits, "semilla": args.semilla}
    r = generar_condicion(args.escena, args.condicion, rejilla, entrenamiento, prueba, origen,
                          destino, args.patron, args.metodo, args.razon, args.semilla, args.lam,
                          args.iter_tv, None, ruido)
    if args.condicion == "original":
        print(f"[original] {len(vistas)} enlaces en {destino}")
        return destino

    info, filas, dt = r["info"], r["filas"], r["tiempo_s"]
    media = {k: float(np.mean([f[k] for f in filas])) for k in ("psnr", "ssim", "lpips")}
    fila = {"escena": args.escena, "patron": info["patron"], "metodo": info["metodo"],
            "razon": info["razon"], "rejilla": info["rejilla"], "vistas": len(vistas),
            "semilla": args.semilla, "PSNR": round(media["psnr"], 4),
            "SSIM": round(media["ssim"], 4), "LPIPS": round(media["lpips"], 4),
            "mediciones": r["mediciones"],
            "ruido": "si" if ruido else "no", "tiempo_s": round(dt, 1)}
    agregar_fila_csv(RAIZ / "results" / "metrics_2d.csv", fila)
    print(f"[{cond}] {len(vistas)} vistas en {dt:.1f} s | PSNR {media['psnr']:.2f} dB | "
          f"SSIM {media['ssim']:.4f} | LPIPS {media['lpips']:.4f}")
    return destino


def agregar_fila_csv(ruta, fila):
    """Agrega una fila a un CSV (ver agregar_filas_csv)."""
    agregar_filas_csv(ruta, [fila])


def agregar_filas_csv(ruta, nuevas):
    """Agrega filas (lista de dict) a un CSV; crea el archivo y el encabezado si
    no existe.

    Si las filas traen columnas nuevas, se reescribe el archivo con la unión de
    columnas (las filas antiguas quedan vacías en las columnas nuevas).
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    columnas = list(dict.fromkeys(c for fila in nuevas for c in fila))
    filas = []
    if ruta.exists():
        with open(ruta, newline="") as f:
            lector = csv.DictReader(f)
            previas = lector.fieldnames or []
            filas = list(lector)
        if set(columnas) <= set(previas):  # caso normal: mismo encabezado
            with open(ruta, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=previas).writerows(nuevas)
            return
        columnas = previas + [c for c in columnas if c not in previas]
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas + nuevas)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--escena", default="truck")
    p.add_argument("--condicion", choices=["original", "control", "spc"], required=True)
    p.add_argument("--patron", choices=["hadamard", "hadamard_cc", "random", "dct"], default="hadamard")
    p.add_argument("--metodo", choices=["directa", "tv"], default="directa")
    p.add_argument("--razon", type=float, default=0.10, help="fracción medida, en (0, 1]")
    p.add_argument("--rejilla", default="256x512", help="alto x ancho, potencias de 2")
    p.add_argument("--vistas", type=int, default=24)
    p.add_argument("--prueba", type=int, default=3)
    p.add_argument("--semilla", type=int, default=SEMILLA)
    p.add_argument("--lam", type=float, default=0.003,
                   help="peso de la TV (elegido con un barrido en la vista 000130, ver README)")
    p.add_argument("--iter-tv", type=int, default=100, help="iteraciones de FISTA")
    p.add_argument("--fotones", type=float, default=None, help="ruido de Poisson")
    p.add_argument("--sigma-lectura", type=float, default=0.0, help="ruido de lectura")
    p.add_argument("--bits", type=int, default=None, help="cuantización (p. ej. 16)")
    crear_condicion(p.parse_args())


if __name__ == "__main__":
    main()
