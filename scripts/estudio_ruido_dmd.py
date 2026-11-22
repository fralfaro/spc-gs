"""Efecto del ruido del detector y del modelo DMD {0,1} en 2D (Truck, 24 vistas, razón 25 %).

Compara la medición ideal con coeficientes +-1 contra el modelo de un DMD real (máscaras {0,1} más
una medición DC), con ruido de Poisson, de lectura y cuantización de 16 bits, para varios niveles de
fotones. Solo 2D: reconstruye en memoria, pasa por JPEG q100 y mide contra la original.

    python scripts/estudio_ruido_dmd.py --rejilla 256x512
"""
import argparse
import csv
import time

import torch

from scripts.comparar_ordenes_hadamard import jpeg
from src.make_dataset import RAIZ, cargar_imagen, degradar, seleccionar_vistas
from src.metrics import calcular_metricas
from src.patterns import SEMILLA
from src.spc import OperadorSPC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rejilla", default="256x512")
    p.add_argument("--razon", type=float, default=0.25)
    p.add_argument("--patrones", nargs="+", default=["hadamard", "random"])
    p.add_argument("--fotones", nargs="+", type=float, default=[0, 1e9, 1e8, 1e7])  # 0 = sin ruido
    p.add_argument("--salida", default="results/ruido_dmd.csv")
    a = p.parse_args()
    rej = tuple(int(v) for v in a.rejilla.split("x"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    origen = RAIZ / "data" / "truck"
    entren, prueba = seleccionar_vistas([f.name for f in (origen / "images").glob("*.jpg")], 24, 3)
    vistas = entren + prueba
    imgs = {n: cargar_imagen(origen / "images" / n).to(dev) for n in vistas}
    filas = []
    for pat in a.patrones:
        op = OperadorSPC(pat, rej, a.razon, SEMILLA, dev)
        for met in ("directa", "tv"):
            for modelo in ("ideal +-1", "DMD {0,1}"):
                for f in a.fotones:
                    ruido = None
                    if f > 0 or modelo.startswith("DMD"):
                        ruido = {"dmd": modelo.startswith("DMD")}
                        if f > 0:
                            ruido.update(fotones=f, sigma_lectura=1e-5, bits=16)
                        if not ruido["dmd"]:
                            ruido.pop("dmd")
                    if f == 0 and modelo.startswith("DMD") is False:
                        ruido = None
                    ms = []
                    for n in vistas:
                        z = degradar(imgs[n], "spc", rej, op, met, {"lam": 0.003, "iteraciones": 100}, ruido)
                        ms.append(calcular_metricas(jpeg(z), imgs[n].cpu(), dev))
                    fila = {"patron": pat, "metodo": met, "modelo": modelo, "fotones": f, "razon": a.razon, "rejilla": a.rejilla,
                            "PSNR": sum(m["psnr"] for m in ms) / len(ms), "SSIM": sum(m["ssim"] for m in ms) / len(ms),
                            "LPIPS": sum(m["lpips"] for m in ms) / len(ms)}
                    filas.append(fila); print({k: round(v, 3) if isinstance(v, float) else v for k, v in fila.items()}, flush=True)
    with open(RAIZ / a.salida, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)


if __name__ == "__main__":
    main()
