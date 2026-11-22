"""Compara en 2D los órdenes de Hadamard (secuencia vs cake-cutting) en Truck.

No escribe imágenes ni enlaces simbólicos: reconstruye cada vista en memoria, la pasa por
JPEG q100 (como el experimento) y mide contra la original a resolución completa.

    python scripts/comparar_ordenes_hadamard.py --rejilla 512x1024
"""
import argparse
import csv
import io
import time
from pathlib import Path

import torch
from PIL import Image

from src.make_dataset import RAIZ, cargar_imagen, degradar, seleccionar_vistas
from src.metrics import calcular_metricas
from src.patterns import SEMILLA
from src.spc import OperadorSPC


def jpeg(x):
    a = (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=100, subsampling=0); b.seek(0)
    return torch.from_numpy(__import__("numpy").asarray(Image.open(b), dtype="float32") / 255).permute(2, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rejilla", default="512x1024")
    p.add_argument("--patrones", nargs="+", default=["hadamard", "hadamard_cc"])
    p.add_argument("--metodos", nargs="+", default=["directa", "tv"])
    p.add_argument("--razones", nargs="+", type=float, default=[0.05, 0.10, 0.25, 0.50])
    p.add_argument("--vistas", type=int, default=24)
    p.add_argument("--manifest", default=None, help="manifest.json de inf449 para usar sus mismas 24 vistas")
    p.add_argument("--ruido", action="store_true", help="1e9 fotones, lectura 1e-5, ADC 16 bits (como inf449)")
    p.add_argument("--salida", default="results/orden_hadamard.csv")
    a = p.parse_args()
    rej = tuple(int(v) for v in a.rejilla.split("x"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    origen = RAIZ / "data" / "truck"
    entren, prueba = seleccionar_vistas([f.name for f in (origen / "images").glob("*.jpg")], a.vistas, 3)
    vistas = entren + prueba
    if a.manifest:
        import json
        vistas = json.load(open(a.manifest))["views"]
    imgs = {n: cargar_imagen(origen / "images" / n).to(dev) for n in vistas}
    ruido = {"fotones": 1e9, "sigma_lectura": 1e-5, "bits": 16} if a.ruido else None
    filas = []
    for pat in a.patrones:
        for met in a.metodos:
            for r in a.razones:
                op = OperadorSPC(pat, rej, r, SEMILLA, dev)
                ms = []
                torch.cuda.synchronize() if dev == "cuda" else None
                t0 = time.time()
                recs = {n: degradar(imgs[n], "spc", rej, op, met, {"lam": 0.003, "iteraciones": 100}, ruido) for n in vistas}
                torch.cuda.synchronize() if dev == "cuda" else None
                dt = time.time() - t0
                for n in vistas:
                    ms.append(calcular_metricas(jpeg(recs[n]), imgs[n].cpu(), dev))
                f = {"patron": pat, "metodo": met, "razon": r, "rejilla": a.rejilla, "vistas": len(vistas),
                     "PSNR": sum(m["psnr"] for m in ms) / len(ms), "SSIM": sum(m["ssim"] for m in ms) / len(ms),
                     "LPIPS": sum(m["lpips"] for m in ms) / len(ms), "s_por_vista_recon": dt / len(vistas)}
                filas.append(f); print({k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()}, flush=True)
    out = RAIZ / a.salida
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)


if __name__ == "__main__":
    main()
