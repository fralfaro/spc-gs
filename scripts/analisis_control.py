"""Análisis del control de remuestreo: ¿de qué depende su PSNR?

El control es: imagen original -> rejilla -> tamaño original (sin SPC). Este
script mide su PSNR y SSIM, en las 24 vistas del experimento, variando un
factor a la vez:

- la rejilla (256x512 o 512x1024),
- si se guarda en JPEG (calidad 100) o se mantiene en punto flotante,
- la resolución de evaluación: tamaño original (979x546, la del experimento) o
  la propia rejilla (se reducen la original y el control a la rejilla y se
  comparan ahí).

Solo usa CPU y no escribe en results/ salvo la tabla final
results/analisis_control.csv (si ya existe, se detiene salvo --forzar).

Uso:  python scripts/analisis_control.py [--forzar]
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.make_dataset import cargar_imagen, remuestrear, seleccionar_vistas  # noqa: E402
from src.salidas import proteger  # noqa: E402


def a_numpy(x):
    return x.permute(1, 2, 0).numpy().astype(np.float64)


def jpeg_ida_y_vuelta(x):
    """Guarda en JPEG (q100, 4:4:4) en memoria y lo vuelve a leer, como make_dataset."""
    a = (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=100, subsampling=0)
    buf.seek(0)
    b = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    import torch
    return torch.from_numpy(b).permute(2, 0, 1).contiguous()


def metricas(a, b):
    a, b = a_numpy(a), a_numpy(b)
    return (peak_signal_noise_ratio(b, a, data_range=1.0),
            structural_similarity(b, a, channel_axis=2, data_range=1.0,
                                  gaussian_weights=True, sigma=1.5,
                                  use_sample_covariance=False))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forzar", action="store_true", help="sobrescribe la tabla si existe")
    salida = RAIZ / "results" / "analisis_control.csv"
    proteger(salida, forzar=p.parse_args().forzar)
    carpeta = RAIZ / "data" / "truck" / "images"
    nombres = sorted(p.name for p in carpeta.iterdir())
    ent, prueba = seleccionar_vistas(nombres, 24, 3)
    filas = []
    for n in ent + prueba:
        x = cargar_imagen(carpeta / n)           # (3, 546, 979)
        tam = tuple(x.shape[-2:])
        for rejilla in [(256, 512), (512, 1024)]:
            g = remuestrear(x, rejilla)
            vuelta = remuestrear(g, tam)
            for jpeg in [True, False]:
                c = jpeg_ida_y_vuelta(vuelta) if jpeg else vuelta
                # Evaluación a resolución completa (la del experimento).
                p, s = metricas(c, x)
                filas.append(dict(vista=n, rejilla=f"{rejilla[0]}x{rejilla[1]}", jpeg=jpeg,
                                  evaluacion=f"{tam[0]}x{tam[1]}", PSNR=p, SSIM=s))
                # Evaluación a la resolución de la rejilla.
                p, s = metricas(remuestrear(c, rejilla), g)
                filas.append(dict(vista=n, rejilla=f"{rejilla[0]}x{rejilla[1]}", jpeg=jpeg,
                                  evaluacion="rejilla", PSNR=p, SSIM=s))
        # Solo JPEG, sin remuestreo: cota del efecto de la recodificación.
        p, s = metricas(jpeg_ida_y_vuelta(x), x)
        filas.append(dict(vista=n, rejilla="sin remuestreo", jpeg=True,
                          evaluacion=f"{tam[0]}x{tam[1]}", PSNR=p, SSIM=s))
    df = pd.DataFrame(filas)
    tabla = (df.groupby(["rejilla", "jpeg", "evaluacion"], sort=False)[["PSNR", "SSIM"]]
             .agg(["mean", "min", "max"]).round(3))
    print(f"Imagen original: {tam[1]}x{tam[0]} (ancho x alto); {len(ent + prueba)} vistas\n")
    print(tabla.to_string())
    tabla.to_csv(salida)


if __name__ == "__main__":
    main()
