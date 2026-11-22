"""Auditoría 2D: pruebas de cordura del SPC sobre imágenes reales y nuevo barrido de λ.

Pruebas (rejilla 256x512, en CPU, sin escribir en las condiciones existentes):
  S1  100 % de mediciones, sin ruido: la directa debe recuperar la imagen de la rejilla
      (error de precisión numérica); la TV, casi.
  S2  Tubería completa con 100 % (directa), a resolución completa: debe coincidir con el
      control (solo remuestreo).
  S3  Operador equivocado: medir con el patrón aleatorio de semilla 449 y reconstruir con
      el de semilla 450. Debe fallar.
  S4  Mediciones asignadas a patrones equivocados (vector y permutado al azar). Debe fallar.
  S5  El patrón aleatorio incluye el coeficiente de intensidad total (DC) en todas las razones.
  S6  Ruido de Poisson: el PSNR debe subir con el número de fotones.
  S7  Referencias triviales: imagen negra (y = 0) y solo el DC (imagen plana del color medio).
  L   Barrido de λ de la TV en 3 vistas que no están entre las 24 del experimento y cuya
      cámara está lejos (> 1.0) de las 3 de prueba.

Escribe results/auditoria/cordura_2d.csv y results/auditoria/barrido_lambda.csv (si ya existen,
se detiene salvo --forzar).
Uso: python scripts/auditoria/cordura_2d.py [--forzar]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.colmap_io import leer_modelo  # noqa: E402
from src.make_dataset import cargar_imagen, degradar, remuestrear, seleccionar_vistas  # noqa: E402
from src.patterns import indices_medidos  # noqa: E402
from src.recon import reconstruir  # noqa: E402
from src.spc import OperadorSPC, agregar_ruido  # noqa: E402
from src.salidas import proteger  # noqa: E402

REJILLA = (256, 512)
CARPETA = RAIZ / "data" / "truck" / "images"


def metricas(pred, ref):
    a = pred.clamp(0, 1).permute(1, 2, 0).double().numpy()
    b = ref.clamp(0, 1).permute(1, 2, 0).double().numpy()
    p = peak_signal_noise_ratio(b, a, data_range=1.0) if not np.array_equal(a, b) else np.inf
    s = structural_similarity(b, a, data_range=1.0, channel_axis=2, gaussian_weights=True,
                              sigma=1.5, use_sample_covariance=False)
    return p, s


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forzar", action="store_true", help="sobrescribe las tablas si existen")
    auditoria = RAIZ / "results" / "auditoria"
    proteger(auditoria / "cordura_2d.csv", auditoria / "barrido_lambda.csv",
             forzar=p.parse_args().forzar)
    torch.manual_seed(449)
    nombres = sorted(p.name for p in CARPETA.iterdir())
    ent, prueba = seleccionar_vistas(nombres, 24, 3)
    vistas = [ent[0], ent[10], ent[20]]
    filas = []

    def anotar(prueba_id, descripcion, vista, pred, ref, esperado):
        p, s = metricas(pred, ref)
        filas.append(dict(prueba=prueba_id, descripcion=descripcion, vista=vista,
                          PSNR=round(p, 2), SSIM=round(s, 4), esperado=esperado))

    for n in vistas:
        x = cargar_imagen(CARPETA / n)
        g = remuestrear(x, REJILLA)
        for patron in ["hadamard", "random", "dct"]:
            op = OperadorSPC(patron, REJILLA, 1.0)
            y = op.medir(g)
            anotar("S1", f"{patron}, 100 %, directa vs rejilla", n,
                   reconstruir(y, op, "directa"), g, "muy alto (> 60 dB)")
            anotar("S1", f"{patron}, 100 %, TV vs rejilla", n,
                   reconstruir(y, op, "tv"), g, "alto (TV con λ>0 suaviza un poco)")
        op = OperadorSPC("hadamard", REJILLA, 1.0)
        control = remuestrear(g, tuple(x.shape[-2:]))
        spc100 = degradar(x, "spc", REJILLA, op, "directa")
        anotar("S2", "hadamard 100 % directa, tubería completa vs control", n, spc100, control,
               "muy alto (igual al control)")
        anotar("S2", "hadamard 100 % directa, tubería completa vs original", n, spc100, x,
               "≈ control vs original")
        anotar("S2", "control vs original (referencia)", n, control, x, "27.2 dB en promedio")

        bueno = OperadorSPC("random", REJILLA, 0.25, semilla=449)
        malo = OperadorSPC("random", REJILLA, 0.25, semilla=450)
        y = bueno.medir(g)
        for metodo in ["directa", "tv"]:
            anotar("S3", f"random 25 %, {metodo}, operador correcto", n,
                   reconstruir(y, bueno, metodo), g, "referencia")
            anotar("S3", f"random 25 %, {metodo}, operador de otra semilla", n,
                   reconstruir(y, malo, metodo), g, "falla (mucho menor)")
        op = OperadorSPC("hadamard", REJILLA, 0.25)
        y = op.medir(g)
        y_perm = y[..., torch.randperm(y.shape[-1])]
        anotar("S4", "hadamard 25 %, directa, y correcto", n, reconstruir(y, op, "directa"), g,
               "referencia")
        anotar("S4", "hadamard 25 %, directa, y permutado al azar", n,
               reconstruir(y_perm, op, "directa"), g, "falla (mucho menor)")

        op = OperadorSPC("hadamard", REJILLA, 1.0)
        y = op.medir(g)
        for fotones in [1e2, 1e3, 1e4, 1e5]:
            anotar("S6", f"hadamard 100 %, directa, Poisson {fotones:.0e} fotones", n,
                   reconstruir(agregar_ruido(y, fotones=fotones), op, "directa"), g,
                   "sube con los fotones")
        anotar("S7", "imagen negra (y = 0) vs original", n, torch.zeros_like(x), x,
               "piso trivial")
        plano = x.mean(dim=(1, 2), keepdim=True).expand_as(x)
        anotar("S7", "solo DC: imagen plana del color medio vs original", n, plano, x,
               "piso trivial")

    for r in [0.05, 0.10, 0.25, 0.50]:
        idx = indices_medidos("random", REJILLA, r)
        filas.append(dict(prueba="S5", descripcion=f"random {int(r * 100)} %: incluye DC (índice 0)",
                          vista="-", PSNR=np.nan, SSIM=np.nan,
                          esperado="sí" if 0 in set(idx.tolist()) else "NO"))
    tabla = pd.DataFrame(filas)
    print(tabla.groupby(["prueba", "descripcion", "esperado"], sort=False)[["PSNR", "SSIM"]]
          .mean().round(3).to_string())
    (RAIZ / "results" / "auditoria").mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RAIZ / "results" / "auditoria" / "cordura_2d.csv", index=False)

    # --- Barrido de lambda en vistas lejanas a las de prueba -----------------------
    _, imgs, _, _ = leer_modelo(RAIZ / "data" / "truck" / "sparse" / "0")
    centro = {k: np.linalg.inv(v["w2c"])[:3, 3] for k, v in imgs.items()}
    candidatas = [n for n in nombres if n not in ent + prueba and
                  min(np.linalg.norm(centro[n] - centro[t]) for t in prueba) > 1.0]
    lejanas = [candidatas[i] for i in np.round(np.linspace(0, len(candidatas) - 1, 3)).astype(int)]
    print("\nvistas para el barrido de λ:", lejanas)
    barrido = []
    for n in lejanas:
        x = cargar_imagen(CARPETA / n)
        for patron in ["hadamard", "random"]:
            for r in [0.05, 0.25]:
                op = OperadorSPC(patron, REJILLA, r)
                for lam in [0.0003, 0.001, 0.003, 0.01, 0.03]:
                    z = degradar(x, "spc", REJILLA, op, "tv", {"lam": lam, "iteraciones": 100})
                    p, s = metricas(z, x)
                    barrido.append(dict(vista=n, patron=patron, razon=r, lam=lam, PSNR=p, SSIM=s))
    b = pd.DataFrame(barrido)
    b.to_csv(RAIZ / "results" / "auditoria" / "barrido_lambda.csv", index=False)
    media = b.groupby(["patron", "razon", "lam"])[["PSNR", "SSIM"]].mean().round(3)
    print(media.to_string())
    mejor = media.reset_index().loc[lambda d: d.groupby(["patron", "razon"])["PSNR"].idxmax()]
    elegido = media.reset_index().query("lam == 0.003")
    comp = mejor.merge(elegido, on=["patron", "razon"], suffixes=("_mejor", "_0.003"))
    comp["perdida_dB"] = (comp["PSNR_mejor"] - comp["PSNR_0.003"]).round(3)
    print("\nλ = 0.003 frente al mejor λ del barrido (promedio de 3 vistas lejanas):")
    print(comp[["patron", "razon", "lam_mejor", "PSNR_mejor", "PSNR_0.003", "perdida_dB"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
