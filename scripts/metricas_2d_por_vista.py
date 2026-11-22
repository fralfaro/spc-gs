"""Métricas 2D por vista, recalculadas desde las reconstrucciones guardadas.

Para cada condición de data/conditions/<escena>/ (salvo "original", que son enlaces a las
imágenes originales), compara cada imagen guardada (JPEG q100 a 979x546: exactamente lo que
recibe 3DGS) con su imagen original, con la misma función `calcular_metricas` que usa
src/make_dataset.py. No entrena ni renderiza 3DGS.

Escribe dos archivos NUEVOS (no modifica ningún CSV existente; si ya existen, se detiene
salvo que se use --forzar):

- results/metrics_2d_por_vista.csv: una fila por (condición, vista), con el conjunto de la
  vista (entrenamiento o prueba) según vistas.json.
- results/auditoria/eje_2d_3d_antes_despues.csv: por condición, el promedio de 24 vistas
  (valor antiguo del eje del gráfico 2D -> 3D, tomado de results/metrics_2d.csv) y el de las
  21 vistas de entrenamiento (valor nuevo), además de la diferencia entre el promedio de 24
  vistas recalculado y el registrado, como control de que el recálculo reproduce lo guardado.

Uso:  python scripts/metricas_2d_por_vista.py [--escena truck] [--forzar]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.make_dataset import cargar_imagen  # noqa: E402
from src.metrics import calcular_metricas  # noqa: E402

METRICAS = ["PSNR", "SSIM", "LPIPS"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--escena", default="truck")
    p.add_argument("--forzar", action="store_true", help="sobrescribe las salidas si existen")
    args = p.parse_args()

    salida_vistas = RAIZ / "results" / "metrics_2d_por_vista.csv"
    salida_comp = RAIZ / "results" / "auditoria" / "eje_2d_3d_antes_despues.csv"
    for ruta in (salida_vistas, salida_comp):
        if ruta.exists() and not args.forzar:
            sys.exit(f"Ya existe {ruta.relative_to(RAIZ)}; use --forzar para regenerarlo.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    originales = RAIZ / "data" / args.escena / "images"
    base = RAIZ / "data" / "conditions" / args.escena
    filas, control = [], []
    for carpeta in sorted(c for c in base.iterdir() if (c / "vistas.json").exists()):
        info = json.loads((carpeta / "vistas.json").read_text())
        if info["tipo"] == "original":
            continue
        guardado = carpeta / "metricas_2d.csv"
        previo = pd.read_csv(guardado).set_index("vista") if guardado.exists() else None
        for conjunto in ("entrenamiento", "prueba"):
            for n in info[conjunto]:
                m = calcular_metricas(cargar_imagen(carpeta / "images" / n),
                                      cargar_imagen(originales / n), device)
                filas.append({"escena": info["escena"], "condicion": info["condicion"],
                              "patron": info["patron"], "metodo": info["metodo"],
                              "razon": info["razon"], "rejilla": info["rejilla"],
                              "semilla": info["semilla"], "lam_tv": info.get("lam_tv"),
                              "iter_tv": info.get("iter_tv"), "conjunto": conjunto, "vista": n,
                              "PSNR": m["psnr"], "SSIM": m["ssim"], "LPIPS": m["lpips"]})
                if previo is not None:
                    control.append({k: abs(m[k.lower()] - previo.loc[n, k.lower()]) for k in METRICAS})
        print(f"[2D por vista] {info['condicion']}: {len(info['entrenamiento'])} + "
              f"{len(info['prueba'])} vistas", flush=True)
    df = pd.DataFrame(filas)

    # Comparación con el promedio de 24 vistas registrado en results/metrics_2d.csv.
    reg = pd.read_csv(RAIZ / "results" / "metrics_2d.csv")
    reg = reg.drop_duplicates(["escena", "patron", "metodo", "razon", "rejilla", "semilla"],
                              keep="last")
    claves = ["escena", "patron", "metodo", "razon", "rejilla", "semilla"]
    todas = df.groupby(["condicion"] + claves)[METRICAS].mean().add_suffix("_24_recalculado")
    ent = (df[df.conjunto == "entrenamiento"].groupby(["condicion"] + claves)[METRICAS].mean()
           .add_suffix("_21_entrenamiento"))
    comp = todas.join(ent).reset_index().merge(
        reg[claves + METRICAS].rename(columns={m: f"{m}_24_metrics_2d_csv" for m in METRICAS}),
        on=claves, how="left")
    for m in METRICAS:
        comp[f"{m}_nuevo_menos_antiguo"] = comp[f"{m}_21_entrenamiento"] - comp[f"{m}_24_metrics_2d_csv"]
        comp[f"{m}_control_recalculo"] = comp[f"{m}_24_recalculado"] - comp[f"{m}_24_metrics_2d_csv"]

    salida_comp.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(salida_vistas, index=False)
    comp.round(4).to_csv(salida_comp, index=False)

    c = pd.DataFrame(control)
    print(f"\n{len(df)} filas -> {salida_vistas.relative_to(RAIZ)}")
    print("Control contra data/conditions/*/metricas_2d.csv (|diferencia| máxima por vista): "
          + ", ".join(f"{k} {c[k].max():.2e}" for k in METRICAS))
    print("Control contra results/metrics_2d.csv (|diferencia| máxima del promedio de 24): "
          + ", ".join(f"{m} {comp[f'{m}_control_recalculo'].abs().max():.2e}" for m in METRICAS))
    print(f"PSNR 21 de entrenamiento - PSNR 24 (dB): mín "
          f"{comp.PSNR_nuevo_menos_antiguo.min():.3f}, máx {comp.PSNR_nuevo_menos_antiguo.max():.3f}, "
          f"media {comp.PSNR_nuevo_menos_antiguo.mean():.3f} -> {salida_comp.relative_to(RAIZ)}")
    if np.isnan(comp.PSNR_24_metrics_2d_csv).any():
        print("Aviso: hay condiciones sin fila en results/metrics_2d.csv.")


if __name__ == "__main__":
    main()
