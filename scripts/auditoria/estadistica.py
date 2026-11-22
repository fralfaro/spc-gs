"""Auditoría estadística: ¿qué diferencias son concluyentes?

3D: recalcula las métricas POR VISTA de prueba a partir de los renders ya guardados
(outputs/<escena>/<condicion>/<corrida>/renders/*.png, 8 bits) contra las imágenes
ORIGINALES, para todas las corridas con semilla 449. Con esas métricas calcula
diferencias pareadas (misma vista) con su intervalo de confianza del 95 %
(t de Student, n - 1 grados de libertad).

2D: usa las métricas por vista de data/conditions/<escena>/<cond>/metricas_2d.csv
(24 vistas) e intervalos t y bootstrap (10 000 remuestreos, semilla 449).

No modifica ningún resultado existente; escribe en results/auditoria/ (si las tablas ya existen,
se detiene salvo --forzar; la caché de métricas por vista se reutiliza si existe).
Uso: python scripts/auditoria/estadistica.py [--forzar]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.metrics import calcular_metricas  # noqa: E402
from src.salidas import proteger  # noqa: E402

SALIDA = RAIZ / "results" / "auditoria"


def intervalo_t(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    m = d.mean()
    if n < 2:
        return m, np.nan, np.nan
    semi = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return m, m - semi, m + semi


def intervalo_bootstrap(d, b=10000, semilla=449):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(semilla)
    medias = d[rng.integers(0, len(d), (b, len(d)))].mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def metricas_por_vista_3d():
    ruta = SALIDA / "metricas_3d_por_vista_desde_renders.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    originales = RAIZ / "data" / "truck" / "images"
    filas = []
    for carpeta in sorted((RAIZ / "outputs" / "truck").glob("*/it*_s449_f1/renders")):
        corrida = carpeta.parent
        it = int(corrida.name.split("_")[0][2:])
        for png in sorted(carpeta.glob("*.png")):
            pred = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64) / 255
            ref = np.asarray(Image.open(originales / (png.stem + ".jpg")).convert("RGB"),
                             dtype=np.float64) / 255
            m = calcular_metricas(pred, ref, "cpu")
            filas.append(dict(condicion=corrida.parent.name, iteraciones=it,
                              vista=png.stem, PSNR=m["psnr"], SSIM=m["ssim"], LPIPS=m["lpips"]))
    df = pd.DataFrame(filas)
    df.to_csv(ruta, index=False)
    return df


def comparar(df, a, b, it, metrica):
    """Diferencia pareada a - b (por vista) en las vistas comunes."""
    x = df[(df.condicion == a) & (df.iteraciones == it)].set_index("vista")[metrica]
    y = df[(df.condicion == b) & (df.iteraciones == it)].set_index("vista")[metrica]
    comunes = x.index.intersection(y.index)
    d = (x[comunes] - y[comunes]).values
    m, lo, hi = intervalo_t(d)
    return dict(a=a, b=b, iteraciones=it, metrica=metrica, n_vistas=len(d),
                diferencia=round(m, 3), ic95_bajo=round(lo, 3), ic95_alto=round(hi, 3),
                signos_iguales=bool(np.all(d > 0) or np.all(d < 0)),
                concluyente=bool(lo > 0 or hi < 0))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forzar", action="store_true", help="sobrescribe las tablas si existen")
    proteger(SALIDA / "diferencias_3d_prueba.csv", SALIDA / "diferencias_2d.csv",
             forzar=p.parse_args().forzar)
    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    # --- 3D, vistas de prueba -----------------------------------------------------
    df = metricas_por_vista_3d()
    ruta_csv = RAIZ / "results" / "metrics_3d.csv"
    respaldo = RAIZ / "outputs" / "metrics_3d_antes_extendida.csv"  # durante la reevaluación
    csv = pd.read_csv(respaldo if respaldo.exists() else ruta_csv)
    csv = csv[csv.semilla == 449].drop_duplicates(["condicion", "iteraciones"], keep="last")
    medias = df.groupby(["condicion", "iteraciones"])["PSNR"].mean().reset_index()
    chequeo = medias.merge(csv[["condicion", "iteraciones", "PSNR"]], on=["condicion", "iteraciones"])
    print(f"Control: |PSNR desde renders PNG - PSNR del CSV| máx = "
          f"{(chequeo.PSNR_x - chequeo.PSNR_y).abs().max():.3f} dB (cuantización a 8 bits)")

    filas = []
    for it in sorted(df.iteraciones.unique()):
        conds = sorted(df[df.iteraciones == it].condicion.unique())
        for c in conds:
            for ref in ["original", "control_256x512"]:
                if c != ref and ref in conds:
                    for m in ["PSNR", "SSIM", "LPIPS"]:
                        filas.append(comparar(df, c, ref, it, m))
        for c in conds:
            if "_directa_" in c and c.replace("_directa_", "_tv_") in conds:
                for m in ["PSNR", "SSIM", "LPIPS"]:
                    filas.append(comparar(df, c.replace("_directa_", "_tv_"), c, it, m))
    t3 = pd.DataFrame(filas)
    t3.to_csv(SALIDA / "diferencias_3d_prueba.csv", index=False)
    print("\n3D, 3 vistas de prueba: diferencias pareadas con IC 95 % (t, 2 g.l.)")
    resumen = t3.groupby(["metrica"]).agg(comparaciones=("concluyente", "size"),
                                          concluyentes=("concluyente", "sum"),
                                          mismo_signo_en_3_vistas=("signos_iguales", "sum"))
    print(resumen.to_string())
    print("\nEjemplos (7000 iteraciones, PSNR):")
    print(t3[(t3.iteraciones == 7000) & (t3.metrica == "PSNR")].to_string(index=False))
    print("\nConcluyentes en PSNR (todas las iteraciones):")
    print(t3[(t3.metrica == "PSNR") & t3.concluyente].to_string(index=False))

    # Dispersión entre vistas: cuánto varía la métrica de una vista a otra.
    disp = df[df.condicion == "original"].pivot_table(index="vista", columns="iteraciones",
                                                      values="PSNR").round(2)
    print("\nPSNR por vista de prueba del baseline:")
    print(disp.to_string())

    # --- 2D, 24 vistas ----------------------------------------------------------------
    base = RAIZ / "data" / "conditions" / "truck"
    por_vista = {}
    for c in sorted(base.glob("*_256x512")):
        f = c / "metricas_2d.csv"
        if f.exists():
            por_vista[c.name] = pd.read_csv(f).set_index("vista")
    filas = []

    def comparar_2d(a, b):
        for m, col in [("PSNR", "psnr"), ("SSIM", "ssim"), ("LPIPS", "lpips")]:
            d = (por_vista[a][col] - por_vista[b][col]).values
            mm, lo, hi = intervalo_t(d)
            blo, bhi = intervalo_bootstrap(d)
            filas.append(dict(a=a, b=b, metrica=m, n_vistas=len(d), diferencia=round(mm, 3),
                              ic95_t=f"[{lo:.3f}, {hi:.3f}]",
                              ic95_bootstrap=f"[{blo:.3f}, {bhi:.3f}]",
                              concluyente=bool(lo > 0 or hi < 0)))

    for c in por_vista:
        if c.startswith("control"):
            continue
        comparar_2d(c, "control_256x512")
        if "_directa_" in c:
            comparar_2d(c.replace("_directa_", "_tv_"), c)
    for metodo in ["directa", "tv"]:
        for pct in ["05", "10", "25", "50"]:
            comparar_2d(f"hadamard_{metodo}_{pct}_256x512", f"random_{metodo}_{pct}_256x512")
            comparar_2d(f"dct_{metodo}_{pct}_256x512", f"hadamard_{metodo}_{pct}_256x512")
    t2 = pd.DataFrame(filas)
    t2.to_csv(SALIDA / "diferencias_2d.csv", index=False)
    print("\n2D, 24 vistas: comparaciones PSNR (IC 95 % t y bootstrap)")
    print(t2[t2.metrica == "PSNR"].to_string(index=False))
    print("\n2D, PSNR: no concluyentes:")
    print(t2[(t2.metrica == "PSNR") & ~t2.concluyente].to_string(index=False))


if __name__ == "__main__":
    main()
