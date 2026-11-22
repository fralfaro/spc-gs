"""Verificación de las dudas de la Fase 0 (inventario del informe).

Solo lee el repositorio (data/, data/conditions/ y results/) y escribe en una carpeta de
salida nueva; no modifica ningún resultado existente ni entrena 3DGS. Partes:

  1  Reconstrucción directa del patrón aleatorio: variantes con y sin la componente DC y con
     y sin el factor n/m, comparadas con la predicción teórica de una proyección ortogonal.
     Métricas en la rejilla 256x512 contra la imagen remuestreada; 24 vistas.
  2  PSNR 2D promedio de 24 vistas frente al de las 21 vistas de entrenamiento.
  3  Techo de la métrica 2D: contra la original (979x546), imagen guardada llevada a la
     rejilla, y reconstrucción pura en la rejilla (recalcula las 24 condiciones SPC de
     256x512 en las 24 vistas: unos 5 min en una RTX 4060, < 1 GB de memoria).
  4  Métricas 3D: semillas 449/450/451, diferencias pareadas en 3 vistas de prueba y en las
     230 vistas no usadas (IC t, bootstrap por bloques y submuestreo de 1 de cada 8).
  5  Vistas usadas para elegir λ y su distancia a las vistas evaluadas.
  6  Valores que toma cada patrón (¿es realizable con un DMD binario?).

Toda la aleatoriedad usa --semilla (449 por defecto): patrones aleatorios, la variante sin
DC (semilla + 1, como src/patterns.py) y el bootstrap.

Uso (desde la raíz del repositorio):
    python scripts/auditoria/verificacion_fase0.py                  # todas las partes
    python scripts/auditoria/verificacion_fase0.py --partes 2,4,5,6  # solo las rápidas
Salida por defecto: results/auditoria/fase0/ (se detiene si ya existe, salvo --forzar).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.colmap_io import leer_modelo  # noqa: E402
from src.make_dataset import cargar_imagen, remuestrear, seleccionar_vistas  # noqa: E402
from src.patterns import SEMILLA as SEMILLA_POR_DEFECTO, patron_2d  # noqa: E402
from src.recon import reconstruir  # noqa: E402
from src.spc import OperadorSPC  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--salida", default="results/auditoria/fase0",
               help="carpeta de salida, relativa a la raíz del repositorio")
p.add_argument("--partes", default="1,2,3,4,5,6")
p.add_argument("--semilla", type=int, default=SEMILLA_POR_DEFECTO)
p.add_argument("--forzar", action="store_true", help="permite escribir en una carpeta existente")
args = p.parse_args()
SAL = Path(args.salida) if Path(args.salida).is_absolute() else RAIZ / args.salida
if SAL.exists() and any(SAL.iterdir()) and not args.forzar:
    sys.exit(f"Ya existe {SAL} con archivos; use otra --salida o --forzar.")
SAL.mkdir(parents=True, exist_ok=True)
PARTES = set(args.partes.split(","))
SEMILLA = args.semilla
print(f"semilla = {SEMILLA}; salida = {SAL}")

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 40)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CARPETA = RAIZ / "data" / "truck" / "images"
NOMBRES = sorted(q.name for q in CARPETA.iterdir())
ENT, PRU = seleccionar_vistas(NOMBRES, 24, 3)
VISTAS = ENT + PRU
REJ = (256, 512)
N = REJ[0] * REJ[1]


def a_np(t):
    return t.detach().clamp(0, 1).permute(1, 2, 0).double().cpu().numpy()


def psnr_ssim(pred, ref, mapa=False):
    a, b = a_np(pred), a_np(ref)
    ps = peak_signal_noise_ratio(b, a, data_range=1.0)
    r = structural_similarity(b, a, data_range=1.0, channel_axis=2, gaussian_weights=True,
                              sigma=1.5, use_sample_covariance=False, full=mapa)
    return (ps, *r) if mapa else (ps, r)


def titulo(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


# ---------------------------------------------------------------------------------------
if "1" in PARTES:
    titulo("1) Reconstrucción directa del patrón aleatorio: escala y componente DC")
    filas = []
    for n in VISTAS:
        x = cargar_imagen(CARPETA / n).to(DEV)
        g = remuestrear(x, REJ)
        media = g.mean(dim=(1, 2), keepdim=True)
        # Desviación local de la referencia (ventana gaussiana como SSIM) para separar
        # zonas planas y con textura.
        gris = a_np(g).mean(-1)
        from scipy.ndimage import gaussian_filter
        sig_local = np.sqrt(np.clip(gaussian_filter(gris ** 2, 1.5) - gaussian_filter(gris, 1.5) ** 2, 0, None))
        cortes = np.quantile(sig_local, [1 / 3, 2 / 3])
        tercil = np.digitize(sig_local, cortes)  # 0 = plano, 1 = medio, 2 = textura
        ps, ss = psnr_ssim(media.expand_as(g), g)
        filas.append(dict(vista=n, variante="solo DC (imagen plana)", razon=0.0, PSNR=ps, SSIM=ss))
        for r in [0.05, 0.10, 0.25, 0.50]:
            op = OperadorSPC("random", REJ, r, semilla=SEMILLA, device=DEV)
            m = op.num_mediciones
            y = op.medir(g)
            xd = reconstruir(y, op, "directa")             # como en el experimento
            dc = media.expand_as(g)                          # la parte DC de xd es la media exacta
            var = {"con DC, sin escalar (experimento)": xd,
                   "con DC, AC escalado por n/m": dc + (N / m) * (xd - dc),
                   "con DC, todo escalado por n/m": (N / m) * xd}
            # Sin DC: el mismo número m de coeficientes al azar, excluyendo el coeficiente 0.
            op2 = OperadorSPC("random", REJ, r, semilla=SEMILLA, device=DEV)
            rng = np.random.default_rng(SEMILLA + 1)
            orden = 1 + rng.permutation(N - 1)
            op2.idx = torch.tensor(np.sort(orden[:m]), device=DEV, dtype=torch.long)
            xs = reconstruir(op2.medir(g), op2, "directa")
            var["sin DC, sin escalar"] = xs
            var["sin DC, escalado por n/m"] = (N / m) * xs
            var["sin DC, + media verdadera (oráculo)"] = xs + dc
            hd = OperadorSPC("hadamard", REJ, r, device=DEV)
            var["hadamard ordenado, directa (referencia)"] = reconstruir(hd.medir(g), hd, "directa")
            for nombre, z in var.items():
                ps, ss, mapa = psnr_ssim(z, g, mapa=True)
                mapa = mapa.mean(-1)
                fila = dict(vista=n, variante=nombre, razon=r, PSNR=ps, SSIM=ss,
                            SSIM_plano=mapa[tercil == 0].mean(), SSIM_medio=mapa[tercil == 1].mean(),
                            SSIM_textura=mapa[tercil == 2].mean())
                if nombre.startswith("con DC, sin"):
                    # Predicción teórica: proyección sobre m-1 de n-1 direcciones AC al azar
                    # => E||error||^2 = (1 - (m-1)/(n-1)) * ||x - media||^2.
                    ac = (g - media).pow(2).sum().item()
                    mse_teo = (1 - (m - 1) / (N - 1)) * ac / g.numel()
                    fila["PSNR_teorico"] = 10 * np.log10(1 / mse_teo)
                    # Coeficiente de correlación entre la parte AC reconstruida y la verdadera.
                    a_ = (z - dc).ravel(); b_ = (g - media).ravel()
                    fila["correlacion_AC"] = float((a_ @ b_) / (a_.norm() * b_.norm()))
                    fila["energia_AC_rec/verdadera"] = float(a_.pow(2).sum() / b_.pow(2).sum())
                filas.append(fila)
    d1 = pd.DataFrame(filas)
    d1.to_csv(SAL / "p1_random_directa_por_vista.csv", index=False)
    t1 = d1.groupby(["variante", "razon"], sort=False).mean(numeric_only=True).round(4)
    t1.to_csv(SAL / "p1_random_directa_resumen.csv")
    print("Métricas a la resolución de la rejilla 256x512 contra la imagen remuestreada g "
          "(sin JPEG ni vuelta a 979x546); 24 vistas.")
    print(t1.to_string())

# ---------------------------------------------------------------------------------------
if "2" in PARTES:
    titulo("2) PSNR 2D: 24 vistas frente a 21 de entrenamiento")
    filas = []
    for c in sorted((RAIZ / "data" / "conditions" / "truck").iterdir()):
        f = c / "metricas_2d.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        e = d[d.conjunto == "entrenamiento"]
        q = d[d.conjunto == "prueba"]
        filas.append(dict(condicion=c.name, n=len(d), PSNR_24=d.psnr.mean(), PSNR_21_ent=e.psnr.mean(),
                          PSNR_3_prueba=q.psnr.mean(), dif_21_menos_24=e.psnr.mean() - d.psnr.mean(),
                          SSIM_24=d.ssim.mean(), SSIM_21_ent=e.ssim.mean()))
    d2 = pd.DataFrame(filas).round(4)
    d2.to_csv(SAL / "p2_psnr2d_24_vs_21.csv", index=False)
    print(d2.to_string(index=False))
    print("\n|diferencia 21 - 24| máx = %.3f dB, media = %.3f dB" %
          (d2.dif_21_menos_24.abs().max(), d2.dif_21_menos_24.mean()))
    m2 = pd.read_csv(RAIZ / "results" / "metrics_2d.csv")
    print("\nFilas de metrics_2d.csv por (rejilla):", m2.groupby("rejilla").size().to_dict())
    print("Filas por (patron, metodo, rejilla):")
    print(m2.groupby(["patron", "metodo", "rejilla"]).size().to_string())

# ---------------------------------------------------------------------------------------
if "3" in PARTES:
    titulo("3) Techo de la métrica 2D: contra la original (979x546) y contra la imagen remuestreada")
    filas = []
    base = RAIZ / "data" / "conditions" / "truck"
    for rej_txt in ["256x512", "512x1024"]:
        rej = tuple(int(v) for v in rej_txt.split("x"))
        conds = sorted(c.name for c in base.iterdir() if c.name.endswith(rej_txt))
        for n in VISTAS:
            x = cargar_imagen(CARPETA / n)
            g = remuestrear(x, rej)
            for c in conds:
                z = cargar_imagen(base / c / "images" / n)       # lo que ve 3DGS (JPEG, 979x546)
                p_full, s_full = psnr_ssim(z, x)
                p_rej, s_rej = psnr_ssim(remuestrear(z, rej), g)
                filas.append(dict(rejilla=rej_txt, condicion=c, vista=n, PSNR_vs_original=p_full,
                                  SSIM_vs_original=s_full, PSNR_guardada_en_rejilla=p_rej,
                                  SSIM_guardada_en_rejilla=s_rej))
    d3 = pd.DataFrame(filas)
    # Error "puro" del SPC: reconstrucción en la rejilla contra g, sin JPEG ni remuestreo de vuelta.
    puros = []
    t0 = time.time()
    for n in VISTAS:
        x = cargar_imagen(CARPETA / n).to(DEV)
        g = remuestrear(x, REJ)
        for patron in ["hadamard", "random", "dct"]:
            for r in [0.05, 0.10, 0.25, 0.50]:
                op = OperadorSPC(patron, REJ, r, semilla=SEMILLA, device=DEV)
                y = op.medir(g)
                for metodo in ["directa", "tv"]:
                    kw = {"lam": 0.003, "iteraciones": 100} if metodo == "tv" else {}
                    z = reconstruir(y, op, metodo, **kw).clamp(0, 1)
                    ps, ss = psnr_ssim(z, g)
                    puros.append(dict(rejilla="256x512", vista=n,
                                      condicion=f"{patron}_{metodo}_{int(round(r * 100)):02d}_256x512",
                                      PSNR_puro_en_rejilla=ps, SSIM_puro_en_rejilla=ss))
    print(f"(reconstrucciones puras: {time.time() - t0:.0f} s en {DEV})")
    dp = pd.DataFrame(puros)
    d3 = d3.merge(dp, on=["rejilla", "condicion", "vista"], how="left")
    d3.to_csv(SAL / "p3_techo_por_vista.csv", index=False)
    t3 = d3.groupby(["rejilla", "condicion"]).mean(numeric_only=True)
    # Descomposición del error a 979x546: MSE_total ≈ MSE_control + MSE_SPC (si son ortogonales).
    mse = lambda ps: 10 ** (-ps / 10)  # noqa: E731
    for rej_txt in ["256x512", "512x1024"]:
        mc = mse(d3[(d3.condicion == f"control_{rej_txt}")].PSNR_vs_original).mean()
        sel = t3.index.get_level_values(0) == rej_txt
        t3.loc[sel, "frac_MSE_por_rejilla"] = mc / d3[d3.rejilla == rej_txt].assign(
            m=lambda d: mse(d.PSNR_vs_original)).groupby("condicion").m.mean().values
    t3 = t3.round(3)
    t3.to_csv(SAL / "p3_techo_resumen.csv")
    print(t3.to_string())
    # Rango (máx - mín) entre condiciones SPC a 50 % según la evaluación.
    for rej_txt in ["256x512", "512x1024"]:
        s = t3.loc[rej_txt]
        s = s[s.index.str.contains("_50_") & ~s.index.str.startswith("random_directa")]
        print(f"\n{rej_txt}, 50 % (sin random_directa): rango entre condiciones")
        for col in ["PSNR_vs_original", "PSNR_guardada_en_rejilla", "PSNR_puro_en_rejilla"]:
            if s[col].notna().any():
                print(f"  {col:28s} {s[col].min():7.2f} – {s[col].max():7.2f} dB  (rango {s[col].max() - s[col].min():.2f})")
    # ¿Cambia el orden de las condiciones?
    s = t3.loc["256x512"].dropna(subset=["PSNR_puro_en_rejilla"])
    rho = stats.spearmanr(s.PSNR_vs_original, s.PSNR_puro_en_rejilla)
    print(f"\nSpearman (orden de las 24 condiciones SPC 256x512) vs-original vs puro-en-rejilla: {rho.statistic:.3f}")

# ---------------------------------------------------------------------------------------
if "4" in PARTES:
    titulo("4) Métricas 3D: 3 vistas de prueba frente a 230 vistas no usadas y otras semillas")
    m3 = pd.read_csv(RAIZ / "results" / "metrics_3d.csv")
    pv = pd.read_csv(RAIZ / "results" / "metrics_3d_por_vista.csv")
    print("Filas de metrics_3d.csv por (iteraciones, semilla):")
    print(m3.groupby(["iteraciones", "semilla"]).size().to_string())
    dup = m3.duplicated(["condicion", "iteraciones", "semilla", "factor"]).sum()
    print("filas duplicadas (condicion, iteraciones, semilla, factor):", dup)
    # 4a) Semillas.
    s = m3[m3.iteraciones == 3000].groupby("condicion").filter(lambda d: d.semilla.nunique() == 3)
    tab = s.pivot_table(index="condicion", columns="semilla", values=["PSNR", "PSNR_ext", "PSNR_ent"])
    print("\n4a) 3000 iteraciones, semillas 449/450/451")
    print(tab.round(3).to_string())
    resumen = s.groupby("condicion")[["PSNR", "PSNR_ext", "PSNR_ent"]].agg(["mean", "std"]).round(3)
    print(resumen.to_string())
    resumen.to_csv(SAL / "p4a_semillas.csv")

    # 4b) Diferencias pareadas por vista, con IC: 3 vistas de prueba vs 230 no usadas.
    def bloques(d, largo=10, b=5000, semilla=SEMILLA):
        """Bootstrap por bloques de vistas consecutivas (las vistas vecinas están correlacionadas)."""
        rng = np.random.default_rng(semilla)
        k = int(np.ceil(len(d) / largo))
        inicios = rng.integers(0, len(d) - largo + 1, (b, k))
        idx = (inicios[..., None] + np.arange(largo)).reshape(b, -1)[:, :len(d)]
        return np.percentile(d[idx].mean(1), [2.5, 97.5])

    filas = []
    for it in [3000, 7000]:
        q = pv[(pv.iteraciones == it) & (pv.semilla == 449)]
        conds = sorted(q.condicion.unique())
        for ref in ["original", "control_256x512"]:
            for c in conds:
                if c == ref:
                    continue
                for conj in ["prueba", "extendida"]:
                    a = q[(q.condicion == c) & (q.conjunto == conj)].set_index("vista").PSNR
                    bb = q[(q.condicion == ref) & (q.conjunto == conj)].set_index("vista").PSNR
                    comunes = a.index.intersection(bb.index).sort_values()
                    d = (a[comunes] - bb[comunes]).values
                    mm = d.mean()
                    semi = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                    fila = dict(iteraciones=it, condicion=c, referencia=ref, conjunto=conj, n=len(d),
                                dif=mm, ic_t_bajo=mm - semi, ic_t_alto=mm + semi)
                    if conj == "extendida":
                        lo, hi = bloques(d)
                        fila.update(ic_bloques_bajo=lo, ic_bloques_alto=hi)
                        sub = d[::8]
                        semi8 = stats.t.ppf(0.975, len(sub) - 1) * sub.std(ddof=1) / np.sqrt(len(sub))
                        fila.update(n_1de8=len(sub), dif_1de8=sub.mean(), ic8_bajo=sub.mean() - semi8,
                                    ic8_alto=sub.mean() + semi8)
                    filas.append(fila)
    d4 = pd.DataFrame(filas)
    d4["concluyente_t"] = (d4.ic_t_bajo > 0) | (d4.ic_t_alto < 0)
    d4["concluyente_bloques"] = ((d4.ic_bloques_bajo > 0) | (d4.ic_bloques_alto < 0)).where(d4.conjunto == "extendida")  # sin bloques con 3 vistas
    d4.round(3).to_csv(SAL / "p4b_diferencias_3d.csv", index=False)
    print("\n4b) Diferencias pareadas de PSNR (dB) contra el baseline original, semilla 449")
    print(d4[d4.referencia == "original"].round(3).to_string(index=False))
    print("\nResumen de concluyentes (IC 95 %):")
    print(d4.groupby(["iteraciones", "referencia", "conjunto"])[["concluyente_t", "concluyente_bloques"]]
          .sum().to_string())
    # 4c) ¿Coincide el orden de las condiciones entre 3 vistas y 230?
    for it in [3000, 7000]:
        q = m3[(m3.iteraciones == it) & (m3.semilla == 449)]
        rho = stats.spearmanr(q.PSNR, q.PSNR_ext)
        rho2 = stats.spearmanr(q.PSNR_ent, q.PSNR_ext)
        print(f"\n4c) {it} it: Spearman PSNR(3 prueba) vs PSNR(230 ext) = {rho.statistic:.3f} "
              f"(n = {len(q)}); PSNR(21 ent) vs PSNR(230 ext) = {rho2.statistic:.3f}")
    # Dispersión entre vistas en el baseline (para dimensionar n).
    q = pv[(pv.condicion == "original") & (pv.iteraciones == 3000) & (pv.semilla == 449)]
    for conj in ["prueba", "extendida"]:
        v = q[q.conjunto == conj].PSNR
        print(f"baseline 3000 it, {conj}: n = {len(v)}, media {v.mean():.2f}, de {v.std():.2f}, "
              f"mín {v.min():.2f}, máx {v.max():.2f}")
    # Autocorrelación de las diferencias en vistas consecutivas.
    a = pv[(pv.condicion == "hadamard_tv_05_256x512") & (pv.iteraciones == 3000) & (pv.semilla == 449)
           & (pv.conjunto == "extendida")].set_index("vista").PSNR.sort_index()
    bb = q[q.conjunto == "extendida"].set_index("vista").PSNR.sort_index()
    d = (a - bb).dropna().values
    print(f"autocorrelación lag-1 de la diferencia hadamard_tv_05 - original (230 vistas): "
          f"{np.corrcoef(d[:-1], d[1:])[0, 1]:.3f}")

# ---------------------------------------------------------------------------------------
if "5" in PARTES:
    titulo("5) Vistas usadas para elegir λ")
    _, imgs, _, _ = leer_modelo(RAIZ / "data" / "truck" / "sparse" / "0")
    centro = {k: np.linalg.inv(v["w2c"])[:3, 3] for k, v in imgs.items()}
    print("24 vistas:", VISTAS)
    print("prueba:", PRU)
    for v in ["000130.jpg", "000009.jpg", "000134.jpg", "000250.jpg"]:
        dist = {t: round(float(np.linalg.norm(centro[v] - centro[t])), 3) for t in PRU}
        dist_ent = min(float(np.linalg.norm(centro[v] - centro[t])) for t in ENT)
        print(f"{v}: ¿entre las 24? {v in VISTAS}; distancia de cámara a las de prueba {dist}; "
              f"a la vista de entrenamiento más cercana {dist_ent:.3f}")
    todas = np.array([centro[k] for k in NOMBRES])
    paso = np.linalg.norm(np.diff(todas, axis=0), axis=1)
    print(f"distancia media entre cámaras consecutivas: {paso.mean():.3f} (mediana {np.median(paso):.3f})")
    import json
    for c in ["hadamard_tv_05_256x512", "random_tv_50_256x512", "dct_tv_25_256x512",
              "hadamard_tv_05_512x1024", "dct_tv_50_512x1024"]:
        j = json.loads((RAIZ / "data" / "conditions" / "truck" / c / "vistas.json").read_text())
        print(f"{c}: lam_tv = {j['lam_tv']}, iter_tv = {j['iter_tv']}, semilla = {j['semilla']}")
    b = pd.read_csv(RAIZ / "results" / "auditoria" / "barrido_lambda.csv")
    t = b.groupby(["patron", "razon", "lam"])[["PSNR"]].mean().reset_index()
    mejor = t.loc[t.groupby(["patron", "razon"]).PSNR.idxmax()]
    elegido = t[t.lam == 0.003]
    comp = mejor.merge(elegido, on=["patron", "razon"], suffixes=("_mejor", "_0.003"))
    comp["perdida_dB"] = comp.PSNR_mejor - comp["PSNR_0.003"]
    print("\nBarrido existente en 3 vistas lejanas (results/auditoria/barrido_lambda.csv):")
    print(comp.round(3).to_string(index=False))

# ---------------------------------------------------------------------------------------
if "6" in PARTES:
    titulo("6) Valores de los patrones (¿realizables con un DMD binario?)")
    filas = []
    for patron in ["hadamard", "random", "dct"]:
        for i in [0, 1, 7, 513, 5000, 65535]:
            P = patron_2d(patron, REJ, i)
            u = np.unique(np.round(P * np.sqrt(N), 8))
            filas.append(dict(patron=patron, indice=i, valores_distintos=len(u),
                              min_x_sqrtN=u.min(), max_x_sqrtN=u.max()))
    d6 = pd.DataFrame(filas)
    d6.to_csv(SAL / "p6_valores_patrones.csv", index=False)
    print(d6.round(4).to_string(index=False))
