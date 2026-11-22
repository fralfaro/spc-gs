"""Punto de entrada único del proyecto.

    python -m src.cli <subcomando> --config configs/<experimento>.yaml [--forzar]

Subcomandos, en el orden en que se ejecutan:

  dataset      Selecciona las vistas (entrenamiento, prueba y el subconjunto fijo cuyas imágenes
               se guardan) y crea la condición "original" (enlaces a las imágenes originales).
  reconstruir  Para cada condición 2D (control y SPC): simula la medición y = Φx, reconstruye,
               guarda las imágenes que verá 3DGS y calcula las métricas 2D por vista. Guarda
               además las reconstrucciones del subconjunto fijo en results/<experimento>/recon/
               (no versionado) y sus métricas en metrics_2d_guardadas.csv.
  entrenar     Entrena 3DGS en cada condición 3D, iteraciones y semilla de la configuración.
               Reutiliza los modelos que ya existen (si coinciden con la configuración).
  evaluar      Renderiza las vistas de prueba, de entrenamiento y (si evaluacion.extendida) todas
               las no usadas, contra las imágenes ORIGINALES: metrics_3d.csv y
               metrics_3d_por_vista.csv; renders del subconjunto fijo en render_3d/. Nunca entrena.

Reglas (ver docs/esquema_resultados.md):
- No sobrescribe nada en data/: si una condición ya existe con los mismos parámetros, la
  reutiliza; si existe con otros parámetros, se detiene.
- No sobrescribe nada en results/<experimento>/ salvo con --forzar. Si config_usada.yaml ya
  existe con otra configuración, se detiene: un experimento distinto necesita otro nombre.
- Cada ejecución agrega una fila a results/<experimento>/registro_corridas.csv con la fecha,
  el subcomando y el commit de git.
"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from .config import (RAIZ, a_yaml, cargar_config, condiciones_2d, condiciones_3d, ruta,
                     texto_rejilla, vistas_guardadas)
from .make_dataset import (agregar_fila_csv, cargar_imagen, generar_condicion, remuestrear,
                           seleccionar_vistas)
from .metrics import psnr_ssim
from .patterns import numero_mediciones
from .recon import reconstruir as reconstruir_imagen
from .salidas import proteger
from .spc import OperadorSPC, medir_imagen

METRICAS = ["PSNR", "SSIM", "LPIPS"]


# --- Experimento: carpeta, configuración usada y registro --------------------------------------
def estado_git():
    """(commit, hay_cambios_sin_commit); ("desconocido", None) si git no está disponible."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
                                text=True, check=True).stdout.strip()
        cambios = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                 cwd=RAIZ, capture_output=True, text=True, check=True).stdout
        return commit, bool(cambios.strip())
    except (OSError, subprocess.CalledProcessError):
        return "desconocido", None


def preparar_experimento(cfg):
    """Crea results/<experimento>/ y su config_usada.yaml; se detiene si ya existe con otra
    configuración."""
    carpeta = ruta(cfg, "resultados")
    carpeta.mkdir(parents=True, exist_ok=True)
    actual = {k: v for k, v in cfg.items() if k != "archivo"}
    usada = carpeta / "config_usada.yaml"
    if usada.exists():
        previa = yaml.safe_load(usada.read_text(encoding="utf-8"))["configuracion"]
        if previa != actual:
            sys.exit(f"{usada} existe con otra configuración. Un experimento distinto necesita "
                     f"otro nombre ('experimento' en {cfg['archivo']}).")
    else:
        commit, cambios = estado_git()
        texto = a_yaml({"archivo": cfg["archivo"], "creado": dt.datetime.now().isoformat(timespec="seconds"),
                        "commit": commit, "cambios_sin_commit": cambios, "configuracion": actual})
        usada.write_text("# Configuración resuelta de este experimento (escrita por src/cli.py).\n"
                         + texto, encoding="utf-8")
    return carpeta


def registrar(carpeta, subcomando, t0, extra=""):
    commit, cambios = estado_git()
    agregar_fila_csv(carpeta / "registro_corridas.csv", {
        "fecha": dt.datetime.now().isoformat(timespec="seconds"), "subcomando": subcomando,
        "commit": commit, "cambios_sin_commit": cambios, "duracion_s": round(time.time() - t0, 1),
        "detalle": extra})


# --- dataset --------------------------------------------------------------------------------
def leer_vistas(cfg, carpeta):
    """Vistas del experimento (results/<experimento>/vistas.json); las crea si no existen."""
    destino = carpeta / "vistas.json"
    nombres = sorted(p.name for p in (ruta(cfg, "imagenes") / "images").iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    ent, pru = seleccionar_vistas(nombres, cfg["vistas"]["total"], cfg["vistas"]["prueba"])
    vistas = {"semilla": cfg["semilla"], "entrenamiento": ent, "prueba": pru,
              "guardadas": vistas_guardadas(cfg, ent, pru)}
    if destino.exists():
        previas = json.loads(destino.read_text(encoding="utf-8"))
        if previas != vistas:
            sys.exit(f"{destino} no coincide con la selección de vistas de la configuración.")
    else:
        destino.write_text(json.dumps(vistas, indent=2, ensure_ascii=False), encoding="utf-8")
    return vistas


def parametros_esperados(cfg, c, vistas):
    """Campos de vistas.json que deben coincidir para reutilizar una condición existente."""
    tv = c.get("metodo") == "tv" and c["tipo"] == "spc"
    esperado = {"escena": cfg["escena"], "condicion": c["nombre"], "tipo": c["tipo"],
                "razon": c["razon"], "entrenamiento": vistas["entrenamiento"],
                "prueba": vistas["prueba"],
                "rejilla": "-" if c["tipo"] == "original" else texto_rejilla(cfg)}
    if c["tipo"] == "spc":
        esperado.update({"patron": c["patron"], "metodo": c["metodo"], "semilla": cfg["semilla"],
                         "lam_tv": cfg["tv"]["lam"] if tv else None,
                         "iter_tv": cfg["tv"]["iteraciones"] if tv else None,
                         "ruido": ({**cfg["spc"]["ruido"], "semilla": cfg["semilla"]}
                                   if cfg["spc"]["ruido"] else None)})
        if tv:
            esperado["iter_prox"] = cfg["tv"]["iter_prox"]
    return esperado


def comprobar_existente(destino, esperado):
    """True si la condición existe y coincide; None si no existe; se detiene si difiere o si
    está incompleta."""
    if not (destino / "vistas.json").exists():
        if destino.exists() and any(destino.iterdir()):
            sys.exit(f"{destino} existe sin vistas.json (incompleta). Revísela a mano: no se toca.")
        return None
    info = json.loads((destino / "vistas.json").read_text(encoding="utf-8"))
    info.setdefault("ruido", None)
    if esperado.get("metodo") == "tv":
        info.setdefault("iter_prox", 20)  # valor fijo de recon.reconstruir_tv antes de registrarlo
    difs = {k: (info.get(k), v) for k, v in esperado.items() if info.get(k) != v}
    if difs:
        sys.exit(f"La condición {destino} ya existe con otros parámetros (existente, pedido): "
                 f"{difs}. No se sobrescribe; use otra ruta 'condiciones' en la configuración.")
    n = len(esperado["entrenamiento"]) + len(esperado["prueba"])
    if len(list((destino / "images").iterdir())) != n or (
            esperado["tipo"] != "original" and not (destino / "metricas_2d.csv").exists()):
        sys.exit(f"La condición {destino} está incompleta. Revísela a mano: no se toca.")
    return True


def cmd_dataset(cfg, args, carpeta):
    vistas = leer_vistas(cfg, carpeta)
    print(f"[dataset] {len(vistas['entrenamiento'])} vistas de entrenamiento, "
          f"{len(vistas['prueba'])} de prueba; guardadas: {', '.join(vistas['guardadas'])}")
    original = [c for c in condiciones_3d(cfg) if c["tipo"] == "original"]
    for c in original:
        destino = ruta(cfg, "condiciones") / c["nombre"]
        if comprobar_existente(destino, parametros_esperados(cfg, c, vistas)):
            print(f"[dataset] original: se reutiliza {destino}")
        else:
            generar_condicion(cfg["escena"], "original", tuple(cfg["rejilla"]),
                              vistas["entrenamiento"], vistas["prueba"], ruta(cfg, "imagenes"),
                              destino, semilla=cfg["semilla"])
            print(f"[dataset] original: enlaces creados en {destino}")
    return vistas


# --- reconstruir ------------------------------------------------------------------------------
def guardar_png(x, ruta_png):
    a = (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(a).save(ruta_png)


def cmd_reconstruir(cfg, args, carpeta):
    salidas = [carpeta / "metrics_2d.csv", carpeta / "metrics_2d_por_vista.csv",
               carpeta / "metrics_2d_guardadas.csv", carpeta / "recon"]
    proteger(*salidas, forzar=args.forzar)
    vistas = cmd_dataset(cfg, args, carpeta)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rejilla = tuple(cfg["rejilla"])
    imagenes = ruta(cfg, "imagenes")
    ruido = {**cfg["spc"]["ruido"], "semilla": cfg["semilla"]} if cfg["spc"]["ruido"] else None
    opciones_tv = {"lam": cfg["tv"]["lam"], "iteraciones": cfg["tv"]["iteraciones"],
                   "iter_prox": cfg["tv"]["iter_prox"]}
    recon = carpeta / "recon"
    (recon / "referencia").mkdir(parents=True, exist_ok=True)  # existe solo con --forzar

    # Referencias: original y original remuestreada a la rejilla.
    originales, en_rejilla = {}, {}
    for n in vistas["entrenamiento"] + vistas["prueba"]:
        originales[n] = cargar_imagen(imagenes / "images" / n)
        en_rejilla[n] = remuestrear(originales[n], rejilla)
    for n in vistas["guardadas"]:
        shutil.copy2(imagenes / "images" / n, recon / "referencia" / n)
        guardar_png(en_rejilla[n], recon / "referencia" / f"{n.rsplit('.', 1)[0]}_rejilla.png")

    filas, guardadas = [], []
    for c in condiciones_2d(cfg):
        t0 = time.time()
        destino = ruta(cfg, "condiciones") / c["nombre"]
        if comprobar_existente(destino, parametros_esperados(cfg, c, vistas)):
            origen = "reutilizada"
        else:
            generar_condicion(cfg["escena"], c["tipo"], rejilla, vistas["entrenamiento"],
                              vistas["prueba"], imagenes, destino, c["patron"], c["metodo"],
                              c["razon"], cfg["semilla"], cfg["tv"]["lam"],
                              cfg["tv"]["iteraciones"], cfg["tv"]["iter_prox"], ruido, device)
            origen = "generada"
        previas = pd.read_csv(destino / "metricas_2d.csv").set_index("vista")
        tv = c["metodo"] == "tv"
        comunes = {"experimento": cfg["experimento"], "escena": cfg["escena"],
                   "condicion": c["nombre"], "tipo": c["tipo"], "patron": c["patron"],
                   "metodo": c["metodo"], "razon": c["razon"], "rejilla": texto_rejilla(cfg),
                   "mediciones": numero_mediciones(rejilla, c["razon"]) if c["tipo"] == "spc" else 0,
                   "semilla": cfg["semilla"], "lam_tv": cfg["tv"]["lam"] if tv else None,
                   "iter_tv": cfg["tv"]["iteraciones"] if tv else None,
                   "iter_prox": cfg["tv"]["iter_prox"] if tv else None,
                   "ruido": "si" if (ruido and c["tipo"] == "spc") else "no", "origen": origen}
        for conjunto in ("entrenamiento", "prueba"):
            for n in vistas[conjunto]:
                z = cargar_imagen(destino / "images" / n)
                p_rej, s_rej = psnr_ssim(remuestrear(z, rejilla), en_rejilla[n])
                filas.append({**comunes, "conjunto": conjunto, "vista": n,
                              "PSNR": previas.loc[n, "psnr"], "SSIM": previas.loc[n, "ssim"],
                              "LPIPS": previas.loc[n, "lpips"],
                              "PSNR_rejilla": p_rej, "SSIM_rejilla": s_rej})
        # Subconjunto fijo: imagen que ve 3DGS y reconstrucción pura en la rejilla.
        (recon / c["nombre"]).mkdir(exist_ok=True)
        op = (OperadorSPC(c["patron"], rejilla, c["razon"], cfg["semilla"], device)
              if c["tipo"] == "spc" else None)
        for n in vistas["guardadas"]:
            shutil.copy2(destino / "images" / n, recon / c["nombre"] / n)
            if op is None:
                continue
            g = en_rejilla[n].to(device)
            kw = opciones_tv if c["metodo"] == "tv" else {}
            pura = reconstruir_imagen(medir_imagen(g, op, ruido), op, c["metodo"], **kw).clamp(0, 1)
            guardar_png(pura, recon / c["nombre"] / f"{n.rsplit('.', 1)[0]}_rejilla.png")
            p, s = psnr_ssim(pura, g)
            guardadas.append({"condicion": c["nombre"], "vista": n, "PSNR_puro_rejilla": p,
                              "SSIM_puro_rejilla": s})
        media = np.mean([f["PSNR"] for f in filas if f["condicion"] == c["nombre"]])
        print(f"[reconstruir] {c['nombre']:32s} {origen:11s} PSNR 2D {media:6.2f} dB "
              f"({time.time() - t0:.1f} s)", flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(carpeta / "metrics_2d_por_vista.csv", index=False, float_format="%.6f")
    resumir_2d(df).to_csv(carpeta / "metrics_2d.csv", index=False, float_format="%.6f")
    pd.DataFrame(guardadas, columns=["condicion", "vista", "PSNR_puro_rejilla",
                                     "SSIM_puro_rejilla"]).to_csv(
        carpeta / "metrics_2d_guardadas.csv", index=False, float_format="%.6f")
    print(f"[reconstruir] {len(df)} filas -> {carpeta / 'metrics_2d_por_vista.csv'}")


def resumir_2d(df):
    """Promedios por condición en tres conjuntos: todas las vistas, entrenamiento y prueba."""
    claves = [c for c in df.columns if c not in
              ["conjunto", "vista", "PSNR", "SSIM", "LPIPS", "PSNR_rejilla", "SSIM_rejilla"]]
    valores = ["PSNR", "SSIM", "LPIPS", "PSNR_rejilla", "SSIM_rejilla"]
    partes = []
    for conjunto, d in [("todas", df), ("entrenamiento", df[df.conjunto == "entrenamiento"]),
                        ("prueba", df[df.conjunto == "prueba"])]:
        g = d.groupby(claves, dropna=False, sort=False)
        r = g[valores].mean().join(g.size().rename("n_vistas")).reset_index()
        r.insert(len(claves), "conjunto", conjunto)
        partes.append(r)
    orden = {n: i for i, n in enumerate(df.condicion.unique())}
    salida = pd.concat(partes, ignore_index=True)
    salida["_o"] = salida.condicion.map(orden)
    salida["_c"] = salida.conjunto.map({"todas": 0, "entrenamiento": 1, "prueba": 2})
    salida = salida.sort_values(["_o", "_c"]).drop(columns=["_o", "_c"])
    return salida[claves + ["conjunto", "n_vistas"] + valores]


# --- entrenar y evaluar (3DGS) -------------------------------------------------------------------
def corridas_3d(cfg):
    """(condición, iteraciones, semilla, carpeta del modelo) de cada entrenamiento 3DGS.
    La carpeta sigue el esquema de scripts/train_gsplat.sh: <modelos>/<cond>/it<N>_s<S>_f<F>."""
    g = cfg["3dgs"]
    for c in condiciones_3d(cfg):
        for it in g["iteraciones"]:
            for s in g["semillas"]:
                yield c, it, s, ruta(cfg, "modelos") / c["nombre"] / f"it{it}_s{s}_f{g['factor']}"


def comprobar_modelo(corrida, c, it, s, cfg, vistas):
    """True si el modelo existe y corresponde a la configuración; None si no existe."""
    if not (corrida / "ckpt.pt").exists():
        if corrida.exists() and any(p.name != "train.log" for p in corrida.iterdir()):
            sys.exit(f"{corrida} existe sin ckpt.pt (incompleta). Revísela a mano: no se toca.")
        return None
    info = json.loads((corrida / "info.json").read_text())
    esperado = {"condicion": c["nombre"], "iteraciones": it, "semilla": s,
                "factor": cfg["3dgs"]["factor"], "grado_sh": cfg["3dgs"]["grado_sh"],
                "entrenamiento": vistas["entrenamiento"], "prueba": vistas["prueba"]}
    difs = {k: (info.get(k), v) for k, v in esperado.items() if info.get(k) != v}
    if difs:
        sys.exit(f"El modelo {corrida} no corresponde a la configuración (existente, pedido): {difs}")
    return True


class _Duplicar:
    """Escribe a la vez en la consola y en un archivo (equivale a `| tee train.log`)."""

    def __init__(self, *flujos):
        self.flujos = flujos

    def write(self, texto):
        for f in self.flujos:
            f.write(texto)

    def flush(self):
        for f in self.flujos:
            f.flush()


def cmd_entrenar(cfg, args, carpeta):
    from .train_3dgs import entrenar  # requiere gsplat con CUDA (WSL)
    vistas = leer_vistas(cfg, carpeta)
    for c, it, s, corrida in corridas_3d(cfg):
        datos = ruta(cfg, "condiciones") / c["nombre"]
        if not comprobar_existente(datos, parametros_esperados(cfg, c, vistas)):
            sys.exit(f"Falta la condición {datos}: ejecute antes 'dataset' y 'reconstruir'.")
        if comprobar_modelo(corrida, c, it, s, cfg, vistas):
            print(f"[entrenar] {c['nombre']} it{it} s{s}: se reutiliza {corrida}")
            continue
        corrida.mkdir(parents=True, exist_ok=True)
        print(f"[entrenar] {c['nombre']} it{it} s{s} -> {corrida}", flush=True)
        g = cfg["3dgs"]
        with open(corrida / "train.log", "w") as log:
            salida, sys.stdout = sys.stdout, _Duplicar(sys.stdout, log)
            try:
                entrenar(argparse.Namespace(datos=str(datos), salida=str(corrida), iteraciones=it,
                                            semilla=s, factor=g["factor"], grado_sh=g["grado_sh"]))
            finally:
                sys.stdout = salida


def cmd_evaluar(cfg, args, carpeta):
    from .evaluacion_3d import cargar_corrida, evaluar_vistas, vistas_no_usadas
    resumen_csv, vistas_csv = carpeta / "metrics_3d.csv", carpeta / "metrics_3d_por_vista.csv"
    renders = carpeta / "render_3d"
    proteger(resumen_csv, vistas_csv, renders, forzar=args.forzar)
    vistas = leer_vistas(cfg, carpeta)
    lista = list(corridas_3d(cfg))
    faltan = [str(r) for c, it, s, r in lista if not comprobar_modelo(r, c, it, s, cfg, vistas)]
    if faltan:  # evaluar nunca entrena
        sys.exit("Faltan modelos (ejecute 'entrenar'):\n  " + "\n  ".join(faltan))
    imagenes = ruta(cfg, "imagenes")
    resumen, por_vista = [], []
    for k, (c, it, s, corrida) in enumerate(lista, 1):
        t0 = time.time()
        info, splats = cargar_corrida(corrida)
        conjuntos = [("prueba", info["prueba"]), ("entrenamiento", info["entrenamiento"])]
        if cfg["evaluacion"]["extendida"]:
            conjuntos.append(("extendida", vistas_no_usadas(info, imagenes)))
        comunes = {"experimento": cfg["experimento"], "escena": cfg["escena"],
                   "condicion": c["nombre"], "tipo": c["tipo"], "patron": c["patron"],
                   "metodo": c["metodo"], "razon": c["razon"],
                   "rejilla": "-" if c["tipo"] == "original" else texto_rejilla(cfg),
                   "iteraciones": it, "semilla": s, "factor": info["factor"]}
        for conjunto, nombres in conjuntos:
            media, filas = evaluar_vistas(splats, info, nombres, carpeta_imagenes=imagenes)
            por_vista += [{**comunes, "conjunto": conjunto, "vista": f["vista"], "PSNR": f["psnr"],
                           "SSIM": f["ssim"], "LPIPS": f["lpips"]} for f in filas]
            resumen.append({**comunes, "conjunto": conjunto, "n_vistas": len(filas),
                            "PSNR": media["psnr"], "SSIM": media["ssim"], "LPIPS": media["lpips"],
                            "gaussianas": info["gaussianas"], "tiempo_entrenamiento_s": info["tiempo_s"],
                            "mem_max_GiB": info["mem_max_reservada_GiB"],
                            "corrida": str(corrida.relative_to(RAIZ) if corrida.is_relative_to(RAIZ)
                                           else corrida)})
        # Renders del subconjunto fijo de vistas.
        destino = renders / c["nombre"] / f"it{it}_s{s}"
        destino.mkdir(parents=True, exist_ok=True)
        evaluar_vistas(splats, info, vistas["guardadas"], carpeta_renders=destino,
                       carpeta_imagenes=imagenes)
        del splats
        torch.cuda.empty_cache()
        # Se reescriben los CSV tras cada modelo: si algo falla, lo evaluado queda guardado.
        pd.DataFrame(resumen).to_csv(resumen_csv, index=False, float_format="%.6f")
        pd.DataFrame(por_vista).to_csv(vistas_csv, index=False, float_format="%.6f")
        p = {r["conjunto"]: r["PSNR"] for r in resumen[-len(conjuntos):]}
        print(f"[evaluar] {k}/{len(lista)} {c['nombre']} it{it} s{s}: "
              + " | ".join(f"{n} {v:.2f} dB" for n, v in p.items())
              + f" ({time.time() - t0:.0f} s)", flush=True)


# --- Programa principal -------------------------------------------------------------------------
SUBCOMANDOS = {"dataset": cmd_dataset, "reconstruir": cmd_reconstruir, "entrenar": cmd_entrenar,
               "evaluar": cmd_evaluar}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subcomando", choices=list(SUBCOMANDOS))
    p.add_argument("--config", required=True, help="archivo configs/<experimento>.yaml")
    p.add_argument("--forzar", action="store_true",
                   help="permite sobrescribir las salidas de results/<experimento>/ (nunca data/)")
    args = p.parse_args(argv)
    cfg = cargar_config(args.config)
    carpeta = preparar_experimento(cfg)
    t0 = time.time()
    SUBCOMANDOS[args.subcomando](cfg, args, carpeta)
    registrar(carpeta, args.subcomando, t0)


if __name__ == "__main__":
    main()
