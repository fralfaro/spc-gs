"""Diagnóstico visual del error en las vistas de prueba de un modelo 3DGS.

Para cada vista de prueba muestra, en una fila:
  original | render | error absoluto | n.º de vistas de entrenamiento que ven el punto | opacidad acumulada

La columna "n.º de vistas" se calcula así: con la profundidad renderizada se
reconstruye el punto 3D de cada píxel y se proyecta en las 21 cámaras de
entrenamiento; se cuentan las cámaras en cuyo campo visual cae (delante de la
cámara y dentro de la imagen). No considera oclusiones, así que es una cota
superior de cuántas veces se observó ese punto.

Además imprime el error medio según ese número de vistas y la opacidad, y
guarda la figura y la tabla en results/figuras/ y results/ (si ya existen, se detiene salvo
--forzar).

Uso:
    python scripts/diagnostico_prueba.py --corrida outputs/truck/original/it7000_s449_f1
    python scripts/diagnostico_prueba.py --corrida ... --comparar original control_256x512 \
        hadamard_tv_05_256x512 hadamard_tv_50_256x512
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.salidas import proteger  # noqa: E402
from src.train_3dgs import cargar_vistas  # noqa: E402

TRAMOS = [(0, 2, "0–2"), (3, 5, "3–5"), (6, 10, "6–10"), (11, 99, "11 o más")]


def renderizar_con_profundidad(splats, v, grado_sh):
    """Devuelve color (H, W, 3), profundidad esperada (H, W) y opacidad (H, W)."""
    from gsplat import rasterization
    alto, ancho = v["imagen"].shape[:2]
    colores = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    with torch.no_grad():
        out, alpha, _ = rasterization(
            means=splats["means"], quats=splats["quats"],
            scales=torch.exp(splats["scales"]), opacities=torch.sigmoid(splats["opacities"]),
            colors=colores, viewmats=v["viewmat"][None], Ks=v["K"][None], width=ancho,
            height=alto, sh_degree=grado_sh, packed=False, render_mode="RGB+ED")
    return out[0, ..., :3].clamp(0, 1), out[0, ..., 3], alpha[0, ..., 0]


def vistas_que_observan(prof, v, entrenamiento):
    """Para cada píxel de la vista v, cuántas cámaras de entrenamiento ven su punto 3D."""
    alto, ancho = prof.shape
    vv, uu = torch.meshgrid(torch.arange(alto, device=prof.device),
                            torch.arange(ancho, device=prof.device), indexing="ij")
    pix = torch.stack([uu + 0.5, vv + 0.5, torch.ones_like(uu)], -1).float().reshape(-1, 3)
    rayos = pix @ torch.linalg.inv(v["K"]).T                  # coordenadas de cámara con z = 1
    p_cam = rayos * prof.reshape(-1, 1)
    c2w = torch.linalg.inv(v["viewmat"])
    p_mundo = p_cam @ c2w[:3, :3].T + c2w[:3, 3]
    cuenta = torch.zeros(alto * ancho, device=prof.device)
    for e in entrenamiento:
        q = p_mundo @ e["viewmat"][:3, :3].T + e["viewmat"][:3, 3]
        z = q[:, 2]
        uv = (q @ e["K"].T)[:, :2] / z.clamp(min=1e-6)[:, None]
        h, w = e["imagen"].shape[:2]
        dentro = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        cuenta += dentro.float()
    return cuenta.reshape(alto, ancho)


def cargar_modelo(corrida):
    info = json.loads((corrida / "info.json").read_text())
    splats = {k: t.cuda() for k, t in torch.load(corrida / "ckpt.pt").items()}
    return info, splats


def diagnosticar(corrida, forzar=False):
    info, splats = cargar_modelo(corrida)
    nombre = f"diagnostico_prueba_{info['condicion']}_it{info['iteraciones']}"
    proteger(RAIZ / "results" / "figuras" / f"{nombre}.png", RAIZ / "results" / f"{nombre}.csv",
             forzar=forzar)
    originales = RAIZ / "data" / info["escena"]
    prueba = cargar_vistas(originales, info["prueba"], info["factor"], "cuda")
    entrenamiento = cargar_vistas(originales, info["entrenamiento"], info["factor"], "cuda")

    fig, ejes = plt.subplots(len(prueba), 5, figsize=(22, 2.9 * len(prueba)))
    filas = []
    for i, v in enumerate(prueba):
        color, prof, alpha = renderizar_con_profundidad(splats, v, info["grado_sh"])
        error = (color - v["imagen"]).abs().mean(-1)          # error absoluto medio por píxel
        cuenta = vistas_que_observan(prof, v, entrenamiento)
        opaco = alpha > 0.5
        for lo, hi, etiqueta in TRAMOS:
            m = opaco & (cuenta >= lo) & (cuenta <= hi)
            filas.append({"vista": v["nombre"], "region": f"{etiqueta} vistas",
                          "fraccion_pixeles": float(m.float().mean()),
                          "error_abs_medio": float(error[m].mean()) if m.any() else np.nan})
        m = ~opaco
        filas.append({"vista": v["nombre"], "region": "opacidad < 0.5 (sin gaussianas)",
                      "fraccion_pixeles": float(m.float().mean()),
                      "error_abs_medio": float(error[m].mean()) if m.any() else np.nan})
        filas.append({"vista": v["nombre"], "region": "toda la imagen", "fraccion_pixeles": 1.0,
                      "error_abs_medio": float(error.mean())})

        paneles = [(v["imagen"], "original", {}), (color, "render 3DGS", {}),
                   (error, "error absoluto", dict(cmap="magma", vmin=0, vmax=0.5)),
                   (cuenta, "n.º de vistas de entrenamiento que ven el punto",
                    dict(cmap="viridis", vmin=0, vmax=len(entrenamiento))),
                   (alpha, "opacidad acumulada", dict(cmap="gray", vmin=0, vmax=1))]
        for eje, (img, titulo, kw) in zip(ejes[i], paneles):
            im = eje.imshow(img.cpu().numpy(), **kw)
            eje.set_xticks([]); eje.set_yticks([])
            if i == 0:
                eje.set_title(titulo, fontsize=10)
            if kw:
                fig.colorbar(im, ax=eje, fraction=0.025, pad=0.01)
        ejes[i, 0].set_ylabel(v["nombre"], fontsize=10)
    fig.suptitle(f"{info['condicion']}, {info['iteraciones']} iteraciones: vistas de prueba")
    plt.tight_layout()
    (RAIZ / "results" / "figuras").mkdir(parents=True, exist_ok=True)
    fig.savefig(RAIZ / "results" / "figuras" / f"{nombre}.png", dpi=90)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RAIZ / "results" / f"{nombre}.csv", index=False)
    pd.set_option("display.width", 200)
    print(tabla.round(4).to_string(index=False))


def comparar(corrida, condiciones, forzar=False):
    """Una fila por vista de prueba: original y el render de cada condición."""
    info0 = json.loads((corrida / "info.json").read_text())
    carpeta = corrida.name                                   # p. ej. it7000_s449_f1
    proteger(RAIZ / "results" / "figuras" / f"comparacion_prueba_{carpeta}.png", forzar=forzar)
    originales = RAIZ / "data" / info0["escena"]
    prueba = cargar_vistas(originales, info0["prueba"], info0["factor"], "cuda")
    fig, ejes = plt.subplots(len(prueba), len(condiciones) + 1,
                             figsize=(4.4 * (len(condiciones) + 1), 2.6 * len(prueba)))
    for j, cond in enumerate(["imagen original"] + list(condiciones)):
        if j > 0:
            info, splats = cargar_modelo(RAIZ / "outputs" / info0["escena"] / cond / carpeta)
        for i, v in enumerate(prueba):
            img = v["imagen"] if j == 0 else renderizar_con_profundidad(splats, v,
                                                                       info["grado_sh"])[0]
            ejes[i, j].imshow(img.cpu().numpy())
            ejes[i, j].set_xticks([]); ejes[i, j].set_yticks([])
            if i == 0:
                ejes[i, j].set_title(cond, fontsize=10)
            if j == 0:
                ejes[i, j].set_ylabel(v["nombre"], fontsize=10)
    fig.suptitle(f"Vistas de prueba, {carpeta}")
    plt.tight_layout()
    fig.savefig(RAIZ / "results" / "figuras" / f"comparacion_prueba_{carpeta}.png", dpi=90)
    print("figura de comparación guardada")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrida", default="outputs/truck/original/it7000_s449_f1")
    p.add_argument("--comparar", nargs="*", default=None,
                   help="condiciones a mostrar lado a lado (mismas iteraciones y semilla)")
    p.add_argument("--forzar", action="store_true", help="sobrescribe la figura y la tabla si existen")
    args = p.parse_args()
    corrida = (RAIZ / args.corrida) if not Path(args.corrida).is_absolute() else Path(args.corrida)
    diagnosticar(corrida, args.forzar)
    if args.comparar:
        comparar(corrida, args.comparar, args.forzar)


if __name__ == "__main__":
    main()
