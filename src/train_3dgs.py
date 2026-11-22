"""Entrenamiento mínimo de 3D Gaussian Splatting con gsplat.

Es una versión compacta del entrenador de ejemplo de gsplat (simple_trainer),
con los mismos hiperparámetros por defecto, para que sea fácil de leer:

1. Lee las cámaras COLMAP (sparse/0) y las imágenes de entrenamiento de la
   condición (vistas.json). Las intrínsecas se escalan al tamaño real de las
   imágenes (en Truck, las imágenes son la mitad del tamaño de la cámara).
2. Inicializa una gaussiana por punto 3D de COLMAP: posición = punto,
   color = color del punto (armónico esférico de grado 0), escala = distancia
   media a los 3 vecinos más cercanos, opacidad = 0.1.
3. En cada iteración renderiza una vista de entrenamiento con
   gsplat.rasterization y minimiza 0.8*L1 + 0.2*(1 - SSIM) con Adam.
4. gsplat.strategy.DefaultStrategy densifica (clona/divide) y poda gaussianas.
5. Guarda los parámetros en <salida>/ckpt.pt y un resumen en <salida>/info.json.

Uso:
    python -m src.train_3dgs --datos data/conditions/truck/original \
        --salida outputs/truck/original/it3000_s449 --iteraciones 3000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

from .colmap_io import leer_modelo
from .patterns import SEMILLA

C0 = 0.28209479177387814  # constante del armónico esférico de grado 0


def fijar_semilla(semilla):
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


# --- Datos ---------------------------------------------------------------------
def cargar_vistas(carpeta, nombres, factor=1, device="cuda"):
    """Carga imágenes y cámaras de `nombres`.

    Devuelve una lista de dict(nombre, imagen (H, W, 3), viewmat (4, 4), K (3, 3)).
    `factor` > 1 reduce la resolución (las intrínsecas se ajustan igual).
    """
    carpeta = Path(carpeta)
    camaras, imagenes, _, _ = leer_modelo(carpeta / "sparse" / "0")
    vistas = []
    for n in nombres:
        img = Image.open(carpeta / "images" / n).convert("RGB")
        if factor > 1:
            img = img.resize((round(img.width / factor), round(img.height / factor)),
                             Image.BICUBIC)
        a = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        cam = camaras[imagenes[n]["camara"]]
        K = cam["K"].copy()
        K[0] *= img.width / cam["ancho"]   # escala de intrínsecas al tamaño real
        K[1] *= img.height / cam["alto"]
        vistas.append({"nombre": n, "imagen": a.to(device),
                       "viewmat": torch.tensor(imagenes[n]["w2c"], dtype=torch.float32,
                                               device=device),
                       "K": torch.tensor(K, dtype=torch.float32, device=device)})
    return vistas


def escala_escena(vistas):
    """Radio de la escena: 1.1 x la distancia máxima de una cámara al centro
    de todas las cámaras (como en 3DGS). Escala tasas y umbrales."""
    centros = torch.stack([torch.linalg.inv(v["viewmat"])[:3, 3] for v in vistas])
    return float((centros - centros.mean(0)).norm(dim=1).max() * 1.1)


# --- Gaussianas ------------------------------------------------------------------
def inicializar_gaussianas(xyz, rgb, grado_sh=3, device="cuda"):
    """Una gaussiana por punto 3D de COLMAP (parámetros en espacio no acotado)."""
    n = xyz.shape[0]
    dist, _ = cKDTree(xyz).query(xyz, k=4)  # el primer vecino es el propio punto
    dist_media = np.sqrt((dist[:, 1:] ** 2).mean(axis=1)).clip(1e-7)
    escalas = np.log(np.repeat(dist_media[:, None], 3, axis=1))
    sh0 = (rgb.astype(np.float32) / 255.0 - 0.5) / C0
    k_sh = (grado_sh + 1) ** 2
    params = {
        "means": torch.tensor(xyz, dtype=torch.float32),
        "scales": torch.tensor(escalas, dtype=torch.float32),          # log-escala
        "quats": F.normalize(torch.rand(n, 4), dim=-1),                  # rotación
        "opacities": torch.logit(torch.full((n,), 0.1)),                # logit
        "sh0": torch.tensor(sh0, dtype=torch.float32)[:, None, :],       # (n, 1, 3)
        "shN": torch.zeros(n, k_sh - 1, 3),                              # (n, 15, 3)
    }
    return torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.to(device)) for k, v in params.items()})


def crear_optimizadores(splats, escala):
    """Un Adam por grupo de parámetros, con las tasas por defecto de gsplat."""
    tasas = {"means": 1.6e-4 * escala, "scales": 5e-3, "quats": 1e-3,
             "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    return {k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
            for k, lr in tasas.items()}


def renderizar(splats, viewmat, K, ancho, alto, grado_sh):
    """Renderiza una vista. Devuelve (imagen (H, W, 3), info de gsplat)."""
    from gsplat import rasterization
    colores = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    render, _alpha, info = rasterization(
        means=splats["means"], quats=splats["quats"],
        scales=torch.exp(splats["scales"]), opacities=torch.sigmoid(splats["opacities"]),
        colors=colores, viewmats=viewmat[None], Ks=K[None], width=ancho, height=alto,
        sh_degree=grado_sh, packed=False)
    return render[0], info


# --- Pérdida ------------------------------------------------------------------------
def ssim_torch(a, b, ventana=11, sigma=1.5):
    """SSIM diferenciable con ventana gaussiana; a, b: (1, 3, H, W)."""
    x = torch.arange(ventana, device=a.device, dtype=a.dtype) - ventana // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    # La ventana gaussiana 2D es separable: se filtra por filas y luego por
    # columnas (mucho más rápido que una convolución 11x11). Los cinco mapas
    # (a, b, a², b², ab) se filtran en una sola llamada con 15 canales.
    z = torch.cat([a, b, a * a, b * b, a * b], dim=1)
    gh = g.view(1, 1, 1, ventana).expand(15, 1, 1, ventana)
    gv = g.view(1, 1, ventana, 1).expand(15, 1, ventana, 1)
    z = F.conv2d(z, gh, padding=(0, ventana // 2), groups=15)
    z = F.conv2d(z, gv, padding=(ventana // 2, 0), groups=15)
    mu_a, mu_b, e_aa, e_bb, e_ab = z.split(3, dim=1)
    var_a = e_aa - mu_a ** 2
    var_b = e_bb - mu_b ** 2
    cov = e_ab - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mapa = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / \
           ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return mapa.mean()


# --- Entrenamiento -------------------------------------------------------------------
def entrenar(args):
    from gsplat.strategy import DefaultStrategy

    fijar_semilla(args.semilla)
    device = "cuda"
    datos = Path(args.datos)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((datos / "vistas.json").read_text())

    vistas = cargar_vistas(datos, cfg["entrenamiento"], args.factor, device)
    escala = escala_escena(vistas)
    _, _, xyz, rgb = leer_modelo(datos / "sparse" / "0")
    splats = inicializar_gaussianas(xyz, rgb, args.grado_sh, device)
    optimizadores = crear_optimizadores(splats, escala)
    # La tasa de las posiciones decae exponencialmente hasta 1 % al final.
    planificador = torch.optim.lr_scheduler.ExponentialLR(
        optimizadores["means"], gamma=0.01 ** (1.0 / args.iteraciones))

    estrategia = DefaultStrategy(verbose=False)  # parámetros por defecto de gsplat
    estrategia.check_sanity(splats, optimizadores)
    estado = estrategia.initialize_state(scene_scale=escala)

    print(f"[train] {len(vistas)} vistas de {vistas[0]['imagen'].shape[1]}x"
          f"{vistas[0]['imagen'].shape[0]} | {len(xyz)} gaussianas iniciales | "
          f"escala de escena {escala:.3f}")
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator().manual_seed(args.semilla)
    orden = []
    t0 = time.time()
    for paso in range(args.iteraciones):
        if not orden:  # recorre las vistas en orden aleatorio (época)
            orden = torch.randperm(len(vistas), generator=gen).tolist()
        v = vistas[orden.pop()]
        alto, ancho = v["imagen"].shape[:2]
        grado = min(paso // 1000, args.grado_sh)  # el grado SH sube cada 1000 pasos

        render, info = renderizar(splats, v["viewmat"], v["K"], ancho, alto, grado)
        estrategia.step_pre_backward(splats, optimizadores, estado, paso, info)
        l1 = (render - v["imagen"]).abs().mean()
        ssim = ssim_torch(render.permute(2, 0, 1)[None], v["imagen"].permute(2, 0, 1)[None])
        perdida = 0.8 * l1 + 0.2 * (1 - ssim)
        perdida.backward()

        for opt in optimizadores.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        planificador.step()
        estrategia.step_post_backward(splats, optimizadores, estado, paso, info, packed=False)

        if paso % 500 == 0 or paso == args.iteraciones - 1:
            print(f"  paso {paso:6d} | pérdida {perdida.item():.4f} | "
                  f"gaussianas {len(splats['means'])} | "
                  f"mem. máx. {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    tiempo = time.time() - t0

    torch.save({k: v.detach().cpu() for k, v in splats.items()}, salida / "ckpt.pt")
    info = {**{k: cfg[k] for k in ("escena", "condicion", "patron", "metodo", "razon",
                                   "rejilla", "prueba", "entrenamiento")},
            "datos": str(datos), "iteraciones": args.iteraciones, "semilla": args.semilla,
            "factor": args.factor, "grado_sh": args.grado_sh,
            "gaussianas": len(splats["means"]), "tiempo_s": round(tiempo, 1),
            "mem_max_asignada_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "mem_max_reservada_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
    (salida / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"[train] listo en {tiempo:.1f} s | {info['gaussianas']} gaussianas | "
          f"mem. máx. asignada {info['mem_max_asignada_GiB']} GiB, "
          f"reservada {info['mem_max_reservada_GiB']} GiB")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", required=True, help="carpeta de la condición")
    p.add_argument("--salida", required=True)
    p.add_argument("--iteraciones", type=int, default=3000)
    p.add_argument("--semilla", type=int, default=SEMILLA)
    p.add_argument("--factor", type=int, default=1, help="reducción de resolución")
    p.add_argument("--grado-sh", type=int, default=3)
    entrenar(p.parse_args())


if __name__ == "__main__":
    main()
