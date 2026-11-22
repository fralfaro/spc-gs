"""Evaluación de un modelo 3DGS entrenado.

Renderiza las vistas de PRUEBA desde las cámaras COLMAP y las compara contra
las imágenes ORIGINALES de data/<escena>/images (nunca contra las imágenes
degradadas con que se entrenó). Estas son las métricas principales
(columnas PSNR, SSIM, LPIPS).

Como complemento, renderiza también las vistas de ENTRENAMIENTO y las compara
con sus imágenes ORIGINALES (columnas PSNR_ent, SSIM_ent, LPIPS_ent). Con
pocas vistas, el error en las vistas de prueba está dominado por artefactos de
generalización (zonas poco observadas); en las vistas de entrenamiento, en
cambio, se ve cuánto de la degradación del SPC hereda el modelo 3D.

Con --extendida evalúa además TODAS las imágenes de la escena que no se usaron
para entrenar (230 en Truck; todas tienen pose COLMAP), también contra las
ORIGINALES (columnas PSNR_ext, SSIM_ext, LPIPS_ext). Con solo 3 vistas de
prueba las diferencias pequeñas no son concluyentes; con 230 sí se puede
estimar la variabilidad entre vistas.

Agrega una fila (promedios) a results/metrics_3d.csv, una fila por vista a
results/metrics_3d_por_vista.csv y guarda los renders de prueba en
<corrida>/renders/.

El núcleo (renderizar y comparar) está en src/evaluacion_3d.py, compartido con
`python -m src.cli evaluar`, que escribe en results/<experimento>/.

Uso:
    python scripts/eval_3d.py --corrida outputs/truck/original/it3000_s449_f1
    python scripts/eval_3d.py --todas truck                # reevalúa todas las corridas
    python scripts/eval_3d.py --todas truck --extendida    # + 230 vistas no usadas
"""

import argparse
import json
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.evaluacion_3d import evaluar_vistas, vistas_no_usadas  # noqa: E402
from src.make_dataset import agregar_fila_csv, agregar_filas_csv  # noqa: E402


def evaluar(corrida, csv_salida, extendida=False):
    corrida = Path(corrida)
    info = json.loads((corrida / "info.json").read_text())
    splats = {k: v.cuda() for k, v in torch.load(corrida / "ckpt.pt").items()}

    (corrida / "renders").mkdir(exist_ok=True)
    prueba, por_vista_p = evaluar_vistas(splats, info, info["prueba"], corrida / "renders",
                                         mostrar=True)
    entren, por_vista_e = evaluar_vistas(splats, info, info["entrenamiento"])
    conjuntos = [("prueba", por_vista_p), ("entrenamiento", por_vista_e)]
    if extendida:
        ext, por_vista_x = evaluar_vistas(splats, info, vistas_no_usadas(info))
        conjuntos.append(("extendida", por_vista_x))

    fila = {"escena": info["escena"], "patron": info["patron"], "metodo": info["metodo"],
            "razon": info["razon"], "rejilla": info["rejilla"],
            "iteraciones": info["iteraciones"], "semilla": info["semilla"],
            "vistas_entrenamiento": len(info["entrenamiento"]),
            "vistas_prueba": len(info["prueba"]), "factor": info["factor"],
            "PSNR": round(prueba["psnr"], 4), "SSIM": round(prueba["ssim"], 4),
            "LPIPS": round(prueba["lpips"], 4), "gaussianas": info["gaussianas"],
            "tiempo_s": info["tiempo_s"], "mem_max_GiB": info["mem_max_reservada_GiB"],
            "condicion": info["condicion"],
            "PSNR_ent": round(entren["psnr"], 4), "SSIM_ent": round(entren["ssim"], 4),
            "LPIPS_ent": round(entren["lpips"], 4)}
    if extendida:
        fila.update({"PSNR_ext": round(ext["psnr"], 4), "SSIM_ext": round(ext["ssim"], 4),
                     "LPIPS_ext": round(ext["lpips"], 4), "vistas_ext": len(por_vista_x)})
    agregar_fila_csv(csv_salida, fila)

    # Métricas por vista, en un CSV aparte (mismo directorio que csv_salida).
    csv_vistas = Path(csv_salida).with_name(Path(csv_salida).stem + "_por_vista.csv")
    agregar_filas_csv(csv_vistas, [
        {"escena": info["escena"], "condicion": info["condicion"],
         "patron": info["patron"], "metodo": info["metodo"], "razon": info["razon"],
         "iteraciones": info["iteraciones"], "semilla": info["semilla"],
         "factor": info["factor"], "conjunto": conjunto, "vista": f["vista"],
         "PSNR": round(f["psnr"], 4), "SSIM": round(f["ssim"], 4),
         "LPIPS": round(f["lpips"], 4)}
        for conjunto, filas in conjuntos for f in filas])

    (corrida / "metricas.json").write_text(json.dumps(fila, indent=2, ensure_ascii=False))
    extra = (f" || {len(por_vista_x)} vistas no usadas PSNR {ext['psnr']:.2f} dB"
             if extendida else "")
    print(f"[eval] {info['condicion']} it{info['iteraciones']} s{info['semilla']}: prueba PSNR "
          f"{prueba['psnr']:.2f} dB | SSIM {prueba['ssim']:.4f} | LPIPS {prueba['lpips']:.4f} || "
          f"entrenamiento PSNR {entren['psnr']:.2f} dB | SSIM {entren['ssim']:.4f} | LPIPS "
          f"{entren['lpips']:.4f}{extra} -> {csv_salida}")
    return fila


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = p.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--corrida", help="carpeta con ckpt.pt e info.json")
    grupo.add_argument("--todas", metavar="ESCENA",
                       help="reevalúa todas las corridas de outputs/<ESCENA> (orden alfabético)")
    p.add_argument("--csv", default=str(RAIZ / "results" / "metrics_3d.csv"))
    p.add_argument("--extendida", action="store_true",
                   help="evalúa también todas las vistas no usadas para entrenar")
    args = p.parse_args()
    if args.corrida:
        evaluar(args.corrida, args.csv, args.extendida)
    else:
        for ckpt in sorted((RAIZ / "outputs" / args.todas).glob("*/it*/ckpt.pt")):
            evaluar(ckpt.parent, args.csv, args.extendida)


if __name__ == "__main__":
    main()
