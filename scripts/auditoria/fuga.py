"""Auditoría: ¿qué información de las imágenes originales entra en el entrenamiento?

1. Lee las pistas (tracks) de points3D.bin: qué imágenes observaron cada punto
   3D de COLMAP. La nube de puntos inicializa las gaussianas (posición y color)
   en TODAS las condiciones.
2. Cuenta cuántos puntos fueron observados por las vistas de prueba, cuántos
   solo por imágenes que no son de entrenamiento, etc.
3. Mide cuán parecida es la vista 000130 (usada para elegir λ de la TV) a la
   vista de prueba 000131.

Solo lee archivos; escribe results/auditoria/fuga.csv (si ya existe, se detiene salvo --forzar).
Uso: python scripts/auditoria/fuga.py [--forzar]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.make_dataset import cargar_imagen, seleccionar_vistas  # noqa: E402
from src.metrics import calcular_metricas  # noqa: E402
from src.salidas import proteger  # noqa: E402


def leer(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def ids_de_imagenes(ruta):
    """{image_id: nombre} desde images.bin."""
    ids = {}
    with open(ruta, "rb") as f:
        (n,) = leer(f, "Q")
        for _ in range(n):
            id_img, *_resto, _cam = leer(f, "idddddddi")
            nombre = b""
            while (c := f.read(1)) != b"\x00":
                nombre += c
            (n2d,) = leer(f, "Q")
            f.seek(24 * n2d, 1)
            ids[id_img] = nombre.decode()
    return ids


def pistas(ruta):
    """Lista de conjuntos de image_id que observaron cada punto 3D."""
    salida = []
    with open(ruta, "rb") as f:
        (n,) = leer(f, "Q")
        for _ in range(n):
            leer(f, "QdddBBBd")
            (largo,) = leer(f, "Q")
            datos = np.frombuffer(f.read(8 * largo), dtype=np.int32).reshape(-1, 2)
            salida.append(set(datos[:, 0].tolist()))
    return salida


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forzar", action="store_true", help="sobrescribe la tabla si existe")
    salida = RAIZ / "results" / "auditoria" / "fuga.csv"
    proteger(salida, forzar=p.parse_args().forzar)
    sparse = RAIZ / "data" / "truck" / "sparse" / "0"
    ids = ids_de_imagenes(sparse / "images.bin")
    nombres = sorted(ids.values())
    ent, prueba = seleccionar_vistas(nombres, 24, 3)
    id_de = {v: k for k, v in ids.items()}
    s_ent = {id_de[n] for n in ent}
    s_pru = {id_de[n] for n in prueba}

    p = pistas(sparse / "points3D.bin")
    n = len(p)
    largo = np.array([len(t) for t in p])
    filas = [
        ("imágenes registradas en COLMAP", len(ids)),
        ("puntos 3D (inicialización de las gaussianas)", n),
        ("largo medio de pista (imágenes por punto)", round(float(largo.mean()), 2)),
        ("% de puntos observados por alguna vista de PRUEBA", 100 * np.mean([bool(t & s_pru) for t in p])),
        ("% de puntos observados por alguna de las 21 vistas de ENTRENAMIENTO",
         100 * np.mean([bool(t & s_ent) for t in p])),
        ("% de puntos que NINGUNA vista de entrenamiento observa",
         100 * np.mean([not (t & s_ent) for t in p])),
        ("% de puntos con al menos 2 vistas de entrenamiento (triangulables solo con ellas)",
         100 * np.mean([len(t & s_ent) >= 2 for t in p])),
    ]
    tabla = pd.DataFrame(filas, columns=["medida", "valor"])
    tabla["valor"] = tabla["valor"].astype(float).round(2)

    # Vista usada para elegir lambda frente a la vista de prueba vecina.
    carpeta = RAIZ / "data" / "truck" / "images"
    a, b = cargar_imagen(carpeta / "000130.jpg"), cargar_imagen(carpeta / "000131.jpg")
    m = calcular_metricas(a, b, "cpu")
    tabla.loc[len(tabla)] = ["PSNR entre 000130 (elección de λ) y 000131 (prueba), dB",
                             round(m["psnr"], 2)]
    tabla.loc[len(tabla)] = ["SSIM entre 000130 y 000131", round(m["ssim"], 4)]
    print("vistas de prueba:", prueba)
    print(tabla.to_string(index=False))
    (RAIZ / "results" / "auditoria").mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida, index=False)


if __name__ == "__main__":
    main()
