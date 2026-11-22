"""Lectura mínima del modelo COLMAP binario (sparse/0): cámaras, imágenes y puntos.

Formato: https://colmap.github.io/format.html (archivos .bin).
Solo se admiten cámaras sin distorsión (PINHOLE y SIMPLE_PINHOLE), que es el
caso de Truck.
"""

import struct
from pathlib import Path

import numpy as np

# id de modelo COLMAP -> (nombre, número de parámetros)
MODELOS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
           3: ("RADIAL", 5), 4: ("OPENCV", 8)}


def _leer(f, fmt):
    fmt = "<" + fmt
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def leer_camaras(ruta):
    """Devuelve {id: dict(modelo, ancho, alto, K)} con K la matriz 3x3."""
    camaras = {}
    with open(ruta, "rb") as f:
        (n,) = _leer(f, "Q")
        for _ in range(n):
            cid, mid, ancho, alto = _leer(f, "iiQQ")
            nombre, npar = MODELOS[mid]
            p = _leer(f, "d" * npar)
            if nombre == "SIMPLE_PINHOLE":
                fx = fy = p[0]
                cx, cy = p[1], p[2]
            elif nombre == "PINHOLE":
                fx, fy, cx, cy = p
            else:
                raise NotImplementedError(f"Modelo de cámara no soportado: {nombre}")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            camaras[cid] = {"modelo": nombre, "ancho": ancho, "alto": alto, "K": K}
    return camaras


def qvec_a_rotacion(q):
    """Cuaternión COLMAP (w, x, y, z) -> matriz de rotación 3x3."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def leer_imagenes(ruta):
    """Devuelve {nombre: dict(camara, w2c)} con w2c la matriz 4x4
    mundo -> cámara (convención OpenCV de COLMAP)."""
    imagenes = {}
    with open(ruta, "rb") as f:
        (n,) = _leer(f, "Q")
        for _ in range(n):
            _id, *qt = _leer(f, "i" + "d" * 7)
            (cid,) = _leer(f, "i")
            nombre = b""
            while (c := f.read(1)) != b"\0":
                nombre += c
            (npts,) = _leer(f, "Q")
            f.seek(24 * npts, 1)  # se omiten los puntos 2D (x, y, id3D)
            w2c = np.eye(4)
            w2c[:3, :3] = qvec_a_rotacion(qt[:4])
            w2c[:3, 3] = qt[4:]
            imagenes[nombre.decode()] = {"camara": cid, "w2c": w2c}
    return imagenes


def leer_puntos(ruta):
    """Devuelve (xyz (P, 3) float32, rgb (P, 3) uint8)."""
    xyz, rgb = [], []
    with open(ruta, "rb") as f:
        (n,) = _leer(f, "Q")
        for _ in range(n):
            _id, x, y, z, r, g, b, _err = _leer(f, "QdddBBBd")
            (largo,) = _leer(f, "Q")
            f.seek(8 * largo, 1)  # se omite la pista (image_id, point2D_idx)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.uint8)


def leer_modelo(carpeta_sparse):
    """Lee cameras.bin, images.bin y points3D.bin de una carpeta sparse/0."""
    d = Path(carpeta_sparse)
    return (leer_camaras(d / "cameras.bin"), leer_imagenes(d / "images.bin"),
            *leer_puntos(d / "points3D.bin"))
