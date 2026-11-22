"""Métricas de calidad de imagen: PSNR, SSIM y LPIPS-VGG.

Se usa una única función, `calcular_metricas`, en todas las condiciones
(reconstrucción 2D y renders de 3DGS), para que los números sean comparables.

- PSNR (dB): con rango de datos 1.0 (imágenes en [0, 1]). Mayor es mejor.
- SSIM: ventana gaussiana 11x11 con sigma 1.5 (definición de Wang et al.
  2004, la usual en 3DGS), promediada sobre los canales RGB. Mayor es mejor.
- LPIPS: red VGG (paquete `lpips`), entradas escaladas a [-1, 1].
  Menor es mejor.
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_LPIPS = {}  # caché del modelo LPIPS por dispositivo (se carga una sola vez)


def _modelo_lpips(device):
    if device not in _LPIPS:
        import warnings

        import lpips  # importación diferida: carga pesos de VGG la primera vez
        with warnings.catch_warnings():  # lpips usa una API antigua de torchvision
            warnings.simplefilter("ignore", UserWarning)
            _LPIPS[device] = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    return _LPIPS[device]


def a_numpy(img):
    """Convierte a numpy float64 (h, w, 3) en [0, 1].
    Acepta tensores (3, h, w) o (h, w, 3) y arreglos numpy."""
    if isinstance(img, torch.Tensor):
        img = img.detach().float().cpu().numpy()
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0.0, 1.0)


def psnr_ssim(pred, ref):
    """PSNR y SSIM (mismas opciones que calcular_metricas, sin LPIPS): (psnr, ssim)."""
    pred = a_numpy(pred)
    ref = a_numpy(ref)
    if pred.shape != ref.shape:
        raise ValueError(f"Tamaños distintos: {pred.shape} vs {ref.shape}")
    psnr = peak_signal_noise_ratio(ref, pred, data_range=1.0)
    ssim = structural_similarity(ref, pred, data_range=1.0, channel_axis=2,
                                 gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False)
    return float(psnr), float(ssim)


def calcular_metricas(pred, ref, device=None):
    """Devuelve {"psnr", "ssim", "lpips"} entre `pred` y la referencia `ref`.

    Ambas imágenes deben tener el mismo tamaño, valores en [0, 1].
    """
    pred = a_numpy(pred)
    ref = a_numpy(ref)
    psnr, ssim = psnr_ssim(pred, ref)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = _modelo_lpips(device)

    def a_tensor(a):
        t = torch.from_numpy(a).permute(2, 0, 1)[None].float().to(device)
        return t * 2 - 1  # LPIPS espera entradas en [-1, 1]

    with torch.no_grad():
        lp = modelo(a_tensor(pred), a_tensor(ref)).item()
    return {"psnr": float(psnr), "ssim": float(ssim), "lpips": float(lp)}
