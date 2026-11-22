"""Figura de patrones (Hadamard por secuencia y aleatorio) para el informe y el póster."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.patterns import patron_2d, indices_medidos

R = Path(__file__).resolve().parents[1]
shape = (64, 128)  # misma estructura que 256x512, solo para dibujar
fig, ax = plt.subplots(2, 5, figsize=(7.2, 2.2))
h, w = shape
coefs = [(0, 1), (1, 0), (1, 1), (3, 5), (12, 20)]
for a, (u, v) in zip(ax[0], coefs):
    a.imshow(patron_2d("hadamard", shape, u * w + v), cmap="gray", interpolation="nearest"); a.set_title(f"Hadamard ({u},{v})", fontsize=7); a.axis("off")
idx = indices_medidos("random", shape, 0.25)
for a, k in zip(ax[1], idx[[1, 40, 400, 1500, 2000]]):
    a.imshow(patron_2d("random", shape, k), cmap="gray", interpolation="nearest"); a.set_title(f"random #{k}", fontsize=7); a.axis("off")
fig.tight_layout(pad=.4)
for d in ("informe", "poster"):
    fig.savefig(R / "latex" / d / "figs_spc" / "fig_patterns.png", dpi=220)
