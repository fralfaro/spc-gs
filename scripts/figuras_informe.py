"""Figuras del informe (en inglés) a partir de results/*.csv y results/truck_256x512/recon."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pathlib import Path

R = Path(__file__).resolve().parents[1]
S = float(os.environ.get("FS", "1"))  # factor de tamaño de letra (póster: 1.5)
OUT = R / ("latex/poster/figs_spc" if S != 1 else "latex/informe/figs_spc"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8 * S, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .3})
EST = {("hadamard", "directa"): ("Hadamard, direct", "#1f77b4", "--", "o"), ("hadamard", "tv"): ("Hadamard, TV", "#d95f02", "-", "o"),
       ("random", "directa"): ("Random, direct", "#1b9e77", "--", "s"), ("random", "tv"): ("Random, TV", "#e6ab02", "-", "s")}

d2 = pd.read_csv(R / "results/metrics_2d.csv")
# 1) 2D: PSNR / SSIM / LPIPS vs ratio en las dos rejillas (PSNR) y SSIM
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
for a, (rej, ttl) in zip(ax[:2], [("256x512", "PSNR (dB), 256×512"), ("512x1024", "PSNR (dB), 512×1024")]):
    s = d2[d2.rejilla == rej]
    for (p, m), (lab, c, ls, mk) in EST.items():
        t = s[(s.patron == p) & (s.metodo == m)].sort_values("razon")
        a.plot(t.razon * 100, t.PSNR, ls, color=c, marker=mk, ms=3.5 * S, label=lab)
    a.axhline(s[s.patron == "control"].PSNR.iloc[0], color="k", ls=":", lw=1); a.text(50, s[s.patron == "control"].PSNR.iloc[0] + .3, "control", ha="right", fontsize=7*S)
    a.set_title(ttl, fontsize=8*S); a.set_xlabel("measurements (%)"); a.set_xticks([5, 10, 25, 50])
h_, l_ = ax[0].get_legend_handles_labels()
s = d2[d2.rejilla == "256x512"]
for (p, m), (lab, c, ls, mk) in EST.items():
    t = s[(s.patron == p) & (s.metodo == m)].sort_values("razon")
    ax[2].plot(t.razon * 100, t.LPIPS, ls, color=c, marker=mk, ms=3.5 * S)
ax[2].axhline(s[s.patron == "control"].LPIPS.iloc[0], color="k", ls=":", lw=1)
ax[2].set_title("LPIPS, 256×512", fontsize=8*S); ax[2].set_xlabel("measurements (%)"); ax[2].set_xticks([5, 10, 25, 50])
fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=7*S, frameon=False); fig.tight_layout(rect=[0, 0.1, 1, 1]); fig.savefig(OUT / "fig_2d_metrics.png", dpi=220); plt.close(fig)

# 2) 3D: metricas en vistas de prueba a 7000 iteraciones
d3 = pd.read_csv(R / "results/metrics_3d.csv"); d3 = d3[(d3.semilla == 449) & (d3.iteraciones == 7000)]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
for a, (col, ttl) in zip(ax, [("PSNR", "Test PSNR (dB)"), ("SSIM", "Test SSIM"), ("LPIPS", "Test LPIPS (lower is better)")]):
    for (p, m), (lab, c, ls, mk) in EST.items():
        t = d3[(d3.patron == p) & (d3.metodo == m)].sort_values("razon")
        a.plot(t.razon * 100, t[col], ls, color=c, marker=mk, ms=3.5 * S, label=lab)
    for cond, st, lab in [("original", "-.", "baseline (original)"), ("control_256x512", ":", "control")]:
        a.axhline(d3[d3.condicion == cond][col].iloc[0], color="k", ls=st, lw=1, label=lab)
    a.set_title(ttl, fontsize=8*S); a.set_xlabel("measurements (%)"); a.set_xticks([5, 10, 25, 50])
ax[0].legend(fontsize=5.5*S, loc="center right", frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_3d_test.png", dpi=220); plt.close(fig)

# 3) 2D -> 3D en vistas de entrenamiento (21)
pv = pd.read_csv(R / "results/metrics_2d_por_vista.csv"); pv = pv[(pv.rejilla == "256x512") & (pv.conjunto == "entrenamiento")]
m2 = pv.groupby("condicion").PSNR.mean()
fig, ax = plt.subplots(figsize=(3.4, 2.7))
for (p, m), (lab, c, ls, mk) in EST.items():
    t = d3[(d3.patron == p) & (d3.metodo == m)]
    ax.scatter([m2[x] for x in t.condicion], t.PSNR_ent, color=c, marker=mk, s=18, label=lab)
for cond, lab in [("control_256x512", "control")]:
    ax.scatter(m2[cond], d3[d3.condicion == cond].PSNR_ent.iloc[0], color="k", marker="D", s=22, label=lab)
ax.axhline(d3[d3.condicion == "original"].PSNR_ent.iloc[0], color="k", ls="-.", lw=1); ax.text(12.5, d3[d3.condicion == "original"].PSNR_ent.iloc[0] + .4, "baseline (original images)", fontsize=6*S)
ax.plot([12, 30], [12, 30], color="gray", lw=.6, ls=":")
ax.set_xlabel("2D PSNR of the 21 training views (dB)"); ax.set_ylabel("3DGS PSNR on the same views (dB)")
ax.legend(fontsize=6*S, frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "fig_2d_to_3d.png", dpi=220); plt.close(fig)

# 4) visual 2D
rec = R / "results/truck_256x512/recon"; v = "000208.jpg"
paneles = [("Original", R / "data/truck/images" / v), ("Control", rec / "control_256x512" / v)]
paneles += [(f"Hadamard+TV {p}%", rec / f"hadamard_tv_{p:02d}_256x512" / v) for p in (5, 10, 25, 50)]
paneles += [("Hadamard direct 10%", rec / "hadamard_directa_10_256x512" / v), ("Random direct 25%", rec / "random_directa_25_256x512" / v),
            ("Random+TV 25%", rec / "random_tv_25_256x512" / v)]
fig, ax = plt.subplots(2, 5, figsize=(7.2, 2.5))
for a, (t, pth) in zip(ax.ravel(), paneles):
    a.imshow(Image.open(pth)); a.set_title(t, fontsize=6.5*S); a.axis("off")
for a in ax.ravel()[len(paneles):]: a.axis("off")
fig.tight_layout(pad=.3); fig.savefig(OUT / "fig_2d_visual.png", dpi=220); plt.close(fig)
print("ok")
