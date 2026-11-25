# spc-gs: cámara de un píxel simulada + Gaussian Splatting

Proyecto del curso INF449 (Fotografía Computacional). Autores: Francisco Alfaro y Ayrton Perez.

## El proyecto

Una **cámara de un píxel** no tiene sensor matricial: proyecta patrones sobre la escena y un único
detector mide la luz total de cada uno. Recuperar una imagen exige muchas mediciones, y una escena
3D exige muchas imágenes.

**Objetivo:** medir cuánta calidad se pierde, en las imágenes 2D reconstruidas y en un modelo
**3D Gaussian Splatting**, al usar solo el 5, 10, 25 o 50 % de las mediciones. La cámara se simula
sobre la escena *Truck* (Tanks and Temples) y se comparan patrones (Hadamard, aleatorio, DCT) y
reconstrucciones (directa y variación total).

**Resultado principal:** con Hadamard + TV la pérdida frente al control sin cámara es de 5,70 dB
(5 %) a 0,89 dB (50 %); el modelo 3D hereda la degradación 2D.

## Contenido

| Qué | Dónde |
|---|---|
| Informe (paper de congreso) | [latex/informe/conference_101719.pdf](latex/informe/conference_101719.pdf) |
| Póster | [latex/poster/main3.pdf](latex/poster/main3.pdf) |
| Presentación | [slides/spc_gs/presentacion2.pdf](slides/spc_gs/presentacion2.pdf) |
| Notebooks | [notebooks/](notebooks/) |
| Código | [src/](src/) y [scripts/](scripts/) |
| Más detalle técnico | [docs/](docs/) |

## Cómo ejecutarlo

Requiere una GPU NVIDIA y Windows con WSL2 (Ubuntu 24.04) con CUDA 12.8 y [uv](https://docs.astral.sh/uv/);
el detalle de esa preparación está en [docs/03_guia_tecnica.md](docs/03_guia_tecnica.md).

```bash
# 1) Clonar
git clone https://github.com/fralfaro/spc-gs.git
cd spc-gs

# 2) Crear el ambiente virtual e instalar dependencias
uv venv --python 3.12 .venv
source scripts/entorno.sh
uv pip install -r requirements.txt

# 3) Descargar la escena Truck en data/truck/ (ver docs/03_guia_tecnica.md)

# 4) Ejecutar los notebooks, en orden
jupyter lab notebooks/
```

Para repetir el experimento completo: `bash scripts/run_experiment.sh`.
