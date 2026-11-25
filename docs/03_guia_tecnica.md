# spc-gs: cámara de un píxel simulada + Gaussian Splatting

Proyecto del curso INF449 (Fotografía Computacional).

## Objetivo

Una **cámara de un píxel** (*single pixel camera*, SPC) no tiene sensor matricial: un modulador
espacial (DMD) proyecta patrones sobre la escena y un único fotodetector registra, para cada
patrón, la luz total. Con `K` patrones se obtiene un vector `y = Φx` de `K` números, a partir del
cual se reconstruye la imagen `x`.

**Pregunta de investigación:** ¿cuánta calidad se pierde, en la imagen 2D reconstruida y en la
reconstrucción 3D con Gaussian Splatting (3DGS), al reducir las mediciones del single pixel al
5, 10, 25 y 50 % de los píxeles, según el **patrón** de medición y el **método** de
reconstrucción?

La cámara se simula sobre las imágenes reales de la escena **Truck** (Tanks and Temples): cada
vista se "fotografía" con la SPC, se reconstruye en 2D y, con las imágenes reconstruidas, se
entrena 3DGS con [gsplat](https://github.com/nerfstudio-project/gsplat).

## Flujo de trabajo

```
 data/truck/images (original, 979×546)
        │
        │  1. remuestreo a la rejilla SPC (256×512, promedio por área)
        ▼
  x en la rejilla ──► 2. MEDIR  y = Φx  (por canal RGB, razón r = 5/10/25/50 %)
                                  │
                                  ▼
                      3. RECONSTRUIR 2D: directa (Φᵀy) o TV-FISTA
                                  │
                                  │  4. vuelta al tamaño original (bicúbica)
                                  ▼
            data/conditions/truck/<condición>/images  (mismos nombres de archivo)
            + enlace simbólico al mismo sparse/0 (mismas cámaras COLMAP)
                                  │
                                  ▼
                      5. ENTRENAR 3DGS (gsplat) con 21 vistas
                                  │
                                  ▼
                      6. EVALUAR: render de 3 vistas de prueba
                         vs imágenes ORIGINALES → PSNR, SSIM, LPIPS
                         (complemento: las 21 vistas de entrenamiento,
                          también vs sus imágenes ORIGINALES)
```

### Qué se mide, qué se reconstruye y contra qué se compara

| Etapa | Qué se mide | Qué se reconstruye | Contra qué se compara | Dónde queda |
|---|---|---|---|---|
| Medición SPC | `y = Φx`: `K = r·256·512` productos internos por canal | — | — | (en memoria) |
| Reconstrucción 2D | — | la imagen en la rejilla (directa o TV) y luego a 979×546 | imagen **original** 979×546, en las 24 vistas | `results/metrics_2d.csv` |
| Control | — | imagen → rejilla → tamaño original, sin SPC | imagen **original** | `results/metrics_2d.csv` (patrón `control`) |
| 3DGS (principal) | — | la escena 3D a partir de 21 vistas (originales, control o SPC) | render de 3 vistas de **prueba** vs imágenes **originales** (nunca las degradadas) | `results/metrics_3d.csv` (`PSNR`, `SSIM`, `LPIPS`) |
| 3DGS (complemento) | — | el mismo modelo | render de las 21 vistas de **entrenamiento** vs sus imágenes **originales** | `results/metrics_3d.csv` (`PSNR_ent`, `SSIM_ent`, `LPIPS_ent`) |

Métricas (una sola función, `src/metrics.py`, para todas las condiciones): **PSNR** (dB, mayor
es mejor), **SSIM** (ventana gaussiana 11×11, σ = 1.5; mayor es mejor) y **LPIPS-VGG** (menor
es mejor).

### Condiciones

| Condición | Descripción |
|---|---|
| `original` | baseline: imágenes originales (enlaces simbólicos) |
| `control_256x512` | solo remuestreo a la rejilla y de vuelta: aísla la pérdida por resolución |
| `<patrón>_<método>_<pct>_<rejilla>` | SPC con patrón `hadamard`, `hadamard_cc`, `random` o `dct`; método `directa` o `tv`; razón 05, 10, 25 o 50 % |

Patrones (`src/patterns.py`), todos ortonormales tras normalizar (`ΦΦᵀ = I`):

- **hadamard**: Hadamard 2D separable, filas ordenadas por secuencia (cambios de signo) y
  coeficientes 2D de bajas a altas frecuencias. Se miden los primeros `K`.
- **hadamard_cc**: la misma base con orden *cake-cutting* (por número de regiones `(u+1)(v+1)`); se compara en `scripts/comparar_ordenes_hadamard.py`.
- **random**: binario aleatorio. Se permutan los píxeles (semilla) y se aplica Hadamard; luego se
  eligen `K` coeficientes al azar (semilla). Siempre incluye el patrón constante (intensidad total).
- **dct**: DCT-II 2D con orden en zigzag. No es binaria; sirve de referencia (solo se evalúa en 2D).

Métodos (`src/recon.py`):

- **directa**: `x = Φᵀy` (ceros en lo no medido; mínimos cuadrados de norma mínima).
- **tv**: `min ½‖Φx − y‖² + λ·TV(x)` con FISTA (paso 1, porque `ΦΦᵀ = I`). El proximal de la TV se
  resuelve con el algoritmo de Chambolle. λ = 0.003, 100 iteraciones.

## Requisitos

- Windows 11 con **WSL2** (Ubuntu 24.04) y driver NVIDIA para Windows (en WSL **no** se instala
  driver Linux).
- GPU NVIDIA con al menos ~3 GB libres (probado en RTX 4060 Laptop, 8 GB).
- En WSL: **CUDA toolkit 12.8**, `gcc` (build-essential), cabeceras de Python
  (`python3.12-dev`) y **uv**. gsplat compila sus kernels CUDA la primera vez que se usa
  (unos 3–4 min).
- RAM para WSL: se recomienda `C:\Users\<usuario>\.wslconfig` con

  ```ini
  [wsl2]
  memory=10GB
  swap=8GB
  ```

## Instalación

Dentro de Ubuntu (WSL2). Conviene trabajar en el sistema de archivos de Linux (`~/spc-gs`), no
en `/mnt/c`, porque la lectura de archivos y la compilación son mucho más rápidas.

```bash
# 1) Paquetes del sistema
sudo apt update
sudo apt install -y build-essential git wget python3.12-dev

# 2) CUDA toolkit 12.8 para WSL (repositorio wsl-ubuntu de NVIDIA; sin driver)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-8

# 3) uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4) Ambiente del proyecto
git clone https://github.com/fralfaro/spc-gs.git ~/spc-gs
cd ~/spc-gs
uv venv --python 3.12 .venv
source scripts/entorno.sh          # CUDA_HOME, PATH, TORCH_CUDA_ARCH_LIST=8.9 y activa .venv
uv pip install -r requirements.txt  # torch/torchvision desde el índice cu128 de PyTorch

# 5) Verificación (gsplat se compila la primera vez que se entrena)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q
```

`scripts/entorno.sh` debe cargarse (`source`) en cada terminal nueva. Fija
`TORCH_CUDA_ARCH_LIST=8.9` (arquitectura Ada de la RTX 4060); con otra GPU, cámbialo a su
*compute capability* (por ejemplo, 8.6 para Ampere).

Nota: como `requirements.txt` agrega el índice de PyTorch con `--extra-index-url`, uv también
toma de allí algunos paquetes comunes (por ejemplo `requests`) en versiones algo antiguas. No
afecta al proyecto.

## Datos: Truck

La escena debe quedar así (formato COLMAP):

```
data/truck/
├── images/        000001.jpg … 000251.jpg   (979×546)
└── sparse/0/      cameras.bin  images.bin  points3D.bin
```

Truck proviene de Tanks and Temples; la versión con COLMAP se distribuye junto con los datos del
artículo original de 3DGS (archivo `tandt_db.zip`, carpeta `tandt/truck`). La cámara COLMAP es
de 1957×1091 y las imágenes de 979×546: el código escala las intrínsecas al tamaño real de cada
imagen.

> `data/truck/` está en `.gitignore` y fuera del índice de git (los archivos locales no se borran; el historial anterior
> aún los contiene). Descarga la escena aparte, como se indica arriba.

## Orden de ejecución

```bash
source scripts/entorno.sh

# 0) Pruebas
python -m pytest -q

# 1) Demostración 2D (una vista)
jupyter lab notebooks/01_single_pixel_demo.ipynb

# 2) Baseline y control 3DGS
jupyter lab notebooks/02_gsplat_baseline.ipynb

# 3) Experimento completo (condiciones 2D + 3DGS a 3000 y 7000 iteraciones)
bash scripts/run_experiment.sh
#    variantes: SOLO_2D=1 bash scripts/run_experiment.sh
#               ITERS="3000" PATRONES="hadamard" bash scripts/run_experiment.sh

# 4) Tablas y gráficos
jupyter lab notebooks/03_resultados.ipynb
```

Piezas sueltas:

```bash
# Crear una condición (y agregar su fila a results/metrics_2d.csv)
python -m src.make_dataset --condicion spc --patron hadamard --metodo tv --razon 0.10 \
    --rejilla 256x512 --vistas 24

# Entrenar + evaluar una condición (agrega una fila a results/metrics_3d.csv)
bash scripts/train_gsplat.sh truck hadamard_tv_10_256x512 7000 449
FACTOR=2 bash scripts/train_gsplat.sh truck original 3000   # mitad de resolución

# Solo evaluar un modelo ya entrenado
python scripts/eval_3d.py --corrida outputs/truck/original/it3000_s449_f1

# Reevaluar todos los modelos de una escena (la evaluación es determinista);
# --extendida agrega las 230 vistas no usadas para entrenar y métricas por vista
python scripts/eval_3d.py --todas truck --extendida --csv results/metrics_3d.csv

# Análisis del control y diagnóstico visual de las vistas de prueba
python scripts/analisis_control.py
python scripts/diagnostico_prueba.py --corrida outputs/truck/original/it7000_s449_f1 \
    --comparar original control_256x512 hadamard_tv_05_256x512 hadamard_tv_50_256x512 random_tv_05_256x512

# Métricas 2D por vista desde las reconstrucciones guardadas (sin 3DGS); el gráfico 2D → 3D de
# notebooks/03_resultados usa su promedio en las 21 vistas de entrenamiento
python scripts/metricas_2d_por_vista.py        # -> results/metrics_2d_por_vista.csv

# Verificación de las dudas de la Fase 0 (escala y componente DC de la directa aleatoria, techo
# de la métrica 2D, fiabilidad de las vistas de prueba, elección de λ, valores de los patrones).
# Semilla 449 (--semilla); no modifica resultados; partes 1 y 3 usan la GPU unos 5 min.
python scripts/auditoria/verificacion_fase0.py   # -> results/auditoria/fase0/

# Variabilidad por semilla (ejemplo: semillas 450 y 451)
bash scripts/train_gsplat.sh truck original 3000 450
```

Opciones de `make_dataset.py`: `--rejilla 512x1024` (rejilla mayor), `--vistas`/`--prueba`
(número de vistas), `--lam`/`--iter-tv` (TV), y ruido opcional apagado por defecto:
`--fotones` (Poisson), `--sigma-lectura` (lectura) y `--bits 16` (cuantización).

## Estructura de carpetas

```
spc-gs/
├── src/
│   ├── patterns.py      patrones (Hadamard, aleatorio, DCT) e índices medidos
│   ├── spc.py           operador de medición y = Φx y ruido opcional
│   ├── recon.py         reconstrucción directa y TV-FISTA
│   ├── metrics.py       PSNR, SSIM y LPIPS-VGG (una sola función)
│   ├── make_dataset.py  crea las carpetas de cada condición + métricas 2D
│   ├── colmap_io.py     lector mínimo de COLMAP binario
│   └── train_3dgs.py    entrenador 3DGS mínimo con gsplat
├── scripts/
│   ├── entorno.sh       variables de entorno (CUDA, arquitectura) y activación del ambiente
│   ├── train_gsplat.sh  entrena y evalúa una condición
│   ├── eval_3d.py       renders (prueba, entrenamiento y, con --extendida, 230 vistas no usadas)
│   │                    vs originales → results/metrics_3d.csv y metrics_3d_por_vista.csv
│   ├── run_experiment.sh  experimento completo
│   ├── analisis_control.py   factores que determinan el PSNR del control (rejilla, JPEG, evaluación)
│   ├── diagnostico_prueba.py error, opacidad y n.º de vistas que observan cada píxel de prueba
│   ├── metricas_2d_por_vista.py  métricas 2D por vista desde las imágenes guardadas
│   └── auditoria/
│       └── verificacion_fase0.py  verificación de las dudas de la Fase 0 → results/auditoria/fase0/
├── notebooks/           01_single_pixel_demo, 02_gsplat_baseline, 03_resultados
├── tests/               pruebas (pytest)
├── results/             metrics_2d.csv, metrics_2d_por_vista.csv, metrics_3d.csv, auditoria/ y figuras/
├── data/truck/          escena (images/, sparse/0)
├── data/conditions/     condiciones generadas (no versionado)
└── outputs/             modelos y renders (no versionado)
```

## Documentación

- [`docs/00_objetivo_y_flujo.md`](docs/00_objetivo_y_flujo.md): objetivo, qué se mide, variables, flujo y pasos del experimento.
- [`docs/01_decisiones_y_configuracion.md`](docs/01_decisiones_y_configuracion.md): decisiones de diseño y configuración exacta.
- [`docs/02_resultados.md`](docs/02_resultados.md): tablas 2D y 3D, observaciones, 30000 iteraciones y limitaciones.
- [`latex/informe/`](latex/informe/): informe (inglés, IEEEtran) con todos los resultados.
