#!/usr/bin/env bash
# Entrena 3DGS (gsplat) sobre una condición y evalúa contra las imágenes originales.
#
# Uso:
#   bash scripts/train_gsplat.sh <escena> <condicion> [iteraciones] [semilla]
# Ejemplo:
#   bash scripts/train_gsplat.sh truck original 3000 449
#   bash scripts/train_gsplat.sh truck hadamard_tv_10_256x512 7000
#
# Variables opcionales:
#   FACTOR=2   reduce la resolución de entrenamiento a la mitad (menos memoria)
#   FORZAR=1   vuelve a entrenar aunque ya exista el modelo
#
# Requisito: la condición debe existir (crearla con python -m src.make_dataset).
set -euo pipefail

ESCENA="${1:?Falta la escena (p. ej. truck)}"
CONDICION="${2:?Falta la condición (p. ej. original)}"
ITER="${3:-3000}"
SEMILLA="${4:-449}"
FACTOR="${FACTOR:-1}"

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"
source scripts/entorno.sh

DATOS="data/conditions/$ESCENA/$CONDICION"
SALIDA="outputs/$ESCENA/$CONDICION/it${ITER}_s${SEMILLA}_f${FACTOR}"

if [ ! -f "$DATOS/vistas.json" ]; then
    echo "No existe la condición $DATOS. Créala primero con src/make_dataset.py" >&2
    exit 1
fi

if [ -f "$SALIDA/ckpt.pt" ] && [ "${FORZAR:-0}" != "1" ]; then
    echo "[train_gsplat] ya existe $SALIDA/ckpt.pt (usa FORZAR=1 para repetir)"
else
    mkdir -p "$SALIDA"
    python -m src.train_3dgs --datos "$DATOS" --salida "$SALIDA" \
        --iteraciones "$ITER" --semilla "$SEMILLA" --factor "$FACTOR" \
        2>&1 | tee "$SALIDA/train.log"
fi

# Evaluación: renders de las vistas de prueba vs imágenes ORIGINALES.
# Si ya se evaluó, no se repite (evita filas duplicadas en results/metrics_3d.csv).
if [ -f "$SALIDA/metricas.json" ] && [ "${FORZAR:-0}" != "1" ]; then
    echo "[train_gsplat] ya evaluado: $SALIDA/metricas.json"
else
    python scripts/eval_3d.py --corrida "$SALIDA"
fi
