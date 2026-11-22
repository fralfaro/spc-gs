#!/usr/bin/env bash
# Experimento completo en una escena:
#   1) crea las condiciones (original, control y SPC) y sus métricas 2D;
#   2) entrena 3DGS en cada condición y evalúa contra las imágenes originales.
#
# Uso:
#   bash scripts/run_experiment.sh                # valores por defecto
#   ITERS="3000 7000" PATRONES="hadamard random" bash scripts/run_experiment.sh
#
# Variables (con sus valores por defecto):
#   ESCENA=truck  REJILLA=256x512  SEMILLA=449  FACTOR=1
#   PATRONES="hadamard random"  METODOS="directa tv"  RAZONES="0.05 0.10 0.25 0.50"
#   ITERS="3000 7000"
#   PATRONES_2D="dct"   patrones adicionales solo para las métricas 2D (sin 3DGS)
#   SOLO_2D=1           crea las condiciones y métricas 2D, sin entrenar
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"
source scripts/entorno.sh

ESCENA="${ESCENA:-truck}"
REJILLA="${REJILLA:-256x512}"
SEMILLA="${SEMILLA:-449}"
PATRONES="${PATRONES:-hadamard random}"
METODOS="${METODOS:-directa tv}"
RAZONES="${RAZONES:-0.05 0.10 0.25 0.50}"
ITERS="${ITERS:-3000 7000}"
PATRONES_2D="${PATRONES_2D:-dct}"

# Crea una condición solo si no existe todavía.
crear() {
    local nombre="$1"; shift
    if [ -f "data/conditions/$ESCENA/$nombre/vistas.json" ] && \
       { [ "$nombre" = "original" ] || [ -f "data/conditions/$ESCENA/$nombre/metricas_2d.csv" ]; }; then
        echo "[2D] ya existe $nombre"
    else
        python -m src.make_dataset --escena "$ESCENA" --rejilla "$REJILLA" \
            --semilla "$SEMILLA" "$@"
    fi
}

pct() { python -c "print(f'{round($1*100):02d}')"; }

# --- 1) Condiciones y métricas 2D -------------------------------------------
CONDICIONES=("original" "control_${REJILLA}")
crear original --condicion original
crear "control_${REJILLA}" --condicion control
for P in $PATRONES $PATRONES_2D; do
    for M in $METODOS; do
        for R in $RAZONES; do
            NOMBRE="${P}_${M}_$(pct "$R")_${REJILLA}"
            crear "$NOMBRE" --condicion spc --patron "$P" --metodo "$M" --razon "$R"
            # Solo los patrones de PATRONES pasan a 3DGS.
            if [[ " $PATRONES " == *" $P "* ]]; then CONDICIONES+=("$NOMBRE"); fi
        done
    done
done

if [ "${SOLO_2D:-0}" = "1" ]; then
    echo "SOLO_2D=1: se omite el entrenamiento 3DGS."
    exit 0
fi

# --- 2) Entrenamiento y evaluación 3DGS --------------------------------------
for IT in $ITERS; do
    for C in "${CONDICIONES[@]}"; do
        echo "=== 3DGS: $C, $IT iteraciones ==="
        bash scripts/train_gsplat.sh "$ESCENA" "$C" "$IT" "$SEMILLA"
    done
done
