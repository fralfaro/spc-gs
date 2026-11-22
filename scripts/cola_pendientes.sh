#!/usr/bin/env bash
# Pendientes en GPU (se ejecuta dentro de WSL): termina 3DGS 512x1024, evalúa y valida el baseline con 251 vistas.
cd /root/spc-gs || exit 1
source scripts/entorno.sh
C=configs/truck_512x1024_3d.yaml
python -m src.cli reconstruir --config $C 2>&1 | tail -3
python -m src.cli entrenar --config $C
python -m src.cli evaluar --config $C
echo "FIN512 $(date)"
B=configs/truck_baseline_validacion.yaml
python -m src.cli reconstruir --config $B 2>&1 | tail -3
python -m src.cli entrenar --config $B
python -m src.cli evaluar --config $B
echo "FIN $(date)"
