#!/usr/bin/env bash
# Cola: 3DGS en rejilla 512x1024 (se ejecuta dentro de WSL).
cd /root/spc-gs || exit 1
source scripts/entorno.sh
C=configs/truck_512x1024_3d.yaml
python -m src.cli reconstruir --config $C 2>&1 | tail -5
python -m src.cli entrenar --config $C
python -m src.cli evaluar --config $C
echo "FIN $(date)"
