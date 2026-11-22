"""Escritura protegida de resultados.

Regla del proyecto: ningún script sobrescribe un archivo existente de results/ (ni de data/)
salvo que se pida explícitamente con --forzar. Para regenerar algo sin perder lo anterior,
se escribe en una carpeta nueva.
"""

import sys
from pathlib import Path


def proteger(*rutas, forzar=False):
    """Termina con un mensaje si alguna de `rutas` ya existe y no se pidió `forzar`."""
    existentes = [Path(r) for r in rutas if Path(r).exists()]
    if existentes and not forzar:
        lista = "\n  ".join(str(r) for r in existentes)
        sys.exit(f"Ya existen y no se sobrescriben (use --forzar para regenerarlos):\n  {lista}")
