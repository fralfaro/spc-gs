# Variables de entorno del proyecto (uso: source scripts/entorno.sh).
# Se definen aquí porque ~/.bashrc no se lee en shells no interactivos.
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# RTX 4060 = arquitectura Ada (8.9): compilar gsplat solo para ella es más rápido.
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS="${MAX_JOBS:-4}"
# Activa el ambiente de uv si existe.
_RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_RAIZ/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$_RAIZ/.venv/bin/activate"
fi
