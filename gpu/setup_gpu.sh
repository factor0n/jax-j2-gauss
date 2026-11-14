#!/usr/bin/env bash
# Set up a Python environment with JAX for NVIDIA GPUs, for jax-j2-gauss.
#
# Usage (from the repository root):
#     bash gpu/setup_gpu.sh            # pick CUDA 12 / CUDA 13 automatically from the driver
#     bash gpu/setup_gpu.sh cuda12     # force CUDA 12
#     bash gpu/setup_gpu.sh cuda13     # force CUDA 13
#
# Only the NVIDIA DRIVER must be installed on the system: JAX downloads
# CUDA and cuDNN by itself as pip packages (about 3 GB).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-gpu"
cd "$ROOT"

echo "== 1/5 Vérification du pilote NVIDIA"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERREUR : nvidia-smi introuvable. Installe le pilote NVIDIA, par exemple :"
    echo "    sudo ubuntu-drivers install   puis redémarre"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)"

FLAVOR="${1:-auto}"
if [[ "$FLAVOR" == "auto" ]]; then
    if (( DRIVER_MAJOR >= 580 )); then FLAVOR="cuda13"; else FLAVOR="cuda12"; fi
fi
if [[ "$FLAVOR" == "cuda12" && DRIVER_MAJOR -lt 525 ]]; then
    echo "ERREUR : pilote $DRIVER_MAJOR trop ancien (il faut >= 525 pour CUDA 12, >= 580 pour CUDA 13)."
    exit 1
fi
echo "   pilote $DRIVER_MAJOR -> installation de jax[$FLAVOR]"

echo "== 2/5 Création de l'environnement virtuel $VENV"
command -v python3 >/dev/null || { echo "ERREUR : python3 introuvable"; exit 1; }
python3 -m venv "$VENV" || { echo "Si l'erreur parle de ensurepip : sudo apt install python3-venv"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null

echo "== 3/5 Installation de JAX GPU (peut prendre plusieurs minutes)"
# only one CUDA plugin at a time, otherwise you get the error "PJRT_Api already exists"
pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt >/dev/null 2>&1 || true
pip install --upgrade "jax[$FLAVOR]"

echo "== 4/5 Installation des autres dépendances"
pip install -r requirements.txt

echo "== 5/5 Test : JAX voit-il le GPU ?"
python - <<'EOF'
import jax
devs = jax.devices()
print("JAX", jax.__version__, "->", devs)
if not any(d.platform == "gpu" for d in devs):
    raise SystemExit("ÉCHEC : JAX ne voit pas le GPU. Voir la section Dépannage de gpu/README_GPU.md")
print("OK : le GPU est utilisable.")
EOF

echo
echo "Installation terminée. Pour la suite :"
echo "    source .venv-gpu/bin/activate"
echo "    bash gpu/run_all.sh"
