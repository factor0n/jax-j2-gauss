#!/usr/bin/env bash
# Run the whole GPU part and build a zip file with the results.
#
# Usage (with the environment active: source .venv-gpu/bin/activate):
#     bash gpu/run_all.sh            # full run, N up to 1e7 (~10 to 15 min)
#     bash gpu/run_all.sh --quick    # quick try, N up to 1e5 (~1 min)
# Options are passed on to bench_gpu.py (for example --tangent, --chunk 524288).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HOST="$(hostname)"
LOG="benchmarks/results/run_gpu_${HOST}.log"
mkdir -p benchmarks/results figures
PRETEND=""
[[ "${PRETEND_GPU:-0}" == "1" ]] && PRETEND="--pretend-gpu"   # only for testing without a GPU

{
echo "== $(date '+%Y-%m-%d %H:%M') sur $HOST"
nvidia-smi --query-gpu=name,driver_version,memory.total,pstate,power.draw,temperature.gpu --format=csv 2>/dev/null || true

echo; echo "== 1/4 Tests unitaires (la loi est-elle correcte sur cette machine ?)"
python -m pytest -q tests 2>&1 | tail -n 3

echo; echo "== 2/4 Diagnostic de la machine"
python gpu/check_gpu.py $PRETEND

echo; echo "== 3/4 Benchmark"
python gpu/bench_gpu.py $PRETEND "$@"

echo; echo "== 4/4 Figures et résumé"
python gpu/plot_gpu.py
} 2>&1 | tee "$LOG"

ZIP="gpu_results_${HOST}.zip"
rm -f "$ZIP"
python - "$ZIP" "$HOST" <<'EOF'
import glob, sys, zipfile
zf, host = sys.argv[1], sys.argv[2]
files = glob.glob(f"benchmarks/results/*{host}*") + glob.glob(f"figures/gpu_*{host}*")
with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f)
print(f"\nArchive prête : {zf} ({len(files)} fichiers) -> envoie-la pour mettre à jour le rapport.")
EOF
