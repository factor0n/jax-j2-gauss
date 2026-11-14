"""Shared settings for the GPU scripts. Import this BEFORE jax."""

import os
import platform
import shutil
import socket
import subprocess
import sys

# By default JAX reserves 75 % of the GPU memory on the first call. We turn this off
# so we can measure the memory that is really needed and avoid false "OOM" errors.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "benchmarks", "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def machine_info():
    """Information about the machine, saved with each result."""
    info = {"host": socket.gethostname(), "os": platform.platform(),
            "python": platform.python_version(), "cpu_count": os.cpu_count()}
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpu = platform.processor()
    info["cpu_model"] = cpu
    if shutil.which("nvidia-smi"):
        q = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit",
                  "--format=csv,noheader"])
        if q:
            parts = [p.strip() for p in q.splitlines()[0].split(",")]
            keys = ["gpu_name", "driver", "vram", "sm_clock_max", "power_limit"]
            info.update(dict(zip(keys, parts)))
    ac = ""
    import glob
    for d in glob.glob("/sys/class/power_supply/*"):
        try:
            if open(os.path.join(d, "type")).read().strip() == "Mains":
                ac = "secteur" if open(os.path.join(d, "online")).read().strip() == "1" else "batterie"
                break
        except OSError:
            continue
    info["alimentation"] = ac or "inconnue"
    return info
