"""Performance model (roofline + latencies) for an RTX 2050 Laptop GPU, compared with CPU measurements.

This is a PREDICTION, to be replaced by the bench_cpu_gpu.py measurement on the machine with the GPU.

    t_compute(N)   = t_launch + N * max(bytes / BW, flops / F_peak)
    t_transfer(N)  = t_compute(N) + 8 t_PCIe_latency + N * bytes / BW_PCIe

Vendor numbers (RTX 2050 Laptop, GA107): 2048 CUDA cores at 1.47 GHz, 64-bit GDDR6
at 14 Gbit/s (112 GB/s), PCIe 3.0 x8. FP64 throughput of consumer Ampere GPUs: 1/64 of FP32.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import SERIES, INK2, plt, save  # noqa: E402

FP32_PEAK = 2048 * 2 * 1.47e9            # FMA = 2 flops -> ~6.0 TFLOP/s
FP64_PEAK = FP32_PEAK / 64               # ~94 GFLOP/s
BW = 0.8 * 112e9                         # 80 % of the theoretical bandwidth
PCIE = 6.0e9                             # PCIe 3.0 x8: 7.9 GB/s in theory, ~6 in practice
T_LAUNCH, T_PCIE = 20e-6, 15e-6          # latencies (s)
FLOPS = {"fp64": 300 + 10 * 20, "fp32": 300 + 10 * 2}   # ~300 flops + ~10 exp (emulated FP64 exp ~20 flops)
BYTES = {"fp64": 38 * 8, "fp32": 38 * 4}            # minimum: read the inputs, write the outputs
XLA_BYTES = {"fp64": 1072, "fp32": 536}             # real traffic estimated by XLA (12 fusions + loop)

N = np.logspace(2, 7, 60)
csv = os.path.join(ROOT, "benchmarks", "results", "bench_cloud-vm-2c.csv")
meas = pd.read_csv(csv) if os.path.exists(csv) else None

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
rows = []
for ax, prec in zip(axes, ("fp64", "fp32")):
    peak = FP64_PEAK if prec == "fp64" else FP32_PEAK
    per_pt = max(BYTES[prec] / BW, FLOPS[prec] / peak)
    bound = "calcul" if FLOPS[prec] / peak > BYTES[prec] / BW else "mémoire"
    t_c = T_LAUNCH + N * per_pt
    t_t = t_c + 8 * T_PCIE + N * BYTES[prec] / PCIE
    per_xla = max(XLA_BYTES[prec] / BW, FLOPS[prec] / peak)
    t_x = T_LAUNCH + N * per_xla
    ax.fill_between(N, N / t_x, N / t_c, color=SERIES[1], alpha=0.12, lw=0)
    ax.loglog(N, N / t_c, color=SERIES[1], ls="--", label="GPU calcul seul (modèle, trafic minimal)")
    ax.loglog(N, N / t_x, color=SERIES[1], ls=":", label="GPU calcul seul (modèle, trafic XLA)")
    ax.loglog(N, N / t_t, color=SERIES[3], ls="--", label="GPU + transferts (modèle)")
    if meas is not None:
        d = meas[(meas.config == "jax-cpu") & (meas.precision == prec)].sort_values("N")
        ax.loglog(d.N, d.throughput_pts_s, "o-", color=SERIES[0], label="JAX CPU (mesuré, VM 2 cœurs)")
    ax.axhline(PCIE / BYTES[prec], color=INK2, lw=0.8, ls=":")
    ax.text(1.2e2, PCIE / BYTES[prec] * 1.15, "plafond PCIe", color=INK2, fontsize=8)
    ax.set_title(f"{prec.upper()} (RTX 2050 : modèle)")
    ax.set_xlabel("nombre de points de Gauss $N$")
    rows.append((prec, bound, 1 / per_xla, 1 / per_pt, PCIE / BYTES[prec], FLOPS[prec] / BYTES[prec], peak / BW))
axes[0].set_ylabel("débit (points / s)")
axes[0].legend(loc="lower right", fontsize=8)
save(fig, "gpu_model")

for prec, bound, thr_x, thr, pcie, I, ridge in rows:
    print(f"{prec}: limité par {bound}; débit asymptotique calcul {thr_x:.2e} à {thr:.2e} pts/s ; "
          f"avec transferts <= {pcie:.2e} pts/s ; intensité {I:.2f} flop/o ; point d'équilibre {ridge:.2f} flop/o")
