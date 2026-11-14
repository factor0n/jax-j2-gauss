"""Benchmark figures from one or more CSV files written by bench_cpu_gpu.py.

Usage: python benchmarks/plot_bench.py benchmarks/results/bench_*.csv [--name suffix]
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import SERIES, INK2, plt, save  # noqa: E402

LABELS = {"numpy": "NumPy (CPU)", "jax-cpu": "JAX CPU", "jax-gpu": "JAX GPU (calcul seul)",
          "jax-gpu-transfer": "JAX GPU + transferts"}
COLORS = {"numpy": SERIES[2], "jax-cpu": SERIES[0], "jax-gpu": SERIES[1], "jax-gpu-transfer": SERIES[3]}
MARK = {"numpy": "^", "jax-cpu": "o", "jax-gpu": "s", "jax-gpu-transfer": "D"}

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--name", default="bench")
args = ap.parse_args()
df = pd.concat([pd.read_csv(f) for f in args.csv], ignore_index=True)
df = df[df.status == "ok"].reset_index(drop=True)
precs = [p for p in ("fp64", "fp32") if p in set(df.precision)]

# --- throughput --------------------------------------------------------------
fig, axes = plt.subplots(1, len(precs), figsize=(5.2 * len(precs), 3.8), sharey=True, squeeze=False)
for ax, prec in zip(axes[0], precs):
    for cfg, lab in LABELS.items():
        d = df[(df.config == cfg) & (df.precision == prec)].sort_values("N")
        if len(d):
            ax.loglog(d.N, d.throughput_pts_s, marker=MARK[cfg], color=COLORS[cfg], label=lab)
    ax.set_title(prec.upper())
    ax.set_xlabel("nombre de points de Gauss $N$")
axes[0][0].set_ylabel("débit (points / s)")
axes[0][0].legend(loc="lower right")
save(fig, f"{args.name}_throughput")

# --- effective bandwidth vs triad (1D roofline) ------------------------------
fig, axes = plt.subplots(1, len(precs), figsize=(5.2 * len(precs), 3.8), sharey=True, squeeze=False)
for ax, prec in zip(axes[0], precs):
    for cfg, lab in LABELS.items():
        d = df[(df.config == cfg) & (df.precision == prec)].sort_values("N")
        if len(d):
            ax.loglog(d.N, d.eff_GBps, marker=MARK[cfg], color=COLORS[cfg], label=lab)
    for dev, ls in (("cpu", "--"), ("gpu", ":")):
        d = df[(df.precision == prec) & ((df.device == "cpu") if dev == "cpu" else (df.device != "cpu"))]
        if len(d):
            bw = d.triad_GBps.median()
            ax.axhline(bw, color=INK2, ls=ls, lw=1)
            ax.text(df.N.min(), bw * 0.8, f"triade {dev.upper()} : {bw:.0f} Go/s", color=INK2, fontsize=8)
    ax.set_title(prec.upper())
    ax.set_xlabel("nombre de points de Gauss $N$")
axes[0][0].set_ylabel("bande passante effective (Go/s)")
axes[0][0].legend(loc="lower right")
save(fig, f"{args.name}_bandwidth")

# --- compile time --------------------------------------------------------------
d = df[df.config.isin(["jax-cpu", "jax-gpu"])]
print("compilation (s) :")
print(d.groupby(["config", "precision"]).compile_s.describe()[["min", "50%", "max"]].round(3))
print("flops/point, octets XLA/point :")
print(d.groupby(["config", "precision"])[["flops_per_pt", "xla_bytes_per_pt"]].median().round(1))
best = df.loc[df.groupby(["config", "precision"]).throughput_pts_s.idxmax()]
print(best[["config", "precision", "N", "throughput_pts_s", "eff_GBps", "triad_GBps"]])
