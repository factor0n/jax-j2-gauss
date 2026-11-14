"""Figures and summary from the bench_gpu.py measurements (and check_gpu.py if available).

Usage:   python gpu/plot_gpu.py                     (uses the files of this machine)
         python gpu/plot_gpu.py --host izanami

Outputs:
  figures/gpu_throughput_<machine>.png/pdf   throughput (points/s) as a function of N
  figures/gpu_speedup_<machine>.png/pdf      speedup GPU / JAX CPU -> break-even point
  figures/gpu_breakdown_<machine>.png/pdf    where the time goes at the largest N: send / compute / return
  benchmarks/results/gpu_summary_<machine>.md  summary with numbers (to copy into the report)
"""

import argparse
import io
import json
import os
import sys

import _env  # noqa: F401
from _env import ROOT, RESULTS, machine_info

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import SERIES, INK2, plt  # noqa: E402

LAB = {"numpy": "NumPy (CPU)", "jax-cpu": "JAX CPU", "gpu-resident": "GPU, données sur le GPU",
       "gpu-transfer": "GPU + transferts PCIe", "jax-cpu+tangent": "JAX CPU, loi + tangente",
       "gpu+tangent": "GPU, loi + tangente"}
COL = {"numpy": SERIES[2], "jax-cpu": SERIES[0], "gpu-resident": SERIES[1], "gpu-transfer": SERIES[3],
       "jax-cpu+tangent": SERIES[0], "gpu+tangent": SERIES[1]}
MRK = {"numpy": "^", "jax-cpu": "o", "gpu-resident": "s", "gpu-transfer": "D",
       "jax-cpu+tangent": "o", "gpu+tangent": "s"}
BYTES = {"fp64": 8, "fp32": 4}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "figures", f"{name}.{ext}"))
    plt.close(fig)
    print("figure ->", os.path.join("figures", name + ".png"))


def load(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)))
    for c in ("throughput_pts_s", "time_median_s", "t_h2d_s", "t_compute_s", "t_d2h_s", "compile_s"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main(args):
    host = args.host or machine_info()["host"]
    csv_path = os.path.join(RESULTS, f"bench_gpu_{host}.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"{csv_path} introuvable : lance d'abord python gpu/bench_gpu.py")
    df = load(csv_path)
    ok = df[df.status == "ok"]
    chk_path = os.path.join(RESULTS, f"gpu_check_{host}.json")
    chk = json.load(open(chk_path)) if os.path.exists(chk_path) else None
    precs = [p for p in ("fp64", "fp32") if p in set(df.precision)]
    gname = str(df.gpu.dropna().iloc[0]) if df.gpu.notna().any() and str(df.gpu.dropna().iloc[0]) else "GPU"
    if bool(df.simulated_gpu.iloc[0]):
        gname = "GPU simulé (CPU)"

    # ---------------------------------------------------------------- 1. throughput
    fig, axes = plt.subplots(1, len(precs), figsize=(5.4 * len(precs), 4.0), sharey=True, squeeze=False)
    for ax, prec in zip(axes[0], precs):
        for cfg in LAB:
            d = ok[(ok.config == cfg) & (ok.precision == prec)].sort_values("N")
            if len(d):
                ls = "--" if "tangent" in cfg else "-"
                ax.loglog(d.N, d.throughput_pts_s, ls=ls, marker=MRK[cfg], color=COL[cfg], label=LAB[cfg], ms=4)
        oom = df[(df.status == "OOM") & (df.precision == prec) & (df.config == "gpu-resident")]
        res = ok[(ok.config == "gpu-resident") & (ok.precision == prec)].sort_values("N")
        if len(oom) and len(res):
            ax.annotate(f"mémoire GPU pleine\nà N={int(oom.N.min()):.0e}", (res.N.iloc[-1], res.throughput_pts_s.iloc[-1]),
                        xytext=(0, 22), textcoords="offset points", fontsize=7.5, color=SERIES[1], ha="center",
                        arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=0.8))
        if chk:
            q = 19 * BYTES[prec]
            ceil = 1.0 / (q / (chk["h2d_GBps"] * 1e9) + q / (chk["d2h_GBps"] * 1e9))
            ax.axhline(ceil, color=INK2, lw=0.9, ls=":")
            ax.text(ok.N.min(), ceil * 1.15, "plafond PCIe", fontsize=8, color=INK2)
            xla = {"fp64": 1072, "fp32": 536}[prec]   # bytes/point really moved (XLA cost_analysis)
            gceil = chk[f"triad_gpu_{prec}_GBps"] * 1e9 / xla
            ax.axhline(gceil, color=SERIES[1], lw=0.9, ls=":")
            ax.text(ok.N.min(), gceil * 1.12, "plafond mémoire GPU (trafic XLA)", fontsize=8, color=SERIES[1])
        ax.set_title(f"{prec.upper()} - {gname}")
        ax.set_xlabel("nombre de points de Gauss $N$")
    axes[0][0].set_ylabel("débit (points / s)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.24)
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    save(fig, f"gpu_throughput_{host}")

    # ---------------------------------------------------------------- 2. speedup
    fig, axes = plt.subplots(1, len(precs), figsize=(5.4 * len(precs), 3.8), sharey=True, squeeze=False)
    summary = []
    for ax, prec in zip(axes[0], precs):
        ref = ok[(ok.config == "jax-cpu") & (ok.precision == prec)].set_index("N").throughput_pts_s
        for cfg in ("gpu-resident", "gpu-transfer"):
            d = ok[(ok.config == cfg) & (ok.precision == prec)].set_index("N").throughput_pts_s
            common = d.index.intersection(ref.index)
            if len(common) == 0:
                continue
            sp = (d[common] / ref[common]).sort_index()
            ax.semilogx(sp.index, sp.values, marker=MRK[cfg], color=COL[cfg], label=LAB[cfg])
            # break-even point: smallest N from which the GPU wins for all larger N
            be = None
            for n in sp.index[::-1]:
                if sp[n] > 1.0:
                    be = int(n)
                else:
                    break
            summary.append((prec, cfg, float(sp.max()), be))
        ax.axhline(1.0, color=INK2, lw=1)
        ax.text(ok.N.min(), 1.2, "au-dessus : GPU plus rapide", fontsize=8, color=INK2)
        ax.set_yscale("log")
        ax.set_title(prec.upper())
        ax.set_xlabel("nombre de points de Gauss $N$")
    axes[0][0].set_ylabel("accélération par rapport à JAX CPU")
    axes[0][0].legend(fontsize=8)
    save(fig, f"gpu_speedup_{host}")

    # ---------------------------------------------------------------- 3. time breakdown
    tr = ok[ok.config == "gpu-transfer"].dropna(subset=["t_h2d_s"])
    bd = []
    if len(tr):
        bars = []   # (label, [(duration_ms, color, legend)])
        for prec in precs:
            t = tr[tr.precision == prec]
            if not len(t):
                continue
            r = t.loc[t.N.idxmax()]
            N = int(r.N)
            h, c, d = r.t_h2d_s, r.t_compute_s, r.t_d2h_s
            bd.append((prec, N, h, c, d))
            bars.append((f"GPU + transferts\n{prec.upper()}, N={N:.0e}",
                         [(h * 1e3, SERIES[3], "envoi hôte -> GPU"), (c * 1e3, SERIES[1], "calcul GPU"),
                          (d * 1e3, SERIES[6], "retour GPU -> hôte")]))
            cpu_t = ok[(ok.config == "jax-cpu") & (ok.precision == prec) & (ok.N == N)].time_median_s
            if len(cpu_t):
                bars.append((f"JAX CPU\n{prec.upper()}, N={N:.0e}",
                             [(float(cpu_t.iloc[0]) * 1e3, SERIES[0], "JAX CPU (tout)")]))
        fig, ax = plt.subplots(figsize=(7.5, 0.8 + 0.6 * len(bars)))
        seen = set()
        for y, (lab, segs) in enumerate(bars):
            left = 0.0
            for w, col, leg in segs:
                ax.barh(y, w, left=left, color=col, height=0.6, label=None if leg in seen else leg)
                seen.add(leg)
                left += w
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([b[0] for b in bars], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("temps pour traiter les N points (ms)")
        ax.legend(fontsize=8, loc="lower right")
        save(fig, f"gpu_breakdown_{host}")

    # ---------------------------------------------------------------- 4. summary
    lines = [f"# Résultats GPU - machine `{host}`", ""]
    with open(csv_path) as f:
        head = [ln[2:].strip() for ln in f if ln.startswith("#")]
    lines += [f"- {h}" for h in head] + [""]
    if chk:
        lines += ["## Plafonds mesurés (check_gpu.py)", "",
                  f"- PCIe hôte -> GPU : {chk['h2d_GBps']:.2f} Go/s ; GPU -> hôte : {chk['d2h_GBps']:.2f} Go/s",
                  f"- triade GPU FP64 : {chk['triad_gpu_fp64_GBps']:.0f} Go/s ; triade CPU FP64 : {chk['triad_cpu_fp64_GBps']:.1f} Go/s",
                  f"- produit de matrices GPU : FP32 {chk['matmul_gpu_fp32_GFLOPs']:.0f} GFLOP/s, "
                  f"FP64 {chk['matmul_gpu_fp64_GFLOPs']:.0f} GFLOP/s (rapport 1/{1/chk['fp64_over_fp32']:.0f})",
                  f"- latence d'un appel : {chk['launch_latency_us']:.0f} µs", ""]
    lines += ["## Meilleur débit par configuration (points/s)", "", "| configuration | précision | N | débit |",
              "|---|---|---|---|"]
    for (cfg, prec), g in ok.groupby(["config", "precision"]):
        r = g.loc[g.throughput_pts_s.idxmax()]
        lines.append(f"| {cfg} | {prec} | {int(r.N):,} | {r.throughput_pts_s:.2e} |")
    lines += ["", "## Accélération GPU / JAX CPU", "", "| précision | configuration | accélération max | GPU rentable dès N = |",
              "|---|---|---|---|"]
    for prec, cfg, mx, be in summary:
        lines.append(f"| {prec} | {cfg} | {mx:.2f} | {('%d' % be) if be else 'jamais'} |")
    if bd:
        lines += ["", "## Décomposition du temps (gpu-transfer, plus grand N)", "",
                  "| précision | N | envoi | calcul | retour |", "|---|---|---|---|---|"]
        for prec, N, h, c, d in bd:
            t = h + c + d
            lines.append(f"| {prec} | {N:,} | {100*h/t:.0f} % | {100*c/t:.0f} % | {100*d/t:.0f} % |")
    oom = df[df.status == "OOM"]
    if len(oom):
        lines += ["", "## Mémoire GPU insuffisante", ""] + \
                 [f"- {r.config} {r.precision} N={int(r.N):,}" for _, r in oom.iterrows()]
    out = os.path.join(RESULTS, f"gpu_summary_{host}.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nrésumé ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    main(ap.parse_args())
