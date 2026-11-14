"""Benchmark of batched integration: NumPy, JAX CPU, JAX GPU (compute only / with transfers).

Usage:
    python benchmarks/bench_cpu_gpu.py                      # everything that is available
    python benchmarks/bench_cpu_gpu.py --nmax 6.5 --configs jax-gpu jax-gpu-transfer

For each configuration and each N (Gauss points) we measure:
  * the compile time (lower + compile, "cold" call);
  * the median time of one compiled call (synchronized with block_until_ready);
  * the throughput in points/s and the effective bandwidth
    (minimum bytes read + written / time);
  * the cost estimated by XLA (flops and bytes) with cost_analysis().
We also measure the bandwidth of a "triad" z = a x + y on each device:
this is the ceiling of the roofline model for a memory-bound kernel.

The results are written to benchmarks/results/bench_<machine>.csv.
"""

import argparse
import csv
import os
import platform
import socket
import sys
import time
from functools import partial

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from j2gauss import Params, State, integrate  # noqa: E402  (turns on x64)
from j2gauss.j2_numpy import integrate_batch  # noqa: E402

PARAMS = Params(Q=150.0, b=50.0, C=5000.0)
WORDS_PER_POINT = 2 * (6 + 6 + 1 + 6)   # inputs (eps, eps_p, p, alpha) + outputs (sigma, eps_p, p, alpha)
DT = {"fp64": np.float64, "fp32": np.float32}


def make_data(N, dtype, seed=0):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 3e-3, (N, 6))
    eps_p = rng.normal(0.0, 5e-4, (N, 6))
    eps_p[:, :3] -= eps_p[:, :3].mean(axis=1, keepdims=True)       # zero trace
    p = rng.uniform(0.0, 0.02, N)
    alpha = rng.normal(0.0, 20.0, (N, 6))
    alpha[:, :3] -= alpha[:, :3].mean(axis=1, keepdims=True)
    return [a.astype(dtype) for a in (eps, eps_p, p, alpha)]


def timeit(fn, min_reps=3, min_total=0.3, max_reps=50):
    ts = []
    t_start = time.perf_counter()
    while len(ts) < min_reps or (sum(ts) < min_total and len(ts) < max_reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        if time.perf_counter() - t_start > 60:
            break
    return float(np.median(ts)), float(np.min(ts))


def build_kernel(n_newton):
    # parameters captured by closure: Python floats -> no fp32 -> fp64 promotion
    def kernel(eps, eps_p, p, alpha):
        f = lambda e, s: integrate(e, s, PARAMS, n_newton=n_newton)
        sig, st = jax.vmap(f)(eps, State(eps_p, p, alpha))
        return sig, st.eps_p, st.p, st.alpha
    return jax.jit(kernel)


def cost_per_point(compiled, N):
    try:
        ca = compiled.cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0]
        return ca.get("flops", np.nan) / N, ca.get("bytes accessed", np.nan) / N
    except Exception:
        return np.nan, np.nan


def triad_bandwidth(device, dtype, n=2**25):
    x = jax.device_put(jnp.ones(n, dtype), device)
    y = jax.device_put(jnp.ones(n, dtype), device)
    f = jax.jit(lambda a, b: 1.5 * a + b)
    f(x, y).block_until_ready()
    t, _ = timeit(lambda: f(x, y).block_until_ready(), min_reps=5)
    return 3 * n * np.dtype(dtype).itemsize / t / 1e9


def run(args):
    host = args.host or socket.gethostname()
    out_path = args.out or os.path.join(ROOT, "benchmarks", "results", f"bench_{host}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cpu = jax.devices("cpu")[0]
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        gpu = None
    if args.pretend_gpu and gpu is None:      # only to test the GPU code path on a CPU
        gpu = cpu
    configs = [c for c in args.configs if gpu is not None or not c.startswith("jax-gpu")]
    if len(configs) < len(args.configs):
        print("pas de GPU visible par JAX : configurations GPU ignorées")
    Ns = np.unique(np.round(10 ** np.arange(args.nmin, args.nmax + 1e-9, 1.0 / args.per_decade)).astype(int))
    kernel = build_kernel(args.n_newton)
    rows = []

    bw = {}
    for prec in args.precisions:
        bw[("cpu", prec)] = triad_bandwidth(cpu, DT[prec])
        if gpu is not None:
            bw[("gpu", prec)] = triad_bandwidth(gpu, DT[prec])
    print("bande passante triade (Go/s):", {k: round(v, 1) for k, v in bw.items()})

    for prec in args.precisions:
        dt = DT[prec]
        for N in Ns:
            host_data = make_data(int(N), dt)
            for cfg in configs:
                rec = dict(host=host, cpu=platform.processor() or platform.machine(),
                           device=(gpu.device_kind if cfg.startswith("jax-gpu") else "cpu"),
                           config=cfg, precision=prec, N=int(N), n_newton=args.n_newton,
                           compile_s=np.nan, flops_per_pt=np.nan, xla_bytes_per_pt=np.nan,
                           triad_GBps=bw[("gpu" if cfg.startswith("jax-gpu") else "cpu", prec)],
                           status="ok")
                try:
                    if cfg == "numpy":
                        fn = lambda: integrate_batch(*host_data, PARAMS, n_newton=args.n_newton)
                        out = fn()
                        plastic = float((out[2] > host_data[2]).mean())
                    else:
                        dev = gpu if cfg.startswith("jax-gpu") else cpu
                        dev_data = [jax.device_put(a, dev) for a in host_data]
                        t0 = time.perf_counter()
                        compiled = kernel.lower(*dev_data).compile()
                        rec["compile_s"] = time.perf_counter() - t0
                        rec["flops_per_pt"], rec["xla_bytes_per_pt"] = cost_per_point(compiled, N)
                        out = compiled(*dev_data)
                        jax.block_until_ready(out)
                        plastic = float((np.asarray(out[2]) > host_data[2]).mean())
                        if cfg == "jax-gpu-transfer":
                            def fn():
                                d = [jax.device_put(a, dev) for a in host_data]
                                o = compiled(*d)
                                return [np.asarray(x) for x in o]        # copy back to the host
                        else:
                            fn = lambda: jax.block_until_ready(compiled(*dev_data))
                    t_med, t_min = timeit(fn)
                    rec.update(time_median_s=t_med, time_min_s=t_min,
                               throughput_pts_s=N / t_med,
                               eff_GBps=N * WORDS_PER_POINT * np.dtype(dt).itemsize / t_med / 1e9,
                               plastic_fraction=plastic)
                except Exception as e:  # out of memory, etc.
                    msg = str(e).split("\n")[0][:80]
                    rec.update(time_median_s=np.nan, time_min_s=np.nan, throughput_pts_s=np.nan,
                               eff_GBps=np.nan, plastic_fraction=np.nan,
                               status="OOM" if "RESOURCE_EXHAUSTED" in str(e) or isinstance(e, MemoryError) else msg)
                rows.append(rec)
                print(f"{prec} N={N:>9d} {cfg:17s} t={rec['time_median_s']:.3e}s "
                      f"debit={rec['throughput_pts_s']:.3e} pts/s  compile={rec['compile_s']:.2f}s  {rec['status']}")
                if args.out_every:
                    _write(out_path, rows)
    _write(out_path, rows)
    print("résultats ->", out_path)


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmin", type=float, default=2.0, help="log10 of the smallest N")
    ap.add_argument("--nmax", type=float, default=7.0, help="log10 of the largest N")
    ap.add_argument("--per-decade", type=int, default=2)
    ap.add_argument("--precisions", nargs="+", default=["fp64", "fp32"])
    ap.add_argument("--configs", nargs="+",
                    default=["numpy", "jax-cpu", "jax-gpu", "jax-gpu-transfer"])
    ap.add_argument("--n-newton", type=int, default=4)
    ap.add_argument("--host", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pretend-gpu", action="store_true",
                    help="(test) use the CPU as a \"GPU\" to run the transfer code")
    ap.add_argument("--out-every", action="store_true", help="rewrite the CSV after each measurement")
    run(ap.parse_args())
