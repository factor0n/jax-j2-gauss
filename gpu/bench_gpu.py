"""Benchmark of batched J2 integration: CPU vs GPU, with and without transfers.

Measured configurations (for each precision and each N):
  numpy          vectorized NumPy reference (CPU, one core)
  jax-cpu        JAX compiled on CPU (all cores)
  gpu-resident   JAX on GPU, data already in GPU memory: compute only
  gpu-transfer   JAX on GPU, with sending the inputs (PCIe), compute, and getting the outputs back,
                 done in blocks of --chunk points (needed with 4 GB of GPU memory);
                 we also measure the split: send / compute / return
  (option --tangent) jax-cpu+tangent and gpu+tangent: law + 6x6 tangent by AD, as
                 a FE code needs at each Newton iteration

Typical usage:
    python gpu/bench_gpu.py                       # full run (~10 min)
    python gpu/bench_gpu.py --quick               # quick (N <= 1e5, ~1 min)
    python gpu/bench_gpu.py --tangent             # also measure the tangent

Output: benchmarks/results/bench_gpu_<machine>.csv (rewritten after each measurement).
"""

import argparse
import csv
import os
import time

import _env  # noqa: F401  (settings before jax)
from _env import RESULTS, machine_info

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from j2gauss import Params, State, integrate  # noqa: E402
from j2gauss.j2_numpy import integrate_batch  # noqa: E402

PARAMS = Params(Q=150.0, b=50.0, C=5000.0)      # Voce + Prager (nonlinear hardening)
WORDS = {"law": 2 * 19, "tangent": 2 * 19 + 36}  # numbers read + written per point (minimum)
DTYPES = {"fp64": np.float64, "fp32": np.float32}


# ----------------------------------------------------------------------------- data
def make_data(N, dtype, seed=0):
    """Random Gauss points: ~99 % become plastic (the most ààexpensive case)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 3e-3, (N, 6))
    eps_p = rng.normal(0.0, 5e-4, (N, 6))
    eps_p[:, :3] -= eps_p[:, :3].mean(axis=1, keepdims=True)
    p = rng.uniform(0.0, 0.02, N)
    alpha = rng.normal(0.0, 20.0, (N, 6))
    alpha[:, :3] -= alpha[:, :3].mean(axis=1, keepdims=True)
    return [np.ascontiguousarray(a, dtype=dtype) for a in (eps, eps_p, p, alpha)]


# ----------------------------------------------------------------------------- kernels
def build_kernels(n_newton):
    def law(e, s):
        return integrate(e, s, PARAMS, n_newton=n_newton)

    def k_law(eps, eps_p, p, alpha):
        sig, st = jax.vmap(law)(eps, State(eps_p, p, alpha))
        return sig, st.eps_p, st.p, st.alpha

    def k_tangent(eps, eps_p, p, alpha):
        def one(e, s):
            (sig, st), tan = law(e, s), jax.jacfwd(lambda x: law(x, s)[0])(e)
            return sig, st, tan
        sig, st, C = jax.vmap(one)(eps, State(eps_p, p, alpha))
        return sig, st.eps_p, st.p, st.alpha, C

    return jax.jit(k_law), jax.jit(k_tangent)


# ----------------------------------------------------------------------------- helpers
def timeit(fn, min_reps=3, min_total=0.5, max_reps=20, max_time=60.0):
    ts, t_start = [], time.perf_counter()
    while len(ts) < min_reps or (sum(ts) < min_total and len(ts) < max_reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        if time.perf_counter() - t_start > max_time:
            break
    return float(np.median(ts)), float(np.min(ts))


def is_oom(e):
    s = str(e)
    return isinstance(e, MemoryError) or "RESOURCE_EXHAUSTED" in s or "out of memory" in s.lower()


def chunks(N, C):
    return [(s, min(s + C, N)) for s in range(0, N, C)]


def padded(arrs, s, e, C):
    """Block [s, e) padded with zeros up to C points: fixed shape, so no recompilation."""
    out = []
    for a in arrs:
        blk = a[s:e]
        if e - s < C:
            pad = np.zeros((C - (e - s),) + a.shape[1:], dtype=a.dtype)
            blk = np.concatenate([blk, pad])
        out.append(blk)
    return out


# ----------------------------------------------------------------------------- configurations
def run_transfer(compiled, host, gpu, N, C, outs):
    """Send in blocks + compute + return, with overlap: we collect block k-1
    while block k is running (JAX is asynchronous). The output arrays `outs`
    are allocated only once, outside the timing, as a FE code would do."""
    prev = None

    def collect(item):
        (s, e), o = item
        for dst, src in zip(outs, o):
            dst[s:e] = np.asarray(src)[: e - s]

    for s, e in chunks(N, C):
        d = [jax.device_put(a, gpu) for a in padded(host, s, e, C)]
        o = compiled(*d)
        if prev is not None:
            collect(prev)
        prev = ((s, e), o)
    collect(prev)
    return outs


def breakdown_transfer(compiled, host, gpu, N, C):
    """Same thing, but we synchronize each stage to measure send / compute / return."""
    t_h2d = t_cmp = t_d2h = 0.0
    for s, e in chunks(N, C):
        blk = padded(host, s, e, C)
        t0 = time.perf_counter()
        d = [jax.device_put(a, gpu) for a in blk]
        jax.block_until_ready(d)
        t1 = time.perf_counter()
        o = compiled(*d)
        jax.block_until_ready(o)
        t2 = time.perf_counter()
        [np.asarray(x) for x in o]
        t3 = time.perf_counter()
        t_h2d += t1 - t0
        t_cmp += t2 - t1
        t_d2h += t3 - t2
    return t_h2d, t_cmp, t_d2h


def check_correctness(k_law, cpu, gpu):
    """Compare GPU (FP64 and FP32) with NumPy FP64 on 20,000 points."""
    ref = integrate_batch(*make_data(20_000, np.float64), PARAMS, n_newton=8)[0]
    scale = np.abs(ref).max()
    out = {}
    for prec, dt in DTYPES.items():
        d = [jax.device_put(a, gpu) for a in make_data(20_000, dt)]
        sig = np.asarray(k_law(*d)[0], dtype=np.float64)
        out[prec] = float(np.abs(sig - ref).max() / scale)
    return out


# ----------------------------------------------------------------------------- main
def main(args):
    info = machine_info()
    host = args.host or info["host"]
    out_path = args.out or os.path.join(RESULTS, f"bench_gpu_{host}.csv")
    cpu = jax.devices("cpu")[0]
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        gpu = cpu if args.pretend_gpu else None
    if gpu is None:
        raise SystemExit("Aucun GPU visible par JAX (voir gpu/README_GPU.md, section Dépannage). "
                         "Pour tester le script sans GPU : --pretend-gpu")
    fake = gpu is cpu
    print(f"JAX {jax.__version__} | CPU : {info['cpu_model']} ({info['cpu_count']} threads) | "
          f"GPU : {info.get('gpu_name', 'simulé (CPU)' if fake else gpu.device_kind)} | {info['alimentation']}")

    k_law, k_tan = build_kernels(args.n_newton)
    err = check_correctness(k_law, cpu, gpu)
    print(f"Vérification GPU vs NumPy (écart relatif max) : FP64 {err['fp64']:.1e}, FP32 {err['fp32']:.1e}")
    if err["fp64"] > 1e-9:
        print("  ATTENTION : écart FP64 anormal, résultats GPU suspects.")

    configs = list(args.configs)
    if args.tangent:
        configs += ["jax-cpu+tangent", "gpu+tangent"]
    Ns = np.unique(np.round(10 ** np.arange(args.nmin, args.nmax + 1e-9, 1.0 / args.per_decade)).astype(int))
    rows = []

    for prec in args.precisions:
        dt = DTYPES[prec]
        for N in Ns:
            N = int(N)
            host_data = make_data(N, dt)
            for cfg in configs:
                if cfg == "numpy" and N > args.numpy_max:
                    continue
                tangent = cfg.endswith("+tangent")
                kernel = k_tan if tangent else k_law
                words = WORDS["tangent" if tangent else "law"]
                dev = cpu if cfg.startswith("jax-cpu") else gpu
                rec = dict(host=host, cpu=info["cpu_model"], gpu=info.get("gpu_name", ""),
                           simulated_gpu=fake, config=cfg, precision=prec, N=N, n_newton=args.n_newton,
                           chunk=(args.chunk if cfg == "gpu-transfer" else ""), compile_s="",
                           time_median_s="", time_min_s="", throughput_pts_s="", eff_GBps="",
                           t_h2d_s="", t_compute_s="", t_d2h_s="", status="ok")
                try:
                    if cfg == "numpy":
                        fn = lambda: integrate_batch(*host_data, PARAMS, n_newton=args.n_newton)
                    elif cfg == "gpu-transfer":
                        C = min(args.chunk, N)
                        blk = [jax.device_put(a, gpu) for a in padded(host_data, 0, C, C)]
                        t0 = time.perf_counter()
                        compiled = kernel.lower(*blk).compile()
                        rec["compile_s"] = time.perf_counter() - t0
                        del blk
                        outs = [np.zeros_like(a) for a in host_data]   # allocated and "touched" once
                        fn = lambda: run_transfer(compiled, host_data, gpu, N, C, outs)
                        fn()
                        h, c, d = breakdown_transfer(compiled, host_data, gpu, N, C)
                        rec.update(t_h2d_s=h, t_compute_s=c, t_d2h_s=d)
                    else:
                        dev_data = [jax.device_put(a, dev) for a in host_data]
                        t0 = time.perf_counter()
                        compiled = kernel.lower(*dev_data).compile()
                        rec["compile_s"] = time.perf_counter() - t0
                        jax.block_until_ready(compiled(*dev_data))
                        fn = lambda: jax.block_until_ready(compiled(*dev_data))
                    t_med, t_min = timeit(fn)
                    rec.update(time_median_s=t_med, time_min_s=t_min, throughput_pts_s=N / t_med,
                               eff_GBps=N * words * np.dtype(dt).itemsize / t_med / 1e9)
                except Exception as e:  # noqa: BLE001
                    rec["status"] = "OOM" if is_oom(e) else str(e).split("\n")[0][:100]
                finally:
                    dev_data = compiled = fn = outs = None   # free GPU memory before the next measurement
                rows.append(rec)
                thr = rec["throughput_pts_s"]
                extra = ""
                if rec["t_h2d_s"] != "":
                    tot = rec["t_h2d_s"] + rec["t_compute_s"] + rec["t_d2h_s"]
                    extra = (f" [envoi {100*rec['t_h2d_s']/tot:.0f}% | calcul {100*rec['t_compute_s']/tot:.0f}%"
                             f" | retour {100*rec['t_d2h_s']/tot:.0f}%]")
                print(f"{prec} N={N:>9,d} {cfg:16s} " +
                      (f"{thr:10.3e} pts/s" if thr != "" else f"{rec['status']:>14s}") + extra)
                write_csv(out_path, rows, info, err)
    print("\nRésultats ->", out_path)


def write_csv(path, rows, info, err):
    with open(path, "w", newline="") as f:
        f.write(f"# machine: {info}\n# verification_ecart_relatif: {err}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nmin", type=float, default=2.0, help="log10 of the smallest N (default 2: 100 points)")
    ap.add_argument("--nmax", type=float, default=7.0, help="log10 of the largest N (default 7: 10 million)")
    ap.add_argument("--per-decade", type=int, default=2)
    ap.add_argument("--precisions", nargs="+", default=["fp64", "fp32"], choices=["fp64", "fp32"])
    ap.add_argument("--configs", nargs="+", default=["numpy", "jax-cpu", "gpu-resident", "gpu-transfer"])
    ap.add_argument("--tangent", action="store_true", help="also measure law + 6x6 tangent")
    ap.add_argument("--chunk", type=int, default=2**20, help="block size for gpu-transfer")
    ap.add_argument("--numpy-max", type=float, default=3e6, help="largest N for NumPy (host memory)")
    ap.add_argument("--n-newton", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="quick try: N up to 1e5")
    ap.add_argument("--pretend-gpu", action="store_true", help="(test without GPU) use the CPU instead")
    ap.add_argument("--host", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.quick:
        a.nmax = min(a.nmax, 5.0)
    main(a)
