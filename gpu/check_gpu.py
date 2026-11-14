"""Machine check and micro-benchmarks: these are the "ceilings" of the roofline model.

Measurements:
  * latency of a tiny JAX call (fixed launch cost);
  * PCIe bus bandwidth: host -> GPU and GPU -> host;
  * memory bandwidth (triad z = a x + y) on CPU and GPU, in FP32 and FP64;
  * compute throughput (matrix product) on GPU in FP32 and FP64 -> FP64/FP32 ratio.

Usage:  python gpu/check_gpu.py            (output: benchmarks/results/gpu_check_<machine>.json)
"""

import argparse
import json
import os
import time

import _env  # noqa: F401  (settings before jax)
from _env import RESULTS, machine_info

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def timeit(fn, reps=7):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main(args):
    info = machine_info()
    cpu = jax.devices("cpu")[0]
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        gpu = cpu if args.pretend_gpu else None
    print(f"JAX {jax.__version__} | appareils : {jax.devices()}")
    for k in ("gpu_name", "driver", "vram", "cpu_model", "cpu_count", "alimentation"):
        print(f"  {k:13s}: {info.get(k, '-')}")
    if gpu is None:
        raise SystemExit("\nAucun GPU visible par JAX. Voir « Dépannage » dans gpu/README_GPU.md.")
    if info.get("alimentation") == "batterie":
        print("\n  ATTENTION : le portable est sur batterie -> branche le secteur avant de mesurer.")

    res = {"machine": info, "jax": jax.__version__, "pretend_gpu": bool(args.pretend_gpu and gpu is cpu)}
    size = args.mb * 2**20

    # launch latency
    f = jax.jit(lambda x: x + 1.0)
    x1 = jax.device_put(jnp.ones(8), gpu)
    res["launch_latency_us"] = 1e6 * timeit(lambda: f(x1).block_until_ready(), reps=200)

    # PCIe
    host = np.ones(size // 8, dtype=np.float64)
    res["h2d_GBps"] = size / timeit(lambda: jax.device_put(host, gpu).block_until_ready()) / 1e9
    dev = jax.device_put(host, gpu)
    bump = jax.jit(lambda x: x + 1.0)
    ts = []
    for _ in range(7):
        y = bump(dev).block_until_ready()   # new array: JAX has not copied it to the host yet
        t0 = time.perf_counter()
        np.asarray(y)
        ts.append(time.perf_counter() - t0)
    res["d2h_GBps"] = size / float(np.median(ts[1:])) / 1e9

    # triad
    tri = jax.jit(lambda a, b: 1.5 * a + b)
    for name, d in (("cpu", cpu), ("gpu", gpu)):
        for prec, dt in (("fp32", jnp.float32), ("fp64", jnp.float64)):
            n = size // np.dtype(dt).itemsize
            a = jax.device_put(jnp.ones(n, dt), d)
            b = jax.device_put(jnp.ones(n, dt), d)
            t = timeit(lambda: tri(a, b).block_until_ready())
            res[f"triad_{name}_{prec}_GBps"] = 3 * size / t / 1e9

    # matrix product on GPU
    mm = jax.jit(lambda a, b: a @ b)
    for prec, dt, n in (("fp32", jnp.float32, args.mat), ("fp64", jnp.float64, args.mat)):
        a = jax.device_put(jnp.ones((n, n), dt), gpu)
        t = timeit(lambda: mm(a, a).block_until_ready(), reps=5)
        res[f"matmul_gpu_{prec}_GFLOPs"] = 2 * n**3 / t / 1e9
    res["fp64_over_fp32"] = res["matmul_gpu_fp64_GFLOPs"] / res["matmul_gpu_fp32_GFLOPs"]

    print("\nRésultats :")
    print(f"  latence d'un appel            : {res['launch_latency_us']:8.1f} µs")
    print(f"  PCIe hôte -> GPU              : {res['h2d_GBps']:8.2f} Go/s")
    print(f"  PCIe GPU -> hôte              : {res['d2h_GBps']:8.2f} Go/s")
    for name in ("cpu", "gpu"):
        print(f"  triade {name.upper()} FP32 / FP64        : {res[f'triad_{name}_fp32_GBps']:8.1f} / "
              f"{res[f'triad_{name}_fp64_GBps']:.1f} Go/s")
    print(f"  produit de matrices GPU FP32  : {res['matmul_gpu_fp32_GFLOPs']:8.0f} GFLOP/s")
    print(f"  produit de matrices GPU FP64  : {res['matmul_gpu_fp64_GFLOPs']:8.0f} GFLOP/s")
    print(f"  rapport FP64 / FP32           : 1/{1 / res['fp64_over_fp32']:.0f}")

    out = args.out or os.path.join(RESULTS, f"gpu_check_{info['host']}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n-> ", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=256, help="size of the test arrays (MB)")
    ap.add_argument("--mat", type=int, default=2048, help="matrix size for the compute throughput")
    ap.add_argument("--pretend-gpu", action="store_true", help="(test without GPU) use the CPU instead")
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
