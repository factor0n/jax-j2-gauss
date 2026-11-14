# Résultats GPU - machine `izanami-LOQ-15IAX9`

- machine: {'host': 'izanami-LOQ-15IAX9', 'os': 'Linux-7.0.0-42-generic-x86_64-with-glibc2.39', 'python': '3.12.3', 'cpu_count': 12, 'cpu_model': '12th Gen Intel(R) Core(TM) i5-12450HX', 'gpu_name': 'NVIDIA GeForce RTX 2050', 'driver': '595.99.02', 'vram': '4096 MiB', 'sm_clock_max': '2100 MHz', 'power_limit': '[N/A]', 'alimentation': 'inconnue'}
- verification_ecart_relatif: {'fp64': 1.1429367273915581e-16, 'fp32': 1.3461415568727652e-07}

## Plafonds mesurés (check_gpu.py)

- PCIe hôte -> GPU : 4.86 Go/s ; GPU -> hôte : 2.20 Go/s
- triade GPU FP64 : 91 Go/s ; triade CPU FP64 : 21.4 Go/s
- produit de matrices GPU : FP32 5475 GFLOP/s, FP64 106 GFLOP/s (rapport 1/52)
- latence d'un appel : 31 µs

## Meilleur débit par configuration (points/s)

| configuration | précision | N | débit |
|---|---|---|---|
| gpu-resident | fp32 | 3,162,278 | 1.79e+08 |
| gpu-resident | fp64 | 31,623 | 5.03e+07 |
| gpu-transfer | fp32 | 10,000,000 | 2.43e+07 |
| gpu-transfer | fp64 | 316,228 | 9.28e+06 |
| jax-cpu | fp32 | 100,000 | 6.27e+07 |
| jax-cpu | fp64 | 10,000 | 2.45e+07 |
| numpy | fp32 | 10,000 | 1.52e+07 |
| numpy | fp64 | 3,162 | 7.59e+06 |

## Accélération GPU / JAX CPU

| précision | configuration | accélération max | GPU rentable dès N = |
|---|---|---|---|
| fp64 | gpu-resident | 2.92 | 3162 |
| fp64 | gpu-transfer | 0.43 | jamais |
| fp32 | gpu-resident | 4.61 | 10000 |
| fp32 | gpu-transfer | 0.56 | jamais |

## Décomposition du temps (gpu-transfer, plus grand N)

| précision | N | envoi | calcul | retour |
|---|---|---|---|---|
| fp64 | 10,000,000 | 25 % | 11 % | 64 % |
| fp32 | 10,000,000 | 36 % | 13 % | 51 % |

## Mémoire GPU insuffisante

- gpu-resident fp64 N=10,000,000
