"""Vectorized NumPy reference (same algorithm as ``j2.integrate``, on batches (N, 6)).

Only used as a baseline in the benchmarks: no jit, no AD,
every operation creates a temporary array of size N.
"""

import numpy as np

_ONE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def integrate_batch(eps, eps_p, p, alpha, P, n_newton=6):
    dt = eps.dtype
    one = _ONE.astype(dt)
    mu, K = P.mu, P.K
    eps_e = eps - eps_p
    tr = eps_e[:, 0] + eps_e[:, 1] + eps_e[:, 2]
    s_tr = 2.0 * mu * (eps_e - (tr / 3.0)[:, None] * one)
    sigma_tr = (K * tr)[:, None] * one + s_tr
    xi = s_tr - alpha
    q_tr = np.sqrt(1.5 * np.einsum("ij,ij->i", xi, xi))

    def R(pp):
        return P.sigma_y + P.H * pp + P.Q * (1.0 - np.exp(-P.b * pp))

    def dR(pp):
        return P.H + P.Q * P.b * np.exp(-P.b * pp)

    R_n = R(p)
    plastic = q_tr > R_n
    q_s = np.where(plastic, q_tr, R_n)
    mu3C = 3.0 * mu + P.C
    dp = (q_s - R_n) / (mu3C + dR(p))
    for _ in range(n_newton):
        g = q_s - mu3C * dp - R(p + dp)
        dp = dp + g / (mu3C + dR(p + dp))
    dp = np.where(plastic, dp, 0.0).astype(dt)

    with np.errstate(divide="ignore", invalid="ignore"):
        N = np.where(q_tr[:, None] > 0, 1.5 * xi / q_tr[:, None], 0.0)
    deps_p = dp[:, None] * N
    sigma = sigma_tr - 2.0 * mu * deps_p
    return sigma, eps_p + deps_p, p + dp, alpha + (2.0 / 3.0) * P.C * deps_p
