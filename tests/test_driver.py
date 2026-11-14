"""Convergence of the global Newton at the material point."""

import numpy as np

from j2gauss import Params
from j2gauss.driver import run_mixed, UNIAXIAL, TENSION_SHEAR


def _path(e11, e12=None):
    path = np.zeros((len(e11), 6))
    path[:, 0] = e11
    if e12 is not None:
        path[:, 5] = np.sqrt(2.0) * np.asarray(e12)
    return path


def test_linear_proportional_converges_in_one_iteration():
    """Linear hardening + proportional loading: sigma_22(eps_22) is piecewise linear."""
    res = run_mixed(_path([0.004, 0.008]), UNIAXIAL, Params())
    assert all(len(h) - 1 == 1 for h in res["residuals"])


def test_voce_quadratic_convergence():
    res = run_mixed(_path([0.004, 0.008]), UNIAXIAL, Params(Q=150.0, b=50.0))
    r = np.array(res["residuals"][1])
    assert len(r) - 1 <= 5
    # measured order ~2 on the "clean" iterations: log r_{k+1} / log r_k
    k = np.argmax(r < 1.0)
    order = np.log(r[k + 1] / r[k]) / np.log(r[k] / r[k - 1])
    assert order > 1.6


def test_elastic_tangent_linear_convergence():
    res = run_mixed(_path([0.004, 0.008]), UNIAXIAL, Params(Q=150.0, b=50.0), tangent="elastic")
    r = np.array(res["residuals"][1])
    ratios = r[1:] / r[:-1]
    assert len(r) > 10
    assert np.std(ratios[2:-1]) < 0.05 and ratios[3] < 0.5     # constant rate < 1


def test_tension_shear_quadratic_even_with_linear_hardening():
    res = run_mixed(_path([0.004, 0.008], [0.003, 0.006]), TENSION_SHEAR, Params())
    assert all(2 <= len(h) - 1 <= 5 for h in res["residuals"])
