"""Physical behavior: yield limit, uniaxial analytic solutions, invariants."""

import jax.numpy as jnp
import numpy as np
import pytest

from j2gauss import Params, initial_state, integrate, tangent_elastic, dev
from j2gauss.driver import run_mixed, UNIAXIAL
from j2gauss.j2 import hardening


def test_small_strain_is_elastic():
    P = Params(Q=150.0, b=50.0, C=5_000.0)
    eps = jnp.array([2e-4, -5e-5, 1e-5, 3e-5, -2e-5, 1e-4])
    sig, st = integrate(eps, initial_state(), P)
    np.testing.assert_allclose(sig, tangent_elastic(P) @ eps, rtol=1e-14, atol=1e-12)
    assert float(st.p) == 0.0 and float(jnp.abs(st.eps_p).max()) == 0.0


@pytest.mark.parametrize("P", [Params(), Params(H=0.0, C=4_000.0), Params(H=800.0, C=2_000.0)])
def test_uniaxial_bilinear_analytic(P):
    """Uniaxial tension, total linear hardening h = H + C:
    sigma = E eps while sigma < sigma_y, then sigma = sigma_y + E_t (eps - eps_y),
    E_t = E h / (E + h)."""
    e11 = np.linspace(0.0, 0.01, 41)[1:]
    path = np.zeros((e11.size, 6)); path[:, 0] = e11
    res = run_mixed(path, UNIAXIAL, P)
    h = P.H + P.C
    eps_y = P.sigma_y / P.E
    Et = P.E * h / (P.E + h)
    expected = np.where(e11 <= eps_y, P.E * e11, P.sigma_y + Et * (e11 - eps_y))
    np.testing.assert_allclose(res["sigma"][:, 0], expected, rtol=1e-10)
    # lateral stresses are zero
    assert np.abs(res["sigma"][:, 1:]).max() < 1e-8


def test_poisson_ratio_elastic():
    P = Params()
    path = np.zeros((1, 6)); path[0, 0] = 5e-4
    res = run_mixed(path, UNIAXIAL, P)
    assert abs(res["eps"][0, 1] / res["eps"][0, 0] + P.nu) < 1e-12


def test_yield_onset():
    """First plastic step exactly when sigma_11 goes above sigma_y."""
    P = Params()
    eps_y = P.sigma_y / P.E
    for f, plastic in [(0.999, False), (1.001, True)]:
        path = np.zeros((1, 6)); path[0, 0] = f * eps_y
        res = run_mixed(path, UNIAXIAL, P)
        assert (res["p"][0] > 0) == plastic


def test_plastic_incompressibility_and_consistency():
    P = Params(Q=150.0, b=50.0, C=5_000.0)
    eps = jnp.array([6e-3, -1e-3, -2e-3, 1e-3, 5e-4, -2e-3])
    sig, st = integrate(eps, initial_state(), P)
    assert abs(float(st.eps_p[:3].sum())) < 1e-16           # tr(eps_p) = 0
    assert abs(float(st.alpha[:3].sum())) < 1e-12           # alpha is deviatoric
    xi = dev(sig) - st.alpha
    q = float(jnp.sqrt(1.5 * xi @ xi))
    assert abs(q - float(hardening(st.p, P))) < 1e-9        # exact return to the yield surface


def test_elastic_unloading_after_plasticity():
    """Unloading from a plastic state: elastic slope E."""
    P = Params()
    e11 = np.array([0.004, 0.0039, 0.0038])
    path = np.zeros((3, 6)); path[:, 0] = e11
    res = run_mixed(path, UNIAXIAL, P)
    slope = (res["sigma"][2, 0] - res["sigma"][1, 0]) / (e11[2] - e11[1])
    assert abs(slope / P.E - 1.0) < 1e-10
    assert res["p"][1] == res["p"][2]
