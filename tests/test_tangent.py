"""Tangent: AD (jacfwd) == analytic (Simo-Hughes) == finite differences."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from j2gauss import (Params, initial_state, integrate, tangent_ad, tangent_analytic,
                     tangent_fd, tangent_elastic, to_mandel)

MATERIALS = {
    "linear": Params(),
    "voce": Params(Q=150.0, b=50.0),
    "prager": Params(H=0.0, C=8_000.0),
    "combined": Params(Q=150.0, b=50.0, C=5_000.0),
}

STRAINS = {
    "elastic": jnp.array([5e-4, -1e-4, -1e-4, 1e-4, 0.0, 2e-4]),
    "plastic_tension": jnp.array([4e-3, -1.3e-3, -1.1e-3, 4e-4, 0.0, 1.2e-3]),
    "plastic_shear": jnp.array([0.0, 0.0, 0.0, 3e-3, 2e-3, 4e-3]),
}


def preloaded_state(P):
    """A state with non-zero eps_p, p and alpha (loading, then start of unloading)."""
    _, st = integrate(jnp.array([6e-3, -2e-3, -2e-3, 0.0, 1e-3, 0.0]), initial_state(), P)
    return st


def rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


@pytest.mark.parametrize("mat", MATERIALS)
@pytest.mark.parametrize("strain", STRAINS)
@pytest.mark.parametrize("preload", [False, True])
def test_ad_equals_analytic(mat, strain, preload):
    P = MATERIALS[mat]
    st = preloaded_state(P) if preload else initial_state()
    eps = STRAINS[strain] + (st.eps_p if preload else 0.0)
    C_ad = tangent_ad(eps, st, P)
    C_an = tangent_analytic(eps, st, P)
    assert rel(C_ad, C_an) < 1e-12


@pytest.mark.parametrize("mat", MATERIALS)
@pytest.mark.parametrize("strain", STRAINS)
def test_fd_equals_analytic(mat, strain):
    P = MATERIALS[mat]
    st = initial_state()
    C_fd = tangent_fd(STRAINS[strain], st, P)
    C_an = tangent_analytic(STRAINS[strain], st, P)
    assert rel(C_fd, C_an) < 1e-7


@pytest.mark.parametrize("mat", ["voce", "combined"])
def test_implicit_equals_unrolled(mat):
    P = MATERIALS[mat]
    eps = STRAINS["plastic_tension"]
    C_imp = tangent_ad(eps, initial_state(), P, implicit=True)
    C_unr = tangent_ad(eps, initial_state(), P, implicit=False)
    assert rel(C_imp, C_unr) < 1e-12


@pytest.mark.parametrize("mat", MATERIALS)
def test_tangent_symmetric(mat):
    """Associated flow -> symmetric consistent tangent (major symmetry)."""
    C = tangent_ad(STRAINS["plastic_tension"], initial_state(), MATERIALS[mat])
    assert rel(C, C.T) < 1e-12


def test_elastic_step_gives_elastic_tangent():
    P = MATERIALS["combined"]
    C = tangent_ad(STRAINS["elastic"], initial_state(), P)
    assert rel(C, tangent_elastic(P)) < 1e-14


def test_plastic_tangent_is_softer():
    """During plastic loading, the stiffness in the flow direction drops."""
    P = MATERIALS["linear"]
    eps = STRAINS["plastic_tension"]
    Ce, Cep = tangent_elastic(P), tangent_analytic(eps, initial_state(), P)
    s = integrate(eps, initial_state(), P)[0]
    n = s - s[:3].mean() * jnp.array([1, 1, 1, 0, 0, 0.0])
    n = n / jnp.linalg.norm(n)
    assert float(n @ Cep @ n) < 0.05 * float(n @ Ce @ n)


def test_3x3_derivative_pitfall():
    """Differentiating with respect to a raw 3x3 matrix breaks the minor symmetries.

    sigma(eps) = lam tr(eps) I + 2 mu eps: jacfwd gives 2 mu d_ik d_jl, not
    2 mu (d_ik d_jl + d_il d_jk)/2. If we first read sym(eps) (to_mandel), it is correct.
    """
    lam, mu = 1.0, 1.0
    I = jnp.eye(3)
    raw = lambda e: lam * jnp.trace(e) * I + 2 * mu * e
    sym = lambda e: raw(0.5 * (e + e.T))
    e0 = jnp.zeros((3, 3))
    A_raw = jax.jacfwd(raw)(e0)
    A_sym = jax.jacfwd(sym)(e0)
    # component (0,1,0,1) vs (0,1,1,0): they must be equal for a tensor with minor symmetry
    assert abs(A_raw[0, 1, 0, 1] - A_raw[0, 1, 1, 0]) > 0.9
    assert abs(A_sym[0, 1, 0, 1] - A_sym[0, 1, 1, 0]) < 1e-15
    # and the Mandel path gives the right 6x6 matrix (identity for mu = 1/2, lam = 0)
    P = Params(E=1.0, nu=0.0)
    C6 = jax.jacfwd(lambda e: integrate(to_mandel(e), initial_state(), P)[0])(e0)
    assert C6.shape == (6, 3, 3)
