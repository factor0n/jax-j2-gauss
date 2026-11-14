"""Hyperelasticity: grad/hessian against analytic formulas, physical properties."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from j2gauss.hyperelastic import (psi_neo_hooke, piola_neo_hooke_analytic,
                                  tangent_neo_hooke_analytic, piola_ad, tangent_ad,
                                  invariants, nominal_stress, psi_icnn, icnn_init, psi_mr)

MU, LAM = 0.8, 4.0


def random_F(seed, amp=0.3):
    rng = np.random.default_rng(seed)
    F = np.eye(3) + amp * rng.normal(size=(3, 3))
    if np.linalg.det(F) < 0:
        F[:, 0] *= -1
    return jnp.asarray(F)


@pytest.mark.parametrize("seed", range(5))
def test_piola_and_tangent(seed):
    F = random_F(seed)
    np.testing.assert_allclose(piola_ad(psi_neo_hooke, F, MU, LAM),
                               piola_neo_hooke_analytic(F, MU, LAM), atol=1e-13)
    np.testing.assert_allclose(tangent_ad(psi_neo_hooke, F, MU, LAM),
                               tangent_neo_hooke_analytic(F, MU, LAM), atol=1e-12)


def test_tangent_major_symmetry_and_stress_free_reference():
    F = random_F(7)
    A = tangent_ad(psi_neo_hooke, F, MU, LAM)
    np.testing.assert_allclose(A, jnp.transpose(A, (2, 3, 0, 1)), atol=1e-13)
    assert float(jnp.abs(piola_ad(psi_neo_hooke, jnp.eye(3), MU, LAM)).max()) < 1e-15


def test_objectivity():
    """P(QF) = Q P(F) for any rotation Q."""
    F = random_F(3)
    Q, _ = jnp.linalg.qr(random_F(11, 1.0))
    Q = Q * jnp.sign(jnp.linalg.det(Q))
    np.testing.assert_allclose(piola_ad(psi_neo_hooke, Q @ F, MU, LAM),
                               Q @ piola_ad(psi_neo_hooke, F, MU, LAM), atol=1e-12)


@pytest.mark.parametrize("test,Fdiag", [("UT", lambda l: [l, l**-0.5, l**-0.5]),
                                        ("ET", lambda l: [l, l, l**-2]),
                                        ("PS", lambda l: [l, 1.0, 1.0 / l])])
def test_invariants_and_stress_free(test, Fdiag):
    l = 1.7
    C = jnp.diag(jnp.array(Fdiag(l)) ** 2)
    I1, I2 = invariants(test, l)
    assert abs(float(jnp.trace(C)) - I1) < 1e-12
    assert abs(0.5 * float(jnp.trace(C) ** 2 - jnp.trace(C @ C)) - I2) < 1e-12
    th = {"C10": 0.2, "C01": 0.05}
    assert abs(float(nominal_stress(psi_mr, th, test, jnp.array([1.0]))[0])) < 1e-15


def test_uniaxial_mooney_rivlin_closed_form():
    l = jnp.linspace(1.0, 3.0, 9)
    th = {"C10": 0.2, "C01": 0.05}
    P = nominal_stress(psi_mr, th, "UT", l)
    np.testing.assert_allclose(P, 2 * (l - l**-2) * (th["C10"] + th["C01"] / l), rtol=1e-13)


def test_icnn_convex_and_monotone():
    th = icnn_init(jax.random.PRNGKey(0), scale=-0.5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        I1, I2 = 3.0 + 40 * rng.random(), 3.0 + 40 * rng.random()
        g = jax.grad(psi_icnn, argnums=(1, 2))(th, I1, I2)
        H = jax.hessian(lambda x: psi_icnn(th, x[0], x[1]))(jnp.array([I1, I2]))
        assert float(g[0]) >= 0 and float(g[1]) >= 0
        assert float(jnp.linalg.eigvalsh(H).min()) > -1e-12
    assert abs(float(psi_icnn(th, 3.0, 3.0))) < 1e-15
