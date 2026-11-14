"""Hyperelasticity: stress and tangent by automatic differentiation.

Part (a): compressible Neo-Hooke potential psi(F); P = d psi / dF with
``jax.grad`` and A = d^2 psi / dF dF with ``jax.hessian``, compared with the
analytic formulas.

Part (b): incompressible material described by psi(I1, I2). For the three
homogeneous Treloar tests (uniaxial tension UT, equibiaxial tension ET, pure shear PS)
the nominal stress (1st Piola-Kirchhoff) has a closed form in terms of
psi_1 = d psi / dI1 and psi_2 = d psi / dI2, obtained with ``jax.grad``.
The potential can be analytic (Neo-Hooke, Mooney-Rivlin, Yeoh) or an
input convex neural network (ICNN), which guarantees polyconvexity.
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# (a) Compressible Neo-Hooke
# ---------------------------------------------------------------------------

def psi_neo_hooke(F, mu, lam):
    """psi = mu/2 (I1 - 3 - 2 ln J) + lam/2 (ln J)^2,  I1 = tr(F^T F), J = det F."""
    I1 = jnp.sum(F * F)
    lnJ = jnp.log(jnp.linalg.det(F))
    return 0.5 * mu * (I1 - 3.0 - 2.0 * lnJ) + 0.5 * lam * lnJ**2


def piola_neo_hooke_analytic(F, mu, lam):
    """P = mu (F - F^{-T}) + lam ln J F^{-T}."""
    FinvT = jnp.linalg.inv(F).T
    lnJ = jnp.log(jnp.linalg.det(F))
    return mu * (F - FinvT) + lam * lnJ * FinvT


def tangent_neo_hooke_analytic(F, mu, lam):
    """A_iJkL = mu d_ik d_JL + (mu - lam ln J) Fi_Li Fi_Jk + lam Fi_Ji Fi_Lk,  Fi = F^{-1}."""
    Fi = jnp.linalg.inv(F)
    lnJ = jnp.log(jnp.linalg.det(F))
    d = jnp.eye(3)
    return (mu * jnp.einsum("ik,jl->ijkl", d, d)
            + (mu - lam * lnJ) * jnp.einsum("li,jk->ijkl", Fi, Fi)
            + lam * jnp.einsum("ji,lk->ijkl", Fi, Fi))


def piola_ad(psi, F, *args):
    return jax.grad(psi)(F, *args)


def tangent_ad(psi, F, *args):
    return jax.hessian(psi)(F, *args)


# ---------------------------------------------------------------------------
# (b) Incompressible, homogeneous tests
# ---------------------------------------------------------------------------

def invariants(test, lam):
    """(I1, I2) for a principal stretch lam, for the given test (incompressible, J = 1)."""
    if test == "UT":   # F = diag(l, l^-1/2, l^-1/2)
        return lam**2 + 2.0 / lam, 2.0 * lam + lam**-2
    if test == "ET":   # F = diag(l, l, l^-2)
        return 2.0 * lam**2 + lam**-4, lam**4 + 2.0 * lam**-2
    if test == "PS":   # F = diag(l, 1, l^-1)
        v = lam**2 + 1.0 + lam**-2
        return v, v
    raise ValueError(test)


def nominal_stress(psi_inv, theta, test, lam):
    """Nominal stress P11 (the pressure is removed using the free-surface condition).

    UT: P = 2 (l - l^-2) (psi1 + psi2 / l)
    ET: P = 2 (l - l^-5) (psi1 + l^2 psi2)
    PS: P = 2 (l - l^-3) (psi1 + psi2)
    """
    I1, I2 = invariants(test, lam)
    g = jax.vmap(jax.grad(psi_inv, argnums=(1, 2)), in_axes=(None, 0, 0))
    psi1, psi2 = g(theta, I1, I2)
    if test == "UT":
        return 2.0 * (lam - lam**-2) * (psi1 + psi2 / lam)
    if test == "ET":
        return 2.0 * (lam - lam**-5) * (psi1 + lam**2 * psi2)
    return 2.0 * (lam - lam**-3) * (psi1 + psi2)


# Analytic potentials psi(theta, I1, I2) (parameters in MPa)
def psi_nh(theta, I1, I2):
    return theta["C10"] * (I1 - 3.0)


def psi_mr(theta, I1, I2):
    return theta["C10"] * (I1 - 3.0) + theta["C01"] * (I2 - 3.0)


def psi_yeoh(theta, I1, I2):
    x = I1 - 3.0
    return theta["C10"] * x + theta["C20"] * x**2 + theta["C30"] * x**3


# ---------------------------------------------------------------------------
# Input convex neural network (ICNN) psi_theta(I1, I2)
# ---------------------------------------------------------------------------
_pos = jax.nn.softplus      # reparametrization: weights >= 0


def icnn_init(key, width=12, scale=-3.0):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = lambda k, s: scale + 0.5 * jax.random.normal(k, s)
    return {
        "W1": n(k1, (width, 2)), "b1": jnp.zeros(width),
        "W2": n(k2, (width, width)), "U2": n(k3, (width, 2)), "b2": jnp.zeros(width),
        "a": n(k4, (width,)), "c": jnp.full(2, scale),
    }


def _icnn_raw(theta, x):
    """x = (I1 - 3, I2 - 3) / 10. Layers with positive weights and convex,
    increasing activations (softplus, softplus^2): the output is convex and increasing in x."""
    h1 = jax.nn.softplus(_pos(theta["W1"]) @ x + theta["b1"])
    z2 = _pos(theta["W2"]) @ h1 + _pos(theta["U2"]) @ x + theta["b2"]
    half = z2.shape[0] // 2
    h2 = jnp.concatenate([jax.nn.softplus(z2[:half]), jax.nn.softplus(z2[half:]) ** 2])
    return _pos(theta["a"]) @ h2 + _pos(theta["c"]) @ x


def psi_icnn(theta, I1, I2):
    """psi(I1, I2) - psi(3, 3): zero energy in the reference configuration."""
    x = jnp.stack([I1 - 3.0, I2 - 3.0]) / 10.0
    return _icnn_raw(theta, x) - _icnn_raw(theta, jnp.zeros(2))
