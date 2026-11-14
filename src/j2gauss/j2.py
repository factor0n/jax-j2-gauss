"""J2 (von Mises) elastoplastic law in small strains, integrated by radial return.

Hardening:
  * isotropic, R(p) = sigma_y + H p + Q (1 - exp(-b p))   (linear if Q = 0, Voce if Q > 0);
  * linear kinematic (Prager), d alpha = (2/3) C d eps_p   (turned off if C = 0).

All tensor quantities are Mandel vectors of size 6.

Interface (the same as a material law in a finite element code):

    sigma_new, state_new = integrate(eps_new, state_old, params)

Tricky points with JAX (see the report, chapter 6):
  1. no Python ``if`` on traced values: we use ``jnp.where``;
  2. ``jnp.where`` computes and differentiates BOTH branches: we must protect
     sqrt(0) and 0/0 with a "double where", otherwise the tangent has NaNs;
  3. the scalar equation for Delta p is solved by Newton; we can differentiate
     through the iterations ("unrolled" mode) or use the implicit function
     theorem ("implicit" mode, with ``jax.custom_jvp``).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .mandel import ONE


class Params(NamedTuple):
    """Material parameters (consistent units, here MPa)."""
    E: float = 210_000.0      # Young's modulus
    nu: float = 0.3           # Poisson's ratio
    sigma_y: float = 250.0    # initial yield stress
    H: float = 1_000.0        # linear isotropic hardening modulus
    Q: float = 0.0            # Voce saturation (0 -> linear hardening)
    b: float = 0.0            # Voce saturation rate
    C: float = 0.0            # Prager kinematic modulus (0 -> no kinematic)

    @property
    def mu(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def K(self):
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))


class State(NamedTuple):
    """Internal variables at one Gauss point."""
    eps_p: jnp.ndarray   # plastic strain (Mandel, 6)
    p: jnp.ndarray       # accumulated plastic strain (scalar)
    alpha: jnp.ndarray   # back-stress (Mandel, 6, deviatoric)


def initial_state(dtype=jnp.float64, shape=()):
    z6 = jnp.zeros(shape + (6,), dtype=dtype)
    return State(eps_p=z6, p=jnp.zeros(shape, dtype=dtype), alpha=z6)


# ---------------------------------------------------------------------------
# Isotropic hardening
# ---------------------------------------------------------------------------
def hardening(p, P: Params):
    """R(p): current radius of the elastic domain."""
    return P.sigma_y + P.H * p + P.Q * (1.0 - jnp.exp(-P.b * p))


def hardening_slope(p, P: Params):
    """R'(p)."""
    return P.H + P.Q * P.b * jnp.exp(-P.b * p)


# ---------------------------------------------------------------------------
# Scalar consistency equation and its solution
# ---------------------------------------------------------------------------
def consistency_residual(dp, q_tr, p_n, P: Params):
    """g(dp) = q_tr - (3 mu + C) dp - R(p_n + dp);  the solution gives g = 0."""
    return q_tr - (3.0 * P.mu + P.C) * dp - hardening(p_n + dp, P)


def _newton_dp(q_tr, p_n, P: Params, n_iter: int):
    """Scalar Newton with a fixed number of iterations (needed for jit/vmap).

    The starting guess is the "linearized" predictor f_tr / (3 mu + C + R'(p_n)).
    It is exact when hardening is linear: the next iterations then do
    nothing (g = 0).
    """
    mu3C = 3.0 * P.mu + P.C
    f_tr = q_tr - hardening(p_n, P)
    dp0 = f_tr / (mu3C + hardening_slope(p_n, P))

    def body(_, dp):
        g = consistency_residual(dp, q_tr, p_n, P)
        dg = -(mu3C + hardening_slope(p_n + dp, P))
        return dp - g / dg

    return jax.lax.fori_loop(0, n_iter, body, dp0)


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _solve_dp_implicit(q_tr, p_n, P, n_iter):
    return _newton_dp(q_tr, p_n, P, n_iter)


@_solve_dp_implicit.defjvp
def _solve_dp_implicit_jvp(n_iter, primals, tangents):
    """Implicit function theorem: g(x(z), z) = 0  =>  dx = -(dg/dz . dz) / (dg/dx).

    We do NOT differentiate through the Newton iterations: the derivative is
    exact as soon as the primal solution has converged, whatever n_iter is.
    """
    q_tr, p_n, P = primals
    x = _solve_dp_implicit(q_tr, p_n, P, n_iter)
    dg_dx = jax.grad(consistency_residual, argnums=0)(x, q_tr, p_n, P)
    _, dg = jax.jvp(lambda q, pn, PP: consistency_residual(x, q, pn, PP), primals, tangents)
    return x, -dg / dg_dx


# ---------------------------------------------------------------------------
# Integration of the law (radial return)
# ---------------------------------------------------------------------------
def integrate(eps_new, state_old: State, params: Params, n_newton: int = 6,
              implicit: bool = True):
    """J2 radial return with isotropic (linear/Voce) and kinematic (Prager) hardening.

    Args:
        eps_new:   total strain at the end of the step (Mandel, 6).
        state_old: internal variables at the start of the step.
        params:    material parameters (``Params``, Python values -> no type promotion).
        n_newton:  number of Newton iterations for Delta p.
        implicit:  True -> derivative of Delta p from the implicit function theorem;
                   False -> derivative obtained by differentiating the iterations ("unrolled").

    Returns:
        (sigma_new, state_new)
    """
    P = params
    one = ONE.astype(eps_new.dtype)
    mu, K = P.mu, P.K

    # 1) elastic predictor
    eps_e_tr = eps_new - state_old.eps_p
    tr = eps_e_tr[0] + eps_e_tr[1] + eps_e_tr[2]
    s_tr = 2.0 * mu * (eps_e_tr - tr / 3.0 * one)        # trial deviator
    sigma_tr = K * tr * one + s_tr
    xi_tr = s_tr - state_old.alpha                        # relative stress

    # 2) "safe" equivalent stress (double where: no sqrt'(0) = inf)
    xx = jnp.dot(xi_tr, xi_tr)
    nonzero = xx > 0.0
    xx_safe = jnp.where(nonzero, xx, 1.0)
    q_tr = jnp.where(nonzero, jnp.sqrt(1.5 * xx_safe), 0.0)

    # 3) yield criterion
    R_n = hardening(state_old.p, P)
    f_tr = q_tr - R_n
    plastic = f_tr > 0.0

    # 4) plastic correction. In the elastic branch we replace q_tr by R_n,
    #    so the equation has the solution dp = 0 (no exp blow-up, no NaN).
    q_solve = jnp.where(plastic, q_tr, R_n)
    if implicit:
        dp = _solve_dp_implicit(q_solve, state_old.p, P, n_newton)
    else:
        dp = _newton_dp(q_solve, state_old.p, P, n_newton)
    dp = jnp.where(plastic, dp, 0.0)

    q_safe = jnp.where(nonzero, q_tr, 1.0)
    N = 1.5 * xi_tr / q_safe                               # flow direction, N:N = 3/2
    deps_p = dp * N

    sigma_new = sigma_tr - 2.0 * mu * deps_p
    state_new = State(
        eps_p=state_old.eps_p + deps_p,
        p=state_old.p + dp,
        alpha=state_old.alpha + (2.0 / 3.0) * P.C * deps_p,
    )
    return sigma_new, state_new


def integrate_naive(eps_new, state_old: State, params: Params):
    """The "naive" version from the project brief (linear isotropic hardening, one where).

    The values are correct, but its Jacobian is NaN as soon as s_tr = 0 (for example
    at eps = 0): sqrt'(0) = inf and 0/0 in the plastic branch, which jnp.where still
    computes. Kept for the tests and the report.
    """
    P = params
    one = ONE.astype(eps_new.dtype)
    mu, K = P.mu, P.K
    eps_e_tr = eps_new - state_old.eps_p
    tr = eps_e_tr[0] + eps_e_tr[1] + eps_e_tr[2]
    s_tr = 2.0 * mu * (eps_e_tr - tr / 3.0 * one)
    q_tr = jnp.sqrt(1.5 * jnp.dot(s_tr, s_tr))
    f_tr = q_tr - (P.sigma_y + P.H * state_old.p)
    dp = f_tr / (3.0 * mu + P.H)
    s_pl = (1.0 - 3.0 * mu * dp / q_tr) * s_tr
    deps_p = 1.5 * dp / q_tr * s_tr
    plastic = f_tr > 0.0
    s_new = jnp.where(plastic, s_pl, s_tr)
    sigma_new = K * tr * one + s_new
    state_new = State(
        eps_p=jnp.where(plastic, state_old.eps_p + deps_p, state_old.eps_p),
        p=jnp.where(plastic, state_old.p + dp, state_old.p),
        alpha=state_old.alpha,
    )
    return sigma_new, state_new
