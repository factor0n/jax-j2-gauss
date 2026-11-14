"""Three ways to get the tangent operator d sigma_{n+1} / d eps_{n+1} (6x6 Mandel matrix).

  * ``tangent_ad``       : automatic differentiation (jax.jacfwd);
  * ``tangent_analytic`` : Simo & Hughes consistent tangent (closed form);
  * ``tangent_fd``       : central finite differences (independent reference).
"""

import jax
import jax.numpy as jnp

from .j2 import Params, State, integrate, hardening_slope
from .mandel import I_VOL, I_DEV


def tangent_elastic(P: Params):
    """C = 3K I_vol + 2 mu I_dev."""
    return 3.0 * P.K * I_VOL + 2.0 * P.mu * I_DEV


def tangent_ad(eps, state: State, P: Params, **kw):
    """Tangent by forward-mode AD: 6 JVPs (one per component of eps)."""
    return jax.jacfwd(lambda e: integrate(e, state, P, **kw)[0])(eps)


def tangent_analytic(eps, state: State, P: Params, **kw):
    """Consistent tangent (Simo & Hughes 1998, box 3.2, rewritten with p):

        C_ep = 3K I_vol + 2 mu theta I_dev - 2 mu theta_bar n (x) n
        theta     = 1 - 3 mu dp / q_tr
        theta_bar = 1 / (1 + (R'(p_{n+1}) + C) / (3 mu)) - (1 - theta)
        n         = xi_tr / ||xi_tr||

    It reduces to elasticity if the step is elastic.
    """
    mu, K = P.mu, P.K
    _, state_new = integrate(eps, state, P, **kw)
    dp = state_new.p - state.p

    eps_e = eps - state.eps_p
    xi = 2.0 * mu * (I_DEV @ eps_e) - state.alpha
    norm_xi = jnp.linalg.norm(xi)
    q_tr = jnp.sqrt(1.5) * norm_xi
    plastic = dp > 0.0

    q_safe = jnp.where(plastic, q_tr, 1.0)
    n = xi / jnp.where(plastic, norm_xi, 1.0)
    theta = 1.0 - 3.0 * mu * dp / q_safe
    h = hardening_slope(state_new.p, P) + P.C
    theta_bar = 1.0 / (1.0 + h / (3.0 * mu)) - (1.0 - theta)

    C_ep = 3.0 * K * I_VOL + 2.0 * mu * theta * I_DEV - 2.0 * mu * theta_bar * jnp.outer(n, n)
    return jnp.where(plastic, C_ep, tangent_elastic(P))


def tangent_fd(eps, state: State, P: Params, h: float = 1e-8, **kw):
    """Central finite differences, column by column. Error O(h^2) + O(eps_mach/h)."""
    cols = []
    for j in range(6):
        e = jnp.zeros(6, dtype=eps.dtype).at[j].set(h)
        sp = integrate(eps + e, state, P, **kw)[0]
        sm = integrate(eps - e, state, P, **kw)[0]
        cols.append((sp - sm) / (2.0 * h))
    return jnp.stack(cols, axis=1)
