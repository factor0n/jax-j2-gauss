"""Drive one material point under mixed loading (some strains / some stresses imposed).

At each step we impose some components of eps (mask ``controlled``)
and we look for the other ones so that the matching components of sigma
are zero. This is a small nonlinear system r(u) = sigma_free(u) = 0,
solved by Newton:

    K_ff du = -r,   u <- u + du

where K_ff is the free/free block of a tangent operator:
  * ``"consistent"``: consistent tangent (AD)  -> quadratic convergence;
  * ``"elastic"``   : elastic tangent C        -> linear convergence.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .j2 import Params, State, integrate, initial_state
from .tangent import tangent_elastic


@partial(jax.jit, static_argnames=("n_newton", "implicit"))
def _law_and_tangent(eps, state, P, n_newton=6, implicit=True):
    f = lambda e: integrate(e, state, P, n_newton=n_newton, implicit=implicit)
    (sig, st_new) = f(eps)
    Kt = jax.jacfwd(lambda e: f(e)[0])(eps)
    return sig, st_new, Kt


def run_mixed(eps_controlled_path, controlled, P: Params, tangent="consistent",
              tol=1e-10, max_iter=200, n_newton=6):
    """Drive the material point along a loading path.

    Args:
        eps_controlled_path: array (n_steps, 6); only the ``controlled`` columns
            are used (imposed values of eps, in Mandel notation).
        controlled: boolean mask (6,) of the imposed eps components.
        tangent: "consistent" or "elastic".
        tol: tolerance on ||r|| (MPa).

    Returns:
        dict with eps (n_steps, 6), sigma (n_steps, 6), p (n_steps,),
        and residuals: list (one per step) of the norms ||r_k||.
    """
    controlled = np.asarray(controlled, dtype=bool)
    free = ~controlled
    Ce = np.asarray(tangent_elastic(P))
    state = initial_state()
    eps = np.zeros(6)
    out_eps, out_sig, out_p, residuals = [], [], [], []

    for target in np.asarray(eps_controlled_path):
        eps = eps.copy()
        eps[controlled] = target[controlled]
        hist = []
        for _ in range(max_iter + 1):
            sig, st_new, Kt = _law_and_tangent(jnp.asarray(eps), state, P, n_newton=n_newton)
            r = np.asarray(sig)[free]
            nr = float(np.linalg.norm(r))
            hist.append(nr)
            if nr < tol:
                break
            K = np.asarray(Kt) if tangent == "consistent" else Ce
            eps[free] -= np.linalg.solve(K[np.ix_(free, free)], r)
        state = st_new
        out_eps.append(eps.copy())
        out_sig.append(np.asarray(sig))
        out_p.append(float(st_new.p))
        residuals.append(hist)

    return dict(eps=np.array(out_eps), sigma=np.array(out_sig), p=np.array(out_p),
                residuals=residuals)


UNIAXIAL = np.array([True, False, False, False, False, False])
TENSION_SHEAR = np.array([True, False, False, False, False, True])
