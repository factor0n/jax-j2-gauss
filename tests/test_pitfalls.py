"""JAX pitfalls: NaN in gradients, jit/vmap, precision."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from j2gauss import Params, initial_state, integrate, integrate_naive
from j2gauss.j2_numpy import integrate_batch

P = Params()
Z = jnp.zeros(6)


def _has_nan(x):
    return bool(np.isnan(np.asarray(x)).any())


def test_naive_reverse_mode_is_nan_at_zero():
    """A single where hides the tangents in forward mode, but not the cotangents
    in reverse mode: 0 * inf = NaN."""
    f = lambda e: integrate_naive(e, initial_state(), P)[0]
    assert not _has_nan(jax.jacfwd(f)(Z))
    assert _has_nan(jax.jacrev(f)(Z))


@pytest.mark.parametrize("mode", ["jacfwd", "jacrev"])
@pytest.mark.parametrize("implicit", [True, False])
def test_safe_has_no_nan(mode, implicit):
    f = lambda e: integrate(e, initial_state(), Params(Q=100.0, b=30.0, C=1e3),
                            implicit=implicit)[0]
    J = getattr(jax, mode)(f)(Z)
    assert not _has_nan(J)


def test_grad_wrt_material_parameters():
    """Reverse mode is used to identify parameters: d p / d sigma_y must be finite."""
    eps = jnp.array([5e-3, -2e-3, -2e-3, 0.0, 0.0, 0.0])

    def p_of(sy):
        return integrate(eps, initial_state(), Params(sigma_y=sy, Q=100.0, b=30.0))[1].p

    g = jax.grad(p_of)(250.0)
    assert np.isfinite(float(g)) and float(g) < 0.0


def test_jit_vmap_matches_loop_and_numpy():
    rng = np.random.default_rng(0)
    N = 64
    Pc = Params(Q=150.0, b=50.0, C=5_000.0)
    eps = jnp.asarray(rng.normal(0, 3e-3, (N, 6)))
    st = initial_state(shape=(N,))
    batched = jax.jit(jax.vmap(integrate, in_axes=(0, 0, None)))
    s_b, st_b = batched(eps, st, Pc)
    s_l = jnp.stack([integrate(eps[i], initial_state(), Pc)[0] for i in range(N)])
    np.testing.assert_allclose(s_b, s_l, rtol=1e-12, atol=1e-9)
    s_np, *_ = integrate_batch(np.asarray(eps), np.zeros((N, 6)), np.zeros(N), np.zeros((N, 6)), Pc)
    np.testing.assert_allclose(s_b, s_np, rtol=1e-10, atol=1e-8)


def test_float32_is_preserved():
    """The parameters are Python floats ("weak types"): no promotion to float64."""
    eps = jnp.ones((8, 6), dtype=jnp.float32) * 1e-3
    st = jax.tree.map(lambda a: a.astype(jnp.float32), initial_state(shape=(8,)))
    s, st2 = jax.vmap(integrate, in_axes=(0, 0, None))(eps, st, P)
    assert s.dtype == jnp.float32 and st2.p.dtype == jnp.float32
