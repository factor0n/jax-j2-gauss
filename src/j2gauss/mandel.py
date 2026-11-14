"""Mandel notation for symmetric 2nd and 4th order tensors.

A symmetric 2nd order tensor a is stored as the vector
    [a11, a22, a33, sqrt(2) a23, sqrt(2) a13, sqrt(2) a12].

With this sqrt(2) factor:
  * the double contraction a : b is the usual dot product;
  * a 4th order tensor with minor symmetries becomes a 6x6 matrix,
    and the composition A : B is the usual matrix product;
  * the symmetric 4th order identity is the identity matrix I6.
(Voigt notation does not have these properties: it uses different
factors of 2 for stresses and strains.)
"""

import jax.numpy as jnp

SQ2 = jnp.sqrt(2.0)

ONE = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])   # 2nd order identity tensor
I6 = jnp.eye(6)                                      # symmetric 4th order identity
I_VOL = jnp.outer(ONE, ONE) / 3.0                    # spherical (volumetric) projector
I_DEV = I6 - I_VOL                                   # deviatoric projector

# (i, j) indices for each Mandel component
_IJ = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def to_mandel(a):
    """3x3 matrix (assumed symmetric) -> Mandel vector (6,).

    We read the symmetric part (a + a^T)/2. This makes sure that the
    derivative with respect to a 3x3 matrix keeps the minor symmetries.
    """
    s = 0.5 * (a + jnp.swapaxes(a, -1, -2))
    return jnp.stack(
        [s[..., 0, 0], s[..., 1, 1], s[..., 2, 2],
         SQ2 * s[..., 1, 2], SQ2 * s[..., 0, 2], SQ2 * s[..., 0, 1]],
        axis=-1,
    )


def from_mandel(v):
    """Mandel vector (..., 6) -> symmetric matrix (..., 3, 3)."""
    a12, a13, a23 = v[..., 5] / SQ2, v[..., 4] / SQ2, v[..., 3] / SQ2
    row0 = jnp.stack([v[..., 0], a12, a13], axis=-1)
    row1 = jnp.stack([a12, v[..., 1], a23], axis=-1)
    row2 = jnp.stack([a13, a23, v[..., 2]], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def trace(v):
    return v[..., 0] + v[..., 1] + v[..., 2]


def dev(v):
    """Deviatoric part: dev(a) = a - tr(a)/3 * 1."""
    return v - trace(v)[..., None] / 3.0 * ONE


def elastic_stiffness(E, nu):
    """Isotropic elasticity tensor C = 3K I_vol + 2 mu I_dev (6x6 matrix)."""
    mu = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    return 3.0 * K * I_VOL + 2.0 * mu * I_DEV
