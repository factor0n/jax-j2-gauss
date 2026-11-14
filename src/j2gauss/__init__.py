"""jax-j2-gauss: a differentiable, vectorized J2 elastoplastic law in JAX.

Importing the package turns on double precision (``jax_enable_x64``).
In solid mechanics, the consistent tangent and the Newton convergence tests
do not make sense in single precision.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .mandel import (  # noqa: E402
    ONE, I6, I_VOL, I_DEV, to_mandel, from_mandel, dev, trace, elastic_stiffness,
)
from .j2 import Params, State, integrate, integrate_naive, initial_state  # noqa: E402
from .tangent import (  # noqa: E402
    tangent_ad, tangent_analytic, tangent_fd, tangent_elastic,
)

__all__ = [
    "ONE", "I6", "I_VOL", "I_DEV", "to_mandel", "from_mandel", "dev", "trace",
    "elastic_stiffness", "Params", "State", "integrate", "integrate_naive",
    "initial_state", "tangent_ad", "tangent_analytic", "tangent_fd", "tangent_elastic",
]
