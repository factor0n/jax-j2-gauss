"""Accuracy of the tangent.

1. Table: relative error AD / analytic and FD / analytic, for each hardening type.
2. Figure (left): finite difference error as a function of the step h (V-shaped curve).
3. Figure (right): error of the AD tangent as a function of the number of Newton iterations,
   differentiating through the iterations ("unrolled") or with the implicit function theorem.
"""

import os

import jax.numpy as jnp
import numpy as np

from _common import RES, SERIES, INK2, plt, save
from j2gauss import (Params, initial_state, integrate, tangent_ad, tangent_analytic,
                     tangent_fd)

eps = jnp.array([4e-3, -1.3e-3, -1.1e-3, 4e-4, 0.0, 1.2e-3])
st = initial_state()
mats = {
    "linéaire": Params(),
    "Voce": Params(Q=150.0, b=50.0),
    "Prager": Params(H=0.0, C=8000.0),
    "Voce + Prager": Params(Q=150.0, b=50.0, C=5000.0),
}


def rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


lines = ["materiau,err_AD,err_DF"]
for name, P in mats.items():
    Ca = tangent_analytic(eps, st, P)
    e_ad = rel(tangent_ad(eps, st, P), Ca)
    e_fd = rel(tangent_fd(eps, st, P), Ca)
    lines.append(f"{name},{e_ad:.2e},{e_fd:.2e}")
    print(f"{name:14s}  AD: {e_ad:.2e}   DF: {e_fd:.2e}")
with open(os.path.join(RES, "tangent_errors.csv"), "w") as f:
    f.write("\n".join(lines) + "\n")

P = mats["Voce + Prager"]
Ca = tangent_analytic(eps, st, P, n_newton=30)

# (1) FD as a function of h
hs = np.logspace(-12, -3, 37)
err_fd = [rel(tangent_fd(eps, st, P, h=float(h), n_newton=30), Ca) for h in hs]

# (2) AD as a function of the number of Newton iterations
ns = np.arange(0, 7)
err_unr, err_imp, err_primal = [], [], []
p_ref = float(integrate(eps, st, P, n_newton=30)[1].p)
for n in ns:
    err_unr.append(rel(tangent_ad(eps, st, P, n_newton=int(n), implicit=False), Ca))
    err_imp.append(rel(tangent_ad(eps, st, P, n_newton=int(n), implicit=True), Ca))
    err_primal.append(abs(float(integrate(eps, st, P, n_newton=int(n))[1].p) - p_ref) / p_ref)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.7))
a1.loglog(hs, np.maximum(err_fd, 1e-17), marker="o", color=SERIES[0])
a1.loglog(hs, 1e-19 / hs, ls="--", color=INK2, lw=1)
a1.loglog(hs, 1.5e3 * hs**2, ls=":", color=INK2, lw=1)
a1.text(1.5e-12, 3e-12, r"arrondi $\propto \epsilon_{mach}/h$", color=INK2, fontsize=8)
a1.text(1.5e-6, 3e-10, r"troncature $\propto h^2$", color=INK2, fontsize=8)
a1.set_ylim(1e-14, 1e0)
a1.set_xlabel("pas $h$"); a1.set_ylabel("écart relatif à la tangente analytique")
a1.set_title("Différences finies centrées")

clip = lambda v: np.maximum(v, 1e-17)
a2.semilogy(ns, clip(err_unr), marker="o", color=SERIES[0], label="AD à travers Newton")
a2.semilogy(ns, clip(err_imp), marker="s", color=SERIES[1], label="AD + fonctions implicites")
a2.semilogy(ns, clip(err_primal), marker="^", color=SERIES[2], label=r"erreur sur $\Delta p$ (primal)")
a2.set_xlabel(r"itérations de Newton sur $\Delta p$")
a2.set_title("Tangente AD, écrouissage Voce + Prager")
a2.set_ylim(1e-17, 1e0)
a2.legend()
save(fig, "tangent_accuracy")

for n, a, b, c in zip(ns, err_unr, err_imp, err_primal):
    print(f"n={n}: unrolled={a:.1e} implicit={b:.1e} primal={c:.1e}")
print("meilleur h DF:", hs[int(np.argmin(err_fd))], "err", min(err_fd))
