"""Convergence of the global Newton at the material point (tension with zero lateral stresses).

Four cases, the same large plastic increment:
  A. Voce, consistent tangent          -> quadratic;
  B. Voce, elastic tangent             -> linear;
  C. linear, proportional uniaxial     -> a single iteration (piecewise linear response);
  D. linear, tension + shear           -> quadratic (the direction n rotates).
"""

import csv
import os

import numpy as np

from _common import FIG, RES, SERIES, INK2, plt, save
from j2gauss import Params
from j2gauss.driver import run_mixed, UNIAXIAL, TENSION_SHEAR

LIN = Params()
VOCE = Params(Q=150.0, b=50.0)


def uni(e11):
    p = np.zeros((len(e11), 6)); p[:, 0] = e11
    return p


def ts(e11, g12):
    p = uni(e11); p[:, 5] = np.sqrt(2.0) * np.asarray(g12)
    return p


# first an elastic step, then a large plastic increment (the one we plot)
cases = [
    ("A  Voce, tangente cohérente", uni([0.001, 0.006]), UNIAXIAL, VOCE, "consistent"),
    ("B  Voce, tangente élastique", uni([0.001, 0.006]), UNIAXIAL, VOCE, "elastic"),
    ("C  Linéaire, uniaxial, cohérente", uni([0.001, 0.006]), UNIAXIAL, LIN, "consistent"),
    ("D  Linéaire, traction+cisaillement, cohérente", ts([0.001, 0.006], [0.0005, 0.004]),
     TENSION_SHEAR, LIN, "consistent"),
]

rows = []
fig, ax = plt.subplots(figsize=(6.4, 4.0))
markers = ["o", "s", "^", "D"]
for (label, path, mask, P, tg), c, m in zip(cases, SERIES, markers):
    res = run_mixed(path, mask, P, tangent=tg, tol=1e-11)
    r = np.array(res["residuals"][1])
    r_plot = np.maximum(r, 1e-16)
    ax.semilogy(np.arange(len(r)), r_plot, marker=m, color=c, label=label)
    # measured order q_k = log(r_{k+1}/r_k) / log(r_k/r_{k-1})
    orders = [np.log(r[k + 1] / r[k]) / np.log(r[k] / r[k - 1])
              for k in range(1, len(r) - 1) if r[k + 1] > 1e-13]
    for k, v in enumerate(r):
        rows.append(dict(case=label, iteration=k, residual=v))
    print(f"{label:48s} iterations={len(r)-1:3d}  orders={np.round(orders, 2)}")

ax.axhline(1e-11, color=INK2, lw=0.8, ls="--")
ax.text(14.5, 1.6e-11, "tolérance", color=INK2, fontsize=8, ha="right")
ax.set_xlabel("itération de Newton $k$")
ax.set_ylabel(r"$\|\mathbf{r}_k\|$ (MPa)")
ax.set_xlim(-0.3, 15)
ax.set_ylim(1e-16, 1e4)
ax.legend(loc="upper right")
save(fig, "newton_convergence")

with open(os.path.join(RES, "newton_convergence.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["case", "iteration", "residual"])
    w.writeheader(); w.writerows(rows)
