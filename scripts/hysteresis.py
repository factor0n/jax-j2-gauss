"""Cyclic uniaxial loading (eps_11 imposed, zero lateral stresses): hysteresis loops.

Three hardening types with the same initial plastic modulus:
  * isotropic (Voce)   : the elastic domain grows, no Bauschinger effect;
  * kinematic (Prager) : the domain moves, Bauschinger effect, stable loop;
  * combined.
"""

import csv
import os

import numpy as np

from _common import RES, SERIES, plt, save
from j2gauss import Params
from j2gauss.driver import run_mixed, UNIAXIAL

amp, n_cycles, n_per_quarter = 0.008, 3, 40
quarter = np.linspace(0, 1, n_per_quarter + 1)[1:]
one_cycle = np.concatenate([quarter, 1 - quarter, -quarter, -1 + quarter])
e11 = amp * np.tile(one_cycle, n_cycles)
path = np.zeros((e11.size, 6)); path[:, 0] = e11

materials = [
    ("isotrope (Voce)", Params(H=500.0, Q=150.0, b=50.0, C=0.0)),
    ("cinématique (Prager)", Params(H=0.0, Q=0.0, b=0.0, C=8000.0)),
    ("combiné", Params(H=500.0, Q=100.0, b=50.0, C=4000.0)),
]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
rows = []
for ax, (name, P), c in zip(axes, materials, SERIES):
    res = run_mixed(path, UNIAXIAL, P)
    s11 = res["sigma"][:, 0]
    ax.plot(np.r_[0.0, e11] * 100, np.r_[0.0, s11], color=c)
    ax.set_title(name)
    ax.set_xlabel(r"$\varepsilon_{11}$ (%)")
    ax.axhline(0, color="#bdbcb7", lw=0.7); ax.axvline(0, color="#bdbcb7", lw=0.7)
    its = [len(h) - 1 for h in res["residuals"]]
    print(f"{name:22s} sigma_max={s11.max():7.1f}  sigma_min={s11.min():7.1f}  "
          f"p_final={res['p'][-1]:.4f}  Newton max={max(its)}")
    for e, s, p in zip(e11, s11, res["p"]):
        rows.append(dict(material=name, eps11=e, sigma11=s, p=p))
axes[0].set_ylabel(r"$\sigma_{11}$ (MPa)")
save(fig, "hysteresis")

with open(os.path.join(RES, "hysteresis.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["material", "eps11", "sigma11", "p"])
    w.writeheader(); w.writerows(rows)
