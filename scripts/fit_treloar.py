"""Fit incompressible hyperelastic potentials to the Treloar (1944) data.

Models: Neo-Hooke (1 parameter), Mooney-Rivlin (2), Yeoh (3), ICNN network psi_theta(I1, I2).
Two scenarios:
  * "tous" (all)    : training on UT + ET + PS;
  * "UT+ET -> PS"   : training on UT + ET, pure shear is used for validation.

Loss: average over the tests of the normalized mean squared error
    L = (1/n_e) sum_e  mean((P_model - P_exp)^2) / mean(P_exp^2).
"""

import csv
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy.optimize import least_squares

from _common import ROOT, RES, SERIES, INK, INK2, plt, save
from j2gauss.hyperelastic import (nominal_stress, psi_nh, psi_mr, psi_yeoh, psi_icnn,
                                  icnn_init)

df = pd.read_csv(os.path.join(ROOT, "data", "treloar.csv"))
DATA = {t: (jnp.asarray(g.stretch.values), jnp.asarray(g.nominal_stress_MPa.values))
        for t, g in df.groupby("test")}
TESTS = ("UT", "ET", "PS")
NAMES = {"UT": "traction uniaxiale", "ET": "traction équibiaxiale", "PS": "cisaillement pur"}


def nmse(theta, psi, test):
    lam, P = DATA[test]
    Pm = nominal_stress(psi, theta, test, lam)
    return jnp.mean((Pm - P) ** 2) / jnp.mean(P ** 2)


def loss(theta, psi, tests):
    return sum(nmse(theta, psi, t) for t in tests) / len(tests)


def fit_analytic(psi, init, tests):
    keys = list(init)

    def residuals(x):
        th = dict(zip(keys, x))
        r = []
        for t in tests:
            lam, P = DATA[t]
            r.append((nominal_stress(psi, th, t, lam) - P) / jnp.sqrt(jnp.mean(P**2) * P.size * len(tests)))
        return jnp.concatenate(r)

    jac = jax.jit(jax.jacfwd(residuals))
    sol = least_squares(lambda x: np.asarray(residuals(jnp.asarray(x))),
                        np.array([init[k] for k in keys]),
                        jac=lambda x: np.asarray(jac(jnp.asarray(x))), xtol=1e-14, ftol=1e-14)
    return dict(zip(keys, map(float, sol.x)))


def fit_icnn(tests, seeds=(0, 1, 2), steps=8000):
    best = None
    for seed in seeds:
        th = icnn_init(jax.random.PRNGKey(seed))
        opt = optax.adam(optax.exponential_decay(2e-2, 2000, 0.3))
        s = opt.init(th)

        @jax.jit
        def step(th, s):
            L, g = jax.value_and_grad(loss)(th, psi_icnn, tests)
            u, s = opt.update(g, s)
            return optax.apply_updates(th, u), s, L

        for _ in range(steps):
            th, s, L = step(th, s)
        L = float(loss(th, psi_icnn, tests))
        print(f"   ICNN seed={seed} perte={L:.3e}")
        if best is None or L < best[1]:
            best = (th, L)
    return best[0]


MODELS = {
    "Neo-Hooke": (psi_nh, {"C10": 0.2}),
    "Mooney-Rivlin": (psi_mr, {"C10": 0.2, "C01": 0.01}),
    "Yeoh": (psi_yeoh, {"C10": 0.2, "C20": -1e-3, "C30": 1e-5}),
}

results, fitted = [], {}
for scen, tests in [("tous", TESTS), ("UT+ET -> PS", ("UT", "ET"))]:
    print(f"== scénario {scen}")
    for name, (psi, init) in MODELS.items():
        th = fit_analytic(psi, init, tests)
        fitted[(scen, name)] = (psi, th)
        print(f"   {name:14s} {th}")
    fitted[(scen, "ICNN")] = (psi_icnn, fit_icnn(tests))
    for name in list(MODELS) + ["ICNN"]:
        psi, th = fitted[(scen, name)]
        row = dict(scenario=scen, model=name)
        for t in TESTS:
            row[t] = float(nmse(th, psi, t))
        results.append(row)
        print(f"   {name:14s} " + "  ".join(f"{t}={row[t]:.2e}" for t in TESTS))

with open(os.path.join(RES, "treloar_errors.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["scenario", "model", *TESTS])
    w.writeheader(); w.writerows(results)

# ---- figure 1: scenario "tous" (all), the 4 models -------------------------------
styles = {"Neo-Hooke": (SERIES[2], ":"), "Mooney-Rivlin": (SERIES[3], "-."),
          "Yeoh": (SERIES[1], "--"), "ICNN": (SERIES[0], "-")}


def panel_figure(scen, models, fname, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, t in zip(axes, TESTS):
        lam, P = DATA[t]
        lfine = jnp.linspace(1.0, float(lam.max()), 200)
        for name in models:
            psi, th = fitted[(scen, name)]
            c, ls = styles[name]
            ax.plot(lfine, nominal_stress(psi, th, t, lfine), color=c, ls=ls, label=name)
        ax.plot(lam, P, "o", mfc="white", mec=INK, mew=1.2, ms=5, label="Treloar (1944)", zorder=5)
        held_out = scen != "tous" and t == "PS"
        ax.set_title(NAMES[t] + ("  (validation)" if held_out else ""))
        ax.set_xlabel(r"élongation $\lambda$")
    axes[0].set_ylabel("contrainte nominale $P$ (MPa)")
    axes[0].legend(loc="upper left")
    save(fig, fname)


panel_figure("tous", ["Neo-Hooke", "Mooney-Rivlin", "Yeoh", "ICNN"], "treloar_fit")
panel_figure("UT+ET -> PS", ["Mooney-Rivlin", "Yeoh", "ICNN"], "treloar_generalisation")

# ---- figure 2: learned derivatives psi_1, psi_2 (physical check) ------------------
psi, th = fitted[("tous", "ICNN")]
I = jnp.linspace(3.0, 60.0, 200)
g = jax.vmap(jax.grad(psi, argnums=(1, 2)), in_axes=(None, 0, 0))
fig, ax = plt.subplots(figsize=(5.2, 3.4))
lam_ut = jnp.linspace(1.0, 7.6, 200)
from j2gauss.hyperelastic import invariants  # noqa: E402
I1u, I2u = invariants("UT", lam_ut)
d1, d2 = g(th, I1u, I2u)
ax.plot(lam_ut, d1, color=SERIES[0], label=r"$\partial\psi/\partial I_1$")
ax.plot(lam_ut, d2, color=SERIES[1], ls="--", label=r"$\partial\psi/\partial I_2$")
ax.set_xlabel(r"élongation $\lambda$ (trajet uniaxial)")
ax.set_ylabel("MPa")
ax.legend()
save(fig, "treloar_icnn_derivatives")
print("psi_1 >= 0 :", bool((d1 >= 0).all()), " psi_2 >= 0 :", bool((d2 >= 0).all()))
