"""Build (and run) the demo notebooks from short cells."""

import os

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB = os.path.join(ROOT, "notebooks")
os.makedirs(NB, exist_ok=True)

SETUP = """import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))
import numpy as np, matplotlib.pyplot as plt
import j2gauss                      # turns on jax_enable_x64
from j2gauss import Params
from j2gauss.driver import run_mixed, UNIAXIAL, TENSION_SHEAR
%matplotlib inline"""

newton = [
    ("md", "# Convergence du Newton au point matériel\n\n"
           "On impose $\\varepsilon_{11}$ et on cherche les autres composantes de "
           "$\\boldsymbol{\\varepsilon}$ telles que les contraintes correspondantes soient nulles. "
           "Le résidu est $\\mathbf{r} = \\boldsymbol{\\sigma}_{\\text{libre}}$ ; on compare la "
           "tangente cohérente (obtenue par AD) et la tangente élastique."),
    ("code", SETUP),
    ("code", "def uni(e11):\n    p = np.zeros((len(e11), 6)); p[:, 0] = e11\n    return p\n\n"
             "voce = Params(Q=150., b=50.)\nlin = Params()\n"
             "cases = {\n"
             "  'Voce, cohérente': run_mixed(uni([1e-3, 6e-3]), UNIAXIAL, voce, 'consistent', tol=1e-11),\n"
             "  'Voce, élastique': run_mixed(uni([1e-3, 6e-3]), UNIAXIAL, voce, 'elastic', tol=1e-11),\n"
             "  'Linéaire, cohérente': run_mixed(uni([1e-3, 6e-3]), UNIAXIAL, lin, 'consistent', tol=1e-11),\n"
             "}\n"
             "for k, r in cases.items():\n    print(f'{k:22s}', ['%.1e' % x for x in r['residuals'][1]])"),
    ("md", "Avec la tangente cohérente le nombre de chiffres exacts double à chaque itération "
           "(convergence quadratique). Avec la tangente élastique le résidu est multiplié par un "
           "facteur constant (convergence linéaire). Avec l'écrouissage **linéaire** en chargement "
           "proportionnel, la réponse $\\sigma_{22}(\\varepsilon_{22})$ est affine par morceaux : "
           "Newton converge en **une** itération, on ne peut pas y voir la convergence quadratique."),
    ("code", "fig, ax = plt.subplots(figsize=(6, 4))\n"
             "for k, r in cases.items():\n"
             "    h = np.maximum(r['residuals'][1], 1e-16)\n"
             "    ax.semilogy(h, 'o-', label=k)\n"
             "ax.set_xlabel('itération k'); ax.set_ylabel('||r_k|| (MPa)'); ax.legend(); ax.grid(alpha=.3)"),
    ("code", "r = np.array(cases['Voce, cohérente']['residuals'][1])\n"
             "print('ordre empirique :', np.log(r[2]/r[1]) / np.log(r[1]/r[0]))"),
]

hyst = [
    ("md", "# Boucles d'hystérésis en traction-compression cyclique\n\n"
           "Comparaison écrouissage isotrope (Voce), cinématique (Prager) et combiné."),
    ("code", SETUP),
    ("code", "amp = 0.008\nq = np.linspace(0, 1, 41)[1:]\n"
             "e11 = amp * np.tile(np.concatenate([q, 1-q, -q, -1+q]), 3)\n"
             "path = np.zeros((e11.size, 6)); path[:, 0] = e11\n"
             "mats = {'isotrope': Params(H=500., Q=150., b=50.),\n"
             "        'cinématique': Params(H=0., C=8000.),\n"
             "        'combiné': Params(H=500., Q=100., b=50., C=4000.)}\n"
             "fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)\n"
             "for ax, (k, P) in zip(axes, mats.items()):\n"
             "    res = run_mixed(path, UNIAXIAL, P)\n"
             "    ax.plot(e11 * 100, res['sigma'][:, 0]); ax.set_title(k); ax.set_xlabel('eps11 (%)')\n"
             "    ax.grid(alpha=.3)\n"
             "axes[0].set_ylabel('sigma11 (MPa)')"),
    ("md", "* **Isotrope** : la limite d'élasticité croît dans les deux sens ; la boucle s'élargit "
           "puis se stabilise quand $R(p)$ sature.\n"
           "* **Cinématique** : le domaine élastique garde sa taille $2\\sigma_y$ mais se translate ; "
           "la plastification en compression commence avant $-\\sigma_y$ (effet Bauschinger) et la "
           "boucle est stable dès le premier cycle.\n"
           "* **Combiné** : les deux effets."),
]


def build(cells, name):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": NB}})
    path = os.path.join(NB, name)
    nbf.write(nb, path)
    print("notebook ->", path)


build(newton, "newton_convergence.ipynb")
build(hyst, "hysteresis.ipynb")
