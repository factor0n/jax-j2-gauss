# Partie GPU : mode d'emploi

Ce dossier mesure, sur **ta** machine, si un GPU NVIDIA accélère l'intégration de la loi J2, et ce que coûtent les transferts entre le processeur et la carte graphique. C'est l'étape 6 du stage : « comparer les exécutions CPU et GPU, en tenant compte des transferts de données ».

Testé pour : portable Lenovo LOQ, Ubuntu 24.04, i5-12450HX, **RTX 2050 (4 Go)**. Le kit fonctionne avec n'importe quel GPU NVIDIA sous Linux.

## En bref : 3 commandes

```bash
cd jax-j2-gauss
bash gpu/setup_gpu.sh             # 1. installe JAX GPU dans .venv-gpu (une seule fois, ~5 min)
source .venv-gpu/bin/activate     # 2. active l'environnement (à chaque nouveau terminal)
bash gpu/run_all.sh               # 3. tests + diagnostic + benchmark + figures (~10-15 min)
```

À la fin, une archive `gpu_results_<machine>.zip` est créée : c'est elle qu'il faut m'envoyer pour mettre à jour le rapport.

Pour un premier essai rapide (environ 1 minute) : `bash gpu/run_all.sh --quick`.

## Avant de lancer

1. **Le pilote NVIDIA** doit être installé. Vérifie avec `nvidia-smi` : la commande doit afficher ta RTX 2050. Sinon, lance `sudo ubuntu-drivers install` puis redémarre.
   - Pas besoin d'installer CUDA toi-même : JAX télécharge sa propre version.
   - Le script choisit **CUDA 13** si ton pilote est en version 580 ou plus, sinon **CUDA 12** (pilote 525 ou plus).
2. **Branche le portable sur secteur**, et mets-le en mode performance (Lenovo Vantage, ou `Fn + Q`). Sur batterie, le GPU tourne au ralenti et les mesures n'ont aucun sens. `check_gpu.py` te prévient si tu es sur batterie.
3. **Ferme les applications lourdes** (navigateur avec vidéos, jeux), car elles prennent de la mémoire GPU et du temps de calcul.
4. Prévois environ **3 Go d'espace disque** (paquets CUDA) et **8 Go de RAM libre** pour les plus grands N.

## Ce que fait chaque script

| Script | Rôle | Sortie |
|---|---|---|
| `setup_gpu.sh` | crée `.venv-gpu`, installe `jax[cuda12/13]` et les dépendances, vérifie que JAX voit le GPU | - |
| `check_gpu.py` | mesure les « plafonds » de la machine : bande passante PCIe dans les deux sens, bande passante mémoire CPU et GPU, débit de calcul FP32 et FP64 du GPU, latence d'un appel | `benchmarks/results/gpu_check_<machine>.json` |
| `bench_gpu.py` | le benchmark (détails ci-dessous) | `benchmarks/results/bench_gpu_<machine>.csv` |
| `plot_gpu.py` | figures et résumé chiffré | `figures/gpu_*_<machine>.png/pdf`, `benchmarks/results/gpu_summary_<machine>.md` |
| `run_all.sh` | enchaîne tout, garde un journal, crée l'archive | `gpu_results_<machine>.zip` |

### Les configurations comparées

| Nom | Ce qu'on mesure | Question à laquelle ça répond |
|---|---|---|
| `numpy` | NumPy vectorisé, 1 cœur CPU | la référence « naïve » |
| `jax-cpu` | JAX compilé sur tous les cœurs CPU | ce qu'on obtient sans GPU |
| `gpu-resident` | calcul GPU, les données sont **déjà** sur la carte | la vitesse maximale du GPU |
| `gpu-transfer` | envoi des données sur la carte, calcul, puis retour, **par blocs** de 1 048 576 points | la vitesse réelle si code_aster (sur CPU) appelle le GPU à chaque itération |
| `jax-cpu+tangent`, `gpu+tangent` (option `--tangent`) | loi **et** tangente 6×6 par AD | le vrai besoin d'un code EF |

Pour chaque configuration, le benchmark couvre les deux précisions (FP64 et FP32) et N de 100 à 10 millions de points de Gauss.

Le benchmark commence par **vérifier les résultats** : le GPU doit donner les mêmes contraintes que NumPy (écart d'environ $10^{-16}$ en FP64 et $10^{-7}$ en FP32). Si ce n'est pas le cas, il le signale.

### Pourquoi traiter les données « par blocs »

En FP64, 10 millions de points représentent 1,5 Go d'entrées plus 1,5 Go de sorties, plus les tableaux intermédiaires : cela dépasse les 4 Go de la RTX 2050. La configuration `gpu-transfer` envoie donc les données par paquets d'un million de points. Le dernier paquet est complété par des zéros pour garder une **forme fixe**, sinon JAX recompilerait le noyau. C'est exactement ce que devra faire une interface code_aster ↔ JAXMat.

Pour `gpu-resident` au plus grand N, il est **normal** de voir `OOM` (*out of memory*) : la carte est trop petite pour tout garder. C'est un résultat en soi.

## Options utiles

```bash
python gpu/bench_gpu.py --help                     # toutes les options
python gpu/bench_gpu.py --quick                    # N <= 1e5
python gpu/bench_gpu.py --tangent                  # ajoute loi + tangente
python gpu/bench_gpu.py --precisions fp64          # une seule précision
python gpu/bench_gpu.py --chunk 524288             # blocs plus petits si OOM en gpu-transfer
python gpu/bench_gpu.py --nmax 6.5                 # s'arrêter à 3 millions de points
bash gpu/run_all.sh --tangent                      # tout, avec la tangente
```

## Ce qu'on s'attend à voir (hypothèses à vérifier)

1. **Données sur le GPU** : le GPU devrait battre JAX CPU à partir de quelques milliers de points, surtout en FP32.
2. **Avec les transferts** : le bus PCIe (environ 6 Go/s attendus sur ce portable) devrait plafonner le débit vers $2\times10^7$ points/s en FP64. Le GPU pourrait alors ne **pas** battre le CPU.
3. **FP64 contre FP32** : une carte grand public calcule le FP64 environ **64 fois** moins vite que le FP32. `check_gpu.py` le mesure directement (ligne « rapport FP64 / FP32 »). Comme la loi est limitée par la mémoire, l'écart réel sur la loi devrait être bien plus faible que ×64.
4. **La tangente** coûte plus cher (36 nombres de plus par point) : c'est là que le GPU a le plus de chances de gagner.

Si les mesures contredisent ces hypothèses, c'est intéressant aussi : le rapport sera corrigé en conséquence.

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `nvidia-smi: command not found` | pas de pilote NVIDIA | `sudo ubuntu-drivers install`, puis redémarrer |
| `jax.devices()` affiche `CpuDevice` seulement | JAX CPU installé, ou plugin CUDA incompatible avec le pilote | relancer `bash gpu/setup_gpu.sh` ; si ça échoue avec CUDA 13, forcer `bash gpu/setup_gpu.sh cuda12` |
| `PJRT_Api already exists for device type cuda` | plugins CUDA 12 et 13 installés en même temps | `pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt` puis relancer `setup_gpu.sh` |
| `CUDA_ERROR_NO_DEVICE` ou GPU absent après une mise en veille | le pilote a perdu la carte | redémarrer le PC |
| `nvidia-smi` marche mais le GPU n'est jamais utilisé | portable en mode « Intel seulement » | `prime-select query` doit répondre `nvidia` ou `on-demand` ; sinon `sudo prime-select on-demand` puis redémarrer |
| `OOM` en `gpu-transfer` | blocs trop gros pour la mémoire libre | `--chunk 524288` ou `--chunk 262144` |
| Le PC rame, swap | trop de RAM pour NumPy au grand N | `--numpy-max 1e6` ou `--nmax 6.5` |
| Résultats très variables d'un lancement à l'autre | batterie, mode économie, autre application | secteur, mode performance, fermer les applications, relancer |
| `ModuleNotFoundError: j2gauss` | script lancé depuis un autre dossier | lancer depuis la racine du dépôt : `python gpu/bench_gpu.py` |

## Tester le kit sans GPU

Tous les scripts Python acceptent `--pretend-gpu`, qui utilise le CPU à la place du GPU. C'est utile pour vérifier que tout fonctionne. Les chiffres « GPU » obtenus ainsi n'ont **aucun** sens physique.
