"""Palette et helpers de visualisation partagés, pour un rendu cohérent d'un
notebook à l'autre (les 5 notebooks choisissaient auparavant des couleurs
séparément, notebook par notebook, plot par plot).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR

# Palette qualitative utilisée pour distinguer les modèles/configurations
# dans les graphiques comparatifs (AUC-PR par année, etc.). Choisie pour
# rester lisible en niveaux de gris (impression) et pour un lecteur
# daltonien (deutéranopie/protanopie) : contrastes de luminosité marqués,
# pas seulement de teinte.
MODEL_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8856a7", "#888780", "#c9526b"]

# Couleurs sémantiques utilisées de façon cohérente pour fraude / non-fraude
COLOR_FRAUD = "#c9526b"
COLOR_NON_FRAUD = "#2a78d6"
COLOR_NEUTRAL = "#888780"


def set_style():
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.titleweight"] = "bold"


def save_fig(fig, name: str, subdir: str = "", dpi: int = 150) -> Path:
    """Sauvegarde `fig` dans `reports/figures/[subdir/]name.png` et retourne le chemin.

    Centralise la sauvegarde des figures : avant ce refactor, `reports/figures/`
    restait vide car aucun notebook n'appelait `savefig()`.
    """
    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_metric_by_year(
    results_by_name: dict[str, pd.DataFrame],
    metric: str = "auc_pr",
    title: str = "",
    ylabel: str | None = None,
    highlight_years: list[int] = (),
    ax=None,
):
    """Trace `metric` par année de test pour plusieurs modèles/configurations
    sur un même graphique, avec une palette cohérente. `highlight_years`
    ombre en rouge clair les années à traiter avec prudence (ex. [2011] pour
    le biais documenté dans 05_modeling_supervised_v2.ipynb).
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    for year in highlight_years:
        ax.axvspan(year - 0.5, year + 0.5, color="red", alpha=0.08, zorder=0)

    for (name, res), color in zip(results_by_name.items(), MODEL_PALETTE):
        if res is None or metric not in res.columns:
            continue
        ax.plot(res.index, res[metric], marker="o", label=name, color=color, linewidth=2)

    ax.set_xlabel("Année de test")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title)
    ax.legend()
    if own_fig:
        fig.tight_layout()
    return fig, ax
