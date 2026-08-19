"""Test de robustesse : que devient le modèle quand les 4 variables absentes de
la liasse fiscale marocaine sont retirées ?

Motivation
----------
La correspondance CGNC → Compustat (`src/liasse_mapping.py`) laisse quatre
variables à `NaN`, faute d'équivalent dans la liasse fiscale marocaine :

    prcc_f   cours de bourse de clôture      (sociétés cotées uniquement)
    csho     nombre de titres                (ETIC, rarement renseigné)
    dltis    augmentation des dettes de fin. (tableau de financement)
    sstk     augmentations de capital        (tableau de financement)

L'analyse SHAP d'un dossier marocain révèle que `prcc_f` et `csho` figurent
parmi les plus forts contributeurs au score — alors même qu'ils sont vides. Le
modèle les route vers la branche par défaut apprise à l'entraînement, ce qui
déplace le score.

**La question n'est donc pas rhétorique : ces absences invalident-elles le
dispositif dans le contexte marocain ?**

Méthode
-------
On score les mêmes observations JAR2020 deux fois — une fois complètes, une
fois avec les 4 variables forcées à `NaN` (situation marocaine) — et l'on
compare :

1. le **niveau** des scores (les scores absolus sont-ils comparables ?) ;
2. la **corrélation de rang** (le classement survit-il ?) ;
3. le **recouvrement des dossiers sélectionnés** à budget de contrôle donné
   (les mêmes dossiers sont-ils programmés ?) ;
4. les **métriques de détection** (la capacité de détection est-elle dégradée ?).

C'est le point 3 qui compte le plus en pratique : c'est lui qui décide de la
liste de travail remise à la brigade.

Sortie
------
    reports/robustesse_variables_absentes.csv
    reports/figures/06_robustesse_variables_absentes.png

Usage
-----
    python scripts/robustesse_variables_absentes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from src.config import DATA_RAW, FIGURES_DIR, REPORTS_DIR
from src.inference import load_model, score_dataframe
from src.scoring import evaluate_scores, precision_recall_at_k

# Palette du projet (cohérente avec le dashboard)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#888780"
INK, INK_SOFT, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

VARIABLES_ABSENTES = ["prcc_f", "csho", "dltis", "sstk"]
BUDGETS = (0.01, 0.05, 0.10, 0.20)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.patch.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d7d7d3")
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(True, color="#e8e8e4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SOFT)
    ax.yaxis.label.set_color(INK_SOFT)
    ax.title.set_color(INK)
    return ax


def main() -> None:
    if not Path(DATA_RAW).exists():
        raise SystemExit(
            f"Données introuvables : {DATA_RAW}\n"
            "Téléchargez data_FraudDetection_JAR2020.csv depuis "
            "github.com/JarFraud/FraudDetection et placez-le dans data/raw/."
        )

    model, meta = load_model()
    train_end = meta.get("train_end_year", 2010)
    gap = meta.get("split_gap_years", 2)
    premiere_annee_hors_echantillon = train_end + gap

    raw = pd.read_csv(DATA_RAW)

    # Le modèle sauvegardé est entraîné sur les exercices <= train_end. Les
    # métriques de DÉTECTION ne sont honnêtes qu'au-delà de train_end + gap ;
    # la stabilité du CLASSEMENT, elle, se mesure sur toute la période, car
    # elle ne dépend pas des étiquettes.
    echantillon = raw[raw["fyear"] >= premiere_annee_hors_echantillon].copy()
    print(f"Échantillon hors échantillon d'entraînement : "
          f"{len(echantillon):,} lignes ({echantillon['fyear'].min()}-"
          f"{echantillon['fyear'].max()}), "
          f"{int(echantillon['misstate'].sum())} fraudes étiquetées.".replace(",", " "))

    # --- Scoring : complet vs dégradé (situation marocaine) ----------------
    complet = score_dataframe(echantillon, model=model, meta=meta).sort_index()

    degrade_entree = echantillon.copy()
    degrade_entree[VARIABLES_ABSENTES] = np.nan
    degrade = score_dataframe(degrade_entree, model=model, meta=meta).sort_index()

    s_complet = complet["risk_score"].values
    s_degrade = degrade["risk_score"].values
    y = echantillon.sort_index()["misstate"].values

    # --- 1. Niveau des scores ----------------------------------------------
    decalage = float(s_degrade.mean() - s_complet.mean())

    # --- 2. Corrélation de rang --------------------------------------------
    rho = float(spearmanr(s_complet, s_degrade).statistic)
    tau = float(kendalltau(s_complet, s_degrade).statistic)

    # --- 3. Recouvrement des dossiers sélectionnés + 4. détection ----------
    lignes = []
    for k in BUDGETS:
        n_k = int(np.ceil(len(s_complet) * k))
        sel_complet = set(np.argsort(-s_complet)[:n_k])
        sel_degrade = set(np.argsort(-s_degrade)[:n_k])
        recouvrement = len(sel_complet & sel_degrade) / n_k

        _, rappel_c = precision_recall_at_k(y, s_complet, k)
        _, rappel_d = precision_recall_at_k(y, s_degrade, k)

        lignes.append({
            "budget_pct": int(k * 100),
            "n_dossiers": n_k,
            "recouvrement_pct": round(recouvrement * 100, 1),
            "rappel_complet_pct": round(rappel_c * 100, 1),
            "rappel_degrade_pct": round(rappel_d * 100, 1),
            "ecart_rappel_pts": round((rappel_d - rappel_c) * 100, 1),
        })

    m_complet = evaluate_scores(y, s_complet)
    m_degrade = evaluate_scores(y, s_degrade)

    resultats = pd.DataFrame(lignes)
    resume = pd.DataFrame([{
        "n_observations": len(echantillon),
        "n_fraudes": int(y.sum()),
        "periode": f"{echantillon['fyear'].min()}-{echantillon['fyear'].max()}",
        "score_moyen_complet": round(float(s_complet.mean()), 4),
        "score_moyen_degrade": round(float(s_degrade.mean()), 4),
        "decalage_score_moyen": round(decalage, 4),
        "spearman": round(rho, 4),
        "kendall": round(tau, 4),
        "auc_pr_complet": round(m_complet["auc_pr"], 5),
        "auc_pr_degrade": round(m_degrade["auc_pr"], 5),
        "auc_roc_complet": round(m_complet["auc_roc"], 4),
        "auc_roc_degrade": round(m_degrade["auc_roc"], 4),
    }])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cible = REPORTS_DIR / "robustesse_variables_absentes.csv"
    with cible.open("w", encoding="utf-8-sig") as f:
        f.write("# Robustesse du modele aux 4 variables absentes de la liasse marocaine\n")
        f.write(f"# Variables retirees : {', '.join(VARIABLES_ABSENTES)}\n#\n# RESUME\n")
        resume.to_csv(f, index=False)
        f.write("#\n# PAR BUDGET DE CONTROLE\n")
        resultats.to_csv(f, index=False)

    # --- Restitution console ------------------------------------------------
    print(f"\nScore moyen   complet = {s_complet.mean():.4f}   "
          f"dégradé = {s_degrade.mean():.4f}   décalage = {decalage:+.4f}")
    print(f"Rang          Spearman = {rho:.4f}   Kendall = {tau:.4f}")
    print(f"AUC-PR        complet = {m_complet['auc_pr']:.5f}   "
          f"dégradé = {m_degrade['auc_pr']:.5f}")
    print(f"AUC-ROC       complet = {m_complet['auc_roc']:.4f}   "
          f"dégradé = {m_degrade['auc_roc']:.4f}")
    print("\n" + resultats.to_string(index=False))
    print(f"\nRapport écrit : {cible}")

    # --- Figure -------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.1))

    # (a) distribution des scores : le décalage de niveau
    style_axes(ax1)
    bins = np.linspace(0, 1, 45)
    ax1.hist(s_complet, bins=bins, color=GRAY, alpha=0.85, label="Toutes variables")
    ax1.hist(s_degrade, bins=bins, color=ORANGE, alpha=0.7,
             label="4 variables absentes\n(contexte marocain)")
    ax1.set_xlabel("Score de risque")
    ax1.set_ylabel("Nombre de dossiers")
    ax1.set_title("Le NIVEAU du score se déplace", fontsize=10.5, loc="left")
    ax1.legend(frameon=False, labelcolor=INK_SOFT, fontsize=8.5)

    # (b) le classement, lui, résiste
    style_axes(ax2)
    echant = np.random.default_rng(0).choice(len(s_complet),
                                             size=min(4000, len(s_complet)),
                                             replace=False)
    r_complet = pd.Series(-s_complet).rank().values
    r_degrade = pd.Series(-s_degrade).rank().values
    ax2.scatter(r_complet[echant], r_degrade[echant], s=3, alpha=0.25,
                color=BLUE, edgecolors="none")
    lim = [0, len(s_complet)]
    ax2.plot(lim, lim, color=GRAY, linewidth=1.4, linestyle="--")
    ax2.set_xlabel("Rang — toutes variables")
    ax2.set_ylabel("Rang — 4 variables absentes")
    ax2.set_title(f"…mais le CLASSEMENT résiste (ρ = {rho:.2f})",
                  fontsize=10.5, loc="left")

    # (c) recouvrement des dossiers programmés
    style_axes(ax3)
    x = np.arange(len(resultats))
    ax3.bar(x, resultats["recouvrement_pct"], color=BLUE, width=0.55)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"top {b} %" for b in resultats["budget_pct"]])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Dossiers communs (%)")
    ax3.set_xlabel("Budget de contrôle")
    ax3.set_title("Dossiers effectivement programmés\nen commun",
                  fontsize=10.5, loc="left")
    for xi, v in zip(x, resultats["recouvrement_pct"]):
        ax3.annotate(f"{v:.0f} %", xy=(xi, v), xytext=(0, 4),
                     textcoords="offset points", ha="center",
                     color=INK_SOFT, fontsize=9)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "06_robustesse_variables_absentes.png"
    fig.savefig(fig_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Figure écrite  : {fig_path}")


if __name__ == "__main__":
    main()
