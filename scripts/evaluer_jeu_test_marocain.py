"""Évalue le ciblage sur le portefeuille simulé de la DRI de Tétouan.

Deux dispositifs sont comparés sur le même portefeuille :

1. **le modèle XGBoost entraîné sur JAR2020**, appliqué tel quel via la
   correspondance CGNC → Compustat ;
2. **la règle de rupture intra-contribuable** (`src/regle_rupture.py`), sans
   apprentissage, fondée uniquement sur des ratios invariants d'échelle.

⚠️ CE QUE CETTE ÉVALUATION ÉTABLIT — ET CE QU'ELLE N'ÉTABLIT PAS
=================================================================
Les anomalies sont **plantées par nous** puis recherchées. Il faut donc lire les
deux résultats avec des exigences différentes :

* **L'échec du modèle JAR2020 est un constat solide.** Le biais de taille qu'il
  révèle découle de l'écart entre la distribution d'entraînement (sociétés
  américaines cotées, total de bilan médian ~105 M$) et un portefeuille de
  direction régionale marocaine (~1 M$). Ce constat ne dépend en rien de la
  façon dont les anomalies ont été construites.

* **Le succès de la règle de rupture est, lui, partiellement circulaire.** Les
  anomalies ont été plantées sous forme de distorsions de ratios ; qu'une règle
  de détection de distorsions de ratios les retrouve n'a rien de surprenant.
  Cela établit que la chaîne de traitement transmet le signal et indique où
  celui-ci se trouve — pas que la règle détecterait de la fraude réelle.

Cette asymétrie doit être énoncée telle quelle dans le mémoire.

Sorties
-------
    reports/evaluation_jeu_test_marocain.csv
    reports/figures/07_evaluation_jeu_test_marocain.png

Usage
-----
    python scripts/evaluer_jeu_test_marocain.py
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
from scipy.stats import spearmanr

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR
from src.inference import score_with_diagnostics
from src.liasse_mapping import mapper_liasse, verifier_liasse
from src.regle_rupture import scorer_ruptures
from src.scoring import evaluate_scores, precision_recall_at_k

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#888780"
INK, INK_SOFT, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

DOSSIER = PROJECT_ROOT / "data" / "exemples_marocains"
BUDGETS = (0.01, 0.05, 0.10, 0.20, 0.30)
TAUX_MAD_USD = 9.85


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.patch.set_facecolor(SURFACE)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color("#d7d7d3")
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(True, color="#e8e8e4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SOFT)
    ax.yaxis.label.set_color(INK_SOFT)
    ax.title.set_color(INK)
    return ax


def rendement(y: np.ndarray, scores: np.ndarray, libelle: str) -> pd.DataFrame:
    lignes = []
    n_anomalies = int(y.sum())
    for k in BUDGETS:
        precision, rappel = precision_recall_at_k(y, scores, k)
        lignes.append({
            "dispositif": libelle,
            "budget_pct": int(k * 100),
            "dossiers_examines": max(1, int(np.ceil(len(y) * k))),
            "anomalies_captees": int(round(rappel * n_anomalies)),
            "part_captee_pct": round(rappel * 100, 1),
            "multiple_du_hasard": round(rappel / k, 2) if k else np.nan,
            "precision_pct": round(precision * 100, 1),
        })
    return pd.DataFrame(lignes)


def main() -> None:
    chemin_portefeuille = DOSSIER / "portefeuille_dri_tetouan.csv"
    chemin_cle = DOSSIER / "cle_anomalies.csv"

    if not chemin_portefeuille.exists() or not chemin_cle.exists():
        raise SystemExit(
            "Jeu de test absent. Générez-le d'abord :\n"
            "    python scripts/generer_jeu_test_marocain.py"
        )

    portefeuille = pd.read_csv(chemin_portefeuille)
    cle = pd.read_csv(chemin_cle)

    # --- 1. Contrôles ------------------------------------------------------
    diagnostic = verifier_liasse(portefeuille)
    print("=== Contrôles de cohérence ===")
    print(f"Scorable       : {diagnostic['scorable']}")
    print(f"Avertissements : {diagnostic['avertissements'] or 'aucun'}")
    if not diagnostic["scorable"]:
        raise SystemExit("Portefeuille non scorable :\n- "
                         + "\n- ".join(diagnostic["anomalies"]))

    # --- 2. Mapping --------------------------------------------------------
    donnees_modele, rapport = mapper_liasse(
        portefeuille, unite="dirhams", mad_par_usd=TAUX_MAD_USD
    )
    print("\n=== Mapping CGNC → variables du modèle ===")
    print(f"Couverture     : {rapport['taux_couverture_global_pct']} % "
          f"({rapport['n_variables_renseignees']}/28)")

    # --- 3. Les deux dispositifs ------------------------------------------
    resultat_modele = score_with_diagnostics(donnees_modele)
    scores_modele = resultat_modele["scored"][["gvkey", "fyear", "risk_score"]]

    scores_regle = scorer_ruptures(donnees_modele)[
        ["gvkey", "fyear", "score_rupture", "n_indicateurs_rompus"]
    ]

    evalue = (
        cle[["identifiant_fiscal", "exercice", "raison_sociale", "secteur",
             "taille", "anomalie", "profil_anomalie"]]
        .merge(scores_modele, left_on=["identifiant_fiscal", "exercice"],
               right_on=["gvkey", "fyear"], how="left", validate="one_to_one")
        .merge(scores_regle, on=["gvkey", "fyear"], how="left", validate="one_to_one")
        .merge(donnees_modele[["gvkey", "fyear", "at"]], on=["gvkey", "fyear"],
               how="left", validate="one_to_one")
    )
    assert evalue["risk_score"].notna().all(), "appariement incomplet"
    evalue["profil_anomalie"] = evalue["profil_anomalie"].fillna("")

    y = evalue["anomalie"].astype(int).values
    s_modele = evalue["risk_score"].values
    s_regle = evalue["score_rupture"].values
    n_anomalies = int(y.sum())

    print("\n=== Portefeuille évalué ===")
    print(f"{len(evalue)} lignes · {evalue['identifiant_fiscal'].nunique()} contribuables "
          f"· {n_anomalies} lignes porteuses d'une anomalie ({y.mean():.1%})")
    print(f"Couverture trajectoire : {resultat_modele['coverage']['available_pct']:.0f} %")

    # --- 4. Le diagnostic central : le modèle classe-t-il par TAILLE ? ----
    log_taille = np.log(np.clip(evalue["at"].values, 1e-9, None))
    correlations = pd.DataFrame([
        {
            "dispositif": "Modèle JAR2020",
            "corr_avec_anomalie": round(float(spearmanr(s_modele, y).statistic), 3),
            "corr_avec_taille": round(float(spearmanr(s_modele, log_taille).statistic), 3),
        },
        {
            "dispositif": "Règle de rupture",
            "corr_avec_anomalie": round(float(spearmanr(s_regle, y).statistic), 3),
            "corr_avec_taille": round(float(spearmanr(s_regle, log_taille).statistic), 3),
        },
    ])
    print("\n=== Diagnostic — sur quoi chaque dispositif classe-t-il ? ===")
    print("(corrélations de Spearman ; on veut fort sur l'anomalie, nul sur la taille)")
    print(correlations.to_string(index=False))

    # --- 5. Rendement comparé ---------------------------------------------
    rendements = pd.concat([
        rendement(y, s_modele, "Modèle JAR2020"),
        rendement(y, s_regle, "Règle de rupture"),
    ], ignore_index=True)

    m_modele = evaluate_scores(y, s_modele)
    m_regle = evaluate_scores(y, s_regle)

    print("\n=== Rendement par budget de contrôle ===")
    for libelle in ("Modèle JAR2020", "Règle de rupture"):
        bloc = rendements[rendements["dispositif"] == libelle]
        print(f"\n{libelle} :")
        print(bloc.drop(columns="dispositif").to_string(index=False))

    print("\n=== Métriques globales ===")
    print("                     AUC-PR    AUC-ROC")
    print(f"Modèle JAR2020      {m_modele['auc_pr']:.4f}    {m_modele['auc_roc']:.4f}")
    print(f"Règle de rupture    {m_regle['auc_pr']:.4f}    {m_regle['auc_roc']:.4f}")
    print(f"Référence hasard    {y.mean():.4f}    0.5000")

    # --- 6. Par profil d'anomalie -----------------------------------------
    evalue["pct_modele"] = pd.Series(s_modele).rank(pct=True).values * 100
    evalue["pct_regle"] = pd.Series(s_regle).rank(pct=True).values * 100

    profils = (
        evalue[evalue["anomalie"] == 1]
        .groupby("profil_anomalie")
        .agg(n_lignes=("anomalie", "size"),
             percentile_modele=("pct_modele", "median"),
             percentile_regle=("pct_regle", "median"))
        .round(1)
        .sort_values("percentile_regle", ascending=False)
    )
    print("\n=== Percentile médian par profil (100 = en tête du classement) ===")
    print(profils.to_string())

    # --- 7. Contrôle des biais de strate ----------------------------------
    seuil_modele = np.quantile(s_modele, 0.80)
    seuil_regle = np.quantile(s_regle, 0.80)
    evalue["signale_modele"] = (s_modele >= seuil_modele).astype(int)
    evalue["signale_regle"] = (s_regle >= seuil_regle).astype(int)

    biais_taille = (
        evalue.groupby("taille")
        .agg(n_lignes=("anomalie", "size"),
             signale_modele_pct=("signale_modele", "mean"),
             signale_regle_pct=("signale_regle", "mean"))
    )
    biais_taille[["signale_modele_pct", "signale_regle_pct"]] = (
        biais_taille[["signale_modele_pct", "signale_regle_pct"]] * 100
    ).round(1)
    biais_taille = biais_taille.sort_values("signale_modele_pct", ascending=False)

    biais_secteur = (
        evalue.groupby("secteur")
        .agg(n_lignes=("anomalie", "size"),
             signale_modele_pct=("signale_modele", "mean"),
             signale_regle_pct=("signale_regle", "mean"))
    )
    biais_secteur[["signale_modele_pct", "signale_regle_pct"]] = (
        biais_secteur[["signale_modele_pct", "signale_regle_pct"]] * 100
    ).round(1)
    biais_secteur = biais_secteur.sort_values("signale_modele_pct", ascending=False)

    print("\n=== Taux de signalement par taille (seuil top 20 % — attendu : 20 % partout) ===")
    print(biais_taille.to_string())
    print("\n=== Taux de signalement par secteur (seuil top 20 %) ===")
    print(biais_secteur.to_string())

    ecart_taille_modele = float(biais_taille["signale_modele_pct"].max()
                                - biais_taille["signale_modele_pct"].min())
    ecart_taille_regle = float(biais_taille["signale_regle_pct"].max()
                               - biais_taille["signale_regle_pct"].min())
    print(f"\nÉcart max entre classes de taille — modèle : {ecart_taille_modele:.1f} pts "
          f"· règle : {ecart_taille_regle:.1f} pts")

    # --- 8. Rapport --------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cible = REPORTS_DIR / "evaluation_jeu_test_marocain.csv"
    with cible.open("w", encoding="utf-8-sig") as f:
        f.write("# Evaluation du ciblage sur le portefeuille simule de la DRI de Tetouan\n")
        f.write("# DONNEES FICTIVES, anomalies plantees.\n")
        f.write("# L'echec du modele JAR2020 est un constat solide (ecart de distribution).\n")
        f.write("# Le succes de la regle de rupture est partiellement circulaire.\n#\n")
        f.write(f"# n_lignes={len(evalue)} n_anomalies={n_anomalies} "
                f"prevalence={y.mean():.4f}\n")
        f.write(f"# modele  : auc_pr={m_modele['auc_pr']:.5f} auc_roc={m_modele['auc_roc']:.4f}\n")
        f.write(f"# regle   : auc_pr={m_regle['auc_pr']:.5f} auc_roc={m_regle['auc_roc']:.4f}\n")
        f.write("#\n# SUR QUOI CHAQUE DISPOSITIF CLASSE-T-IL\n")
        correlations.to_csv(f, index=False)
        f.write("#\n# RENDEMENT PAR BUDGET DE CONTROLE\n")
        rendements.to_csv(f, index=False)
        f.write("#\n# PERCENTILE MEDIAN PAR PROFIL D'ANOMALIE\n")
        profils.to_csv(f)
        f.write("#\n# TAUX DE SIGNALEMENT PAR TAILLE (seuil top 20%)\n")
        biais_taille.to_csv(f)
        f.write("#\n# TAUX DE SIGNALEMENT PAR SECTEUR (seuil top 20%)\n")
        biais_secteur.to_csv(f)
    print(f"\nRapport écrit : {cible}")

    # --- 9. Figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax1, ax2, ax3 = axes

    # (a) courbes de captation comparées
    style_axes(ax1)
    ks = np.linspace(0.01, 0.50, 60)
    ax1.plot(ks * 100, ks * 100, color=GRAY, linewidth=2, linestyle="--",
             label="Tirage aléatoire")
    ax1.plot(ks * 100, [precision_recall_at_k(y, s_modele, k)[1] * 100 for k in ks],
             color=ORANGE, linewidth=2.4, label="Modèle JAR2020")
    ax1.plot(ks * 100, [precision_recall_at_k(y, s_regle, k)[1] * 100 for k in ks],
             color=BLUE, linewidth=2.4, label="Règle de rupture")
    ax1.set_xlabel("Part du portefeuille contrôlée (%)")
    ax1.set_ylabel("Anomalies captées (%)")
    ax1.set_title("Rendement à budget de contrôle donné", fontsize=10.5, loc="left")
    ax1.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9)

    # (b) sur quoi chaque dispositif classe-t-il
    style_axes(ax2)
    x = np.arange(2)
    largeur = 0.36
    ax2.bar(x - largeur / 2, correlations["corr_avec_anomalie"], largeur,
            color=BLUE, label="corr. avec l'anomalie")
    ax2.bar(x + largeur / 2, correlations["corr_avec_taille"], largeur,
            color=ORANGE, label="corr. avec la taille")
    ax2.axhline(0, color="#c9c9c4", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Modèle\nJAR2020", "Règle de\nrupture"])
    ax2.set_ylabel("Corrélation de Spearman")
    ax2.set_title("Sur quoi le classement repose-t-il ?", fontsize=10.5, loc="left")
    ax2.legend(frameon=False, labelcolor=INK_SOFT, fontsize=8.5)
    for xi, (a, t) in enumerate(zip(correlations["corr_avec_anomalie"],
                                    correlations["corr_avec_taille"])):
        ax2.annotate(f"{a:+.2f}", xy=(xi - largeur / 2, a), xytext=(0, 4 if a >= 0 else -12),
                     textcoords="offset points", ha="center", color=INK_SOFT, fontsize=8.5)
        ax2.annotate(f"{t:+.2f}", xy=(xi + largeur / 2, t), xytext=(0, 4 if t >= 0 else -12),
                     textcoords="offset points", ha="center", color=INK_SOFT, fontsize=8.5)

    # (c) biais de taille
    style_axes(ax3)
    ordre_tailles = ["TPE", "Petite entreprise", "Moyenne entreprise", "Grande entreprise"]
    b = biais_taille.reindex([t for t in ordre_tailles if t in biais_taille.index])
    x = np.arange(len(b))
    ax3.bar(x - largeur / 2, b["signale_modele_pct"], largeur, color=ORANGE,
            label="Modèle JAR2020")
    ax3.bar(x + largeur / 2, b["signale_regle_pct"], largeur, color=BLUE,
            label="Règle de rupture")
    ax3.axhline(20, color=GRAY, linestyle="--", linewidth=1.6)
    ax3.annotate("attendu si neutre", xy=(0, 20), xytext=(0, 5),
                 textcoords="offset points", color=INK_SOFT, fontsize=8.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels([t.replace(" entreprise", "\nentreprise") for t in b.index],
                        fontsize=8)
    ax3.set_ylabel("Dossiers signalés (%)")
    ax3.set_title("Biais de taille (seuil top 20 %)", fontsize=10.5, loc="left")
    ax3.legend(frameon=False, labelcolor=INK_SOFT, fontsize=8.5)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    chemin_figure = FIGURES_DIR / "07_evaluation_jeu_test_marocain.png"
    fig.savefig(chemin_figure, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Figure écrite : {chemin_figure}")

    print("\n" + "=" * 76)
    print("LECTURE — l'échec du modèle JAR2020 est un constat SOLIDE : il découle de")
    print("l'écart entre distributions et ne dépend pas de la façon dont les anomalies")
    print("ont été plantées. Le succès de la règle de rupture est, lui, PARTIELLEMENT")
    print("CIRCULAIRE : les anomalies ont été construites comme des distorsions de")
    print("ratios. Voir docs/06_transferabilite.md.")
    print("=" * 76)


if __name__ == "__main__":
    main()
