"""Orchestration du ciblage : combine les approches disponibles en un classement.

Trois lectures complémentaires d'une même liasse
------------------------------------------------

======================== ============================================= ==========
Lecture                  Question posée                                Apprentissage
======================== ============================================= ==========
Rupture de trajectoire   Les ratios de ce contribuable ont-ils          non
                         brusquement changé par rapport à son propre
                         exercice précédent ?
Écart sectoriel          S'écarte-t-il de ses pairs du même secteur,    non
                         le même exercice ?
Reconstitution (IA)      Déclare-t-il moins que ce que sa liasse        **oui**
                         laisse attendre ?
======================== ============================================= ==========

Les deux premières relèvent de `src.regle_rupture`, la troisième de
`src.modele_marocain`. Ce module les met en regard et produit le classement
unique remis à la brigade.

Pourquoi combiner
-----------------
Les trois lectures ne détectent pas les mêmes choses, et l'évaluation sur
portefeuille simulé le montre nettement :

* le **modèle de reconstitution** excelle sur les **minorations déclaratives** —
  chiffre d'affaires minoré, charges majorées, impôt anormalement faible
  (AUC-ROC 0,95) — mais reste, à dessein, aveugle à ce qui ne réduit pas la base
  imposable : stocks surévalués, accruals élevés, créances gonflées (AUC-ROC
  0,48, soit le hasard) ;
* les **règles** captent l'irrégularité sous toutes ses formes, sans distinction.

Le modèle apporte en revanche quelque chose qu'aucune règle ne donne : un
**montant attendu chiffré**. « Vous avez déclaré 22 MDH, la liasse en laisse
attendre 40 » est directement exploitable dans une procédure de reconstitution
(article 213 du CGI) ; « votre ratio a varié de 60 % » ne l'est pas.

⚠️ Réserve d'interprétation
---------------------------
Sur le portefeuille simulé, les règles obtiennent de meilleurs chiffres que le
modèle (AUC-ROC 0,98 contre 0,83). Ce résultat doit être lu avec précaution :
les anomalies de ce portefeuille ont été **plantées sous forme de distorsions de
ratios**, ce que les règles détectent par construction. La comparaison est donc
biaisée en leur faveur et ne permet pas de conclure qu'elles seraient
supérieures sur des dossiers réels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .modele_marocain import ModeleInsuffisant, scorer_reconstitution
from .regle_rupture import motif_principal, scorer_global

METHODES = {
    "combine": "Règles + modèle IA",
    "regles": "Règles seules (rupture + secteur)",
    "ia": "Modèle IA seul (reconstitution)",
}


def analyser_portefeuille(
    df_modele: pd.DataFrame,
    methode: str = "combine",
    poids_regles: float = 0.6,
    colonne_secteur: str = "secteur",
) -> tuple[pd.DataFrame, dict]:
    """Produit le classement de risque du portefeuille.

    Parameters
    ----------
    methode :
        ``"combine"``, ``"regles"`` ou ``"ia"``.
    poids_regles :
        Part des règles dans le score combiné. La valeur par défaut de 0,6
        privilégie légèrement les règles, qui couvrent un éventail plus large
        d'irrégularités, tout en conservant l'apport du modèle sur les
        minorations déclaratives.

    Returns
    -------
    (classement, rapport)
        ``classement`` trié par risque décroissant, avec ``score_ciblage``,
        ``rang``, ``percentile`` et ``motif``.
        ``rapport`` indique la méthode réellement appliquée, la qualité des
        modèles entraînés, et le cas échéant pourquoi le volet IA a été écarté.
    """
    if methode not in METHODES:
        raise ValueError(f"Méthode inconnue : {methode!r}. Choix : {list(METHODES)}")

    rapport: dict = {"methode_demandee": methode, "ia_disponible": False}

    # --- Volet règles ------------------------------------------------------
    regles = scorer_global(df_modele, colonne_secteur=colonne_secteur)

    # --- Volet apprentissage ----------------------------------------------
    ia = None
    if methode in ("combine", "ia"):
        try:
            ia = scorer_reconstitution(df_modele, colonne_secteur=colonne_secteur)
            rapport["ia_disponible"] = True
            rapport["diagnostics_ia"] = ia.attrs.get("diagnostics", {})
        except ModeleInsuffisant as e:
            # Portefeuille trop petit : on le dit, et on bascule sur les règles
            # plutôt que d'entraîner un modèle sur du bruit.
            rapport["ia_indisponible_motif"] = str(e)

    if ia is None:
        classement = regles.copy()
        classement["score_ciblage"] = classement["score_global"]
        rapport["methode_appliquee"] = "regles"
    else:
        colonnes_ia = [
            c for c in ia.columns
            if c.startswith(("prevu_", "residu_", "ecart_"))
            and c.endswith(("_pct", "_sale", "_txt", "_ni"))
            or c in ("score_reconstitution", "n_cibles_sous_declarees",
                     "percentile_reconstitution")
        ]
        classement = regles.merge(
            ia[["gvkey", "fyear", *dict.fromkeys(colonnes_ia)]],
            on=["gvkey", "fyear"], how="left", validate="one_to_one",
        )

        if methode == "ia":
            classement["score_ciblage"] = classement["score_reconstitution"]
            rapport["methode_appliquee"] = "ia"
        else:
            # Les deux scores n'ont pas la même échelle : on les ramène en rangs
            # relatifs avant de les pondérer, sans quoi celui dont les valeurs
            # sont numériquement les plus grandes dominerait pour une raison
            # purement arithmétique.
            rang_regles = classement["score_global"].rank(pct=True)
            rang_ia = classement["score_reconstitution"].rank(pct=True)
            classement["score_ciblage"] = (
                poids_regles * rang_regles + (1 - poids_regles) * rang_ia
            )
            rapport["methode_appliquee"] = "combine"
            rapport["poids_regles"] = poids_regles

    if methode == "regles":
        classement["score_ciblage"] = classement["score_global"]
        rapport["methode_appliquee"] = "regles"

    classement = classement.sort_values(
        "score_ciblage", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    classement["rang"] = np.arange(1, len(classement) + 1)
    classement["percentile"] = classement["score_ciblage"].rank(pct=True) * 100
    classement["motif"] = [
        _motif_enrichi(classement, i) for i in range(len(classement))
    ]

    rapport["n_lignes"] = len(classement)
    rapport["n_contribuables"] = int(classement["gvkey"].nunique())
    rapport["n_sans_historique"] = int((~classement["a_un_historique"]).sum())
    return classement, rapport


def _motif_enrichi(classement: pd.DataFrame, index_ligne: int) -> str:
    """Motif affiché dans la liste de travail.

    Une sous-déclaration chiffrée est un motif plus fort qu'une rupture de
    ratio : quand le modèle en signale une nette, elle prend le pas.
    """
    ligne = classement.iloc[index_ligne]

    # Le libellé porte son genre : « charge d'impôt » est féminin, les deux
    # autres masculins — l'accord doit suivre.
    meilleur_ecart = 0.0
    meilleur_libelle = ""
    meilleur_accord = "inférieur"
    for cle, libelle, accord in (("sale", "Chiffre d'affaires", "inférieur"),
                                 ("txt", "Charge d'impôt", "inférieure"),
                                 ("ni", "Résultat net", "inférieur")):
        colonne = f"ecart_{cle}_pct"
        if colonne in classement.columns:
            valeur = ligne.get(colonne)
            if pd.notna(valeur) and valeur < meilleur_ecart:
                meilleur_ecart = float(valeur)
                meilleur_libelle = libelle
                meilleur_accord = accord

    if meilleur_ecart <= -25:
        return (f"{meilleur_libelle} {meilleur_accord} de "
                f"{abs(meilleur_ecart):.0f} % à l'attendu")
    return motif_principal(classement, index_ligne)
