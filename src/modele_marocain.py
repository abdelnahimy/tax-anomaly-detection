"""Modèle d'apprentissage entraîné sur données marocaines — par reconstitution.

Le problème que ce module résout
--------------------------------
Un modèle supervisé de détection de fraude a besoin d'étiquettes : « cette
entreprise a fraudé, celle-là non ». Au Maroc, ces étiquettes n'existent pas et
ne peuvent pas exister hors de l'administration (secret professionnel,
article 246 du CGI). C'est ce qui interdit de refaire, sur données marocaines,
ce que Bao et al. (2020) ont fait sur données américaines.

**Mais il existe une cible qui, elle, est disponible sur chaque liasse : le
montant déclaré lui-même.**

Le principe
-----------
On n'apprend pas à prédire la fraude. On apprend à prédire **ce qu'un
contribuable aurait normalement déclaré**, compte tenu du reste de sa liasse —
son appareil productif, ses stocks, ses achats, ses créances, son secteur — puis
on mesure l'écart avec ce qu'il a effectivement déclaré.

    Une entreprise avec ces immobilisations, ces stocks, ces achats et ces
    créances déclare habituellement 40 MDH de chiffre d'affaires.
    Celle-ci en déclare 22. Écart : −45 %.

Trois grandeurs sont reconstituées, correspondant aux trois principaux vecteurs
de minoration de la base imposable :

===================== ====================================================
Cible                 Ce qu'un écart négatif suggère
===================== ====================================================
Chiffre d'affaires    Ventes non déclarées — l'appareil productif mobilisé
                      est sans rapport avec l'activité déclarée
Charge d'impôt        Charges non justifiées, reports déficitaires
                      contestables, provisions non déductibles
Résultat net          Minoration du résultat imposable, quel qu'en soit le
                      canal
===================== ====================================================

C'est la transposition, en apprentissage automatique, de la **reconstitution du
chiffre d'affaires** que pratique un vérificateur lorsqu'il écarte une
comptabilité (article 213 du CGI). La différence est que la norme n'est plus
fixée à dire d'expert, mais apprise sur l'ensemble du portefeuille.

Ce que c'est, et ce que ce n'est pas
------------------------------------
✅ De l'apprentissage supervisé véritable : XGBoost, validation croisée,
   importance des variables, explication SHAP.
✅ Entraîné sur des données **marocaines**, au format de la liasse fiscale.
✅ Sans aucune étiquette de fraude — donc applicable dès aujourd'hui.

❌ Ce n'est **pas** un détecteur de fraude. Un écart important signale une
   **incohérence déclarative**, ce qui n'est pas la même chose. Une entreprise
   peut légitimement s'écarter de la norme : sinistre, changement d'activité,
   sous-traitance massive, démarrage, restructuration. Le dispositif produit
   des dossiers **à examiner**, jamais des conclusions.

Deux précautions techniques, sans lesquelles le résultat serait faux
--------------------------------------------------------------------
1. **Prédiction hors échantillon obligatoire.** Si le modèle prédit le chiffre
   d'affaires d'une entreprise qu'il a vue à l'entraînement, il reproduit sa
   déclaration — y compris minorée — et l'écart s'annule. Toutes les prédictions
   sont donc produites par **validation croisée**, chaque dossier étant prédit
   par un modèle qui ne l'a jamais vu.

2. **Regroupement par contribuable.** Les exercices d'une même entreprise se
   ressemblent fortement. Séparer 2023 et 2024 d'un même contribuable entre
   entraînement et test laisserait fuir l'information. Le découpage se fait donc
   par `GroupKFold` sur l'identifiant fiscal.

Usage
-----
>>> from src.liasse_mapping import mapper_liasse
>>> from src.modele_marocain import scorer_reconstitution
>>> donnees, _ = mapper_liasse(liasses, unite="dirhams")
>>> resultat = scorer_reconstitution(donnees)
>>> resultat[["gvkey", "fyear", "score_reconstitution", "ecart_sale_pct"]].head()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

EPS = 1e-9

# ---------------------------------------------------------------------------
# Définition des cibles
# ---------------------------------------------------------------------------
# Le choix des variables explicatives est le point délicat. Deux écueils
# symétriques :
#
#   - trop peu de variables : le modèle prédit mal, et tout le monde paraît
#     anormal ;
#   - une variable trop proche de la cible : le modèle la recopie et plus
#     personne ne paraît anormal.
#
# Exemple concret : prédire la charge d'impôt à partir du résultat courant
# donnerait un modèle excellent et parfaitement inutile — l'impôt est
# mécaniquement proportionnel au résultat. Un contribuable qui minore son
# résultat verrait sa charge d'impôt jugée « normale ». On prédit donc l'impôt à
# partir de l'APPAREIL PRODUCTIF, pas du résultat déclaré.


@dataclass(frozen=True)
class Cible:
    """Une grandeur déclarée à reconstituer à partir du reste de la liasse."""

    nom: str
    libelle: str
    variables: tuple[str, ...]
    interpretation: str
    poids: float = 1.0
    exclusions: tuple[str, ...] = field(default=(), repr=False)


CIBLES: tuple[Cible, ...] = (
    Cible(
        nom="sale",
        libelle="Chiffre d'affaires",
        # Tout ce qui décrit les MOYENS mis en œuvre, rien qui découle du CA.
        # `cogs` (achats consommés) est le prédicteur le plus fort et c'est
        # voulu : acheter beaucoup en déclarant peu de ventes est précisément
        # le schéma de la minoration de recettes.
        variables=("at", "ppegt", "invt", "rect", "ap", "cogs", "dp",
                   "ceq", "lt", "act", "lct", "che", "dltt", "xint"),
        exclusions=("ni", "ib", "txt", "re"),
        interpretation=(
            "Le chiffre d'affaires déclaré est très inférieur à celui qu'implique "
            "l'appareil productif mobilisé (immobilisations, stocks, achats, "
            "créances). Oriente vers des ventes non déclarées."
        ),
        poids=1.2,
    ),
    Cible(
        nom="txt",
        libelle="Charge d'impôt sur les résultats",
        # Volontairement SANS `ni` ni `ib` : sinon le modèle apprendrait la
        # simple proportionnalité impôt/résultat et ne verrait plus rien.
        variables=("at", "ppegt", "sale", "invt", "rect", "ap", "cogs",
                   "dp", "ceq", "lt", "act", "lct", "che", "xint"),
        exclusions=("ni", "ib", "txp"),
        interpretation=(
            "La charge d'impôt est très inférieure à celle d'entreprises de "
            "dimension et de structure comparables. Oriente vers des charges non "
            "justifiées, des provisions non déductibles ou des reports "
            "déficitaires contestables."
        ),
        poids=1.0,
    ),
    Cible(
        nom="ni",
        libelle="Résultat net",
        variables=("at", "ppegt", "sale", "invt", "rect", "ap", "cogs",
                   "dp", "ceq", "lt", "act", "lct", "che", "xint", "dltt"),
        exclusions=("ib", "txt", "re"),
        interpretation=(
            "Le résultat déclaré est très inférieur à celui d'entreprises de "
            "dimension et de structure comparables. Oriente vers une minoration "
            "de la base imposable, quel qu'en soit le canal."
        ),
        poids=1.0,
    ),
)

NOMS_CIBLES: tuple[str, ...] = tuple(c.nom for c in CIBLES)

# Nombre minimal de contribuables sous lequel l'apprentissage n'a aucun sens.
# En deçà, le modèle apprendrait le bruit et signalerait au hasard.
EFFECTIF_MINIMUM = 40


class ModeleInsuffisant(RuntimeError):
    """Portefeuille trop petit pour entraîner un modèle de reconstitution."""


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------
def _log_signe(serie: pd.Series) -> pd.Series:
    """Log signé, **normalisé par l'échelle de la colonne**.

    Le log signé naïf, ``sign(x)·log(1+|x|)``, n'est pas invariant d'échelle :
    autour de 1 il se comporte comme l'identité, autour de 10⁶ comme un
    logarithme. Selon que les montants sont saisis en dirhams ou en millions,
    la compression n'est donc pas la même — et le modèle réapprend la taille de
    l'entreprise au lieu de la neutraliser. C'est exactement le défaut qui
    disqualifie le modèle JAR2020 sur données marocaines : il ne fallait pas le
    réintroduire ici.

    En divisant d'abord par l'ordre de grandeur médian de la colonne, on place
    toutes les valeurs dans un domaine où ``log1p`` se comporte comme un vrai
    logarithme. Le résidu du modèle devient alors ``log(déclaré / attendu)``,
    c'est-à-dire un **écart relatif**, indépendant de l'unité de saisie comme de
    la taille du contribuable.
    """
    valeurs = pd.to_numeric(serie, errors="coerce")
    # Facteur 10⁻³ : les valeurs typiques atterrissent autour de 10³, domaine où
    # log1p(x) ≈ log(x) à moins de 0,1 % près.
    unite = _echelle_robuste(valeurs) * 1e-3
    return np.sign(valeurs) * np.log1p(valeurs.abs() / unite)


def _echelle_robuste(serie: pd.Series) -> float:
    """Ordre de grandeur d'une colonne, insensible aux valeurs extrêmes."""
    echelle = pd.to_numeric(serie, errors="coerce").abs().replace(0, np.nan).median()
    return float(echelle) if np.isfinite(echelle) and echelle > 0 else 1.0


def _inverse_log_signe(serie_log: pd.Series, echelle: float) -> pd.Series:
    """Repasse de l'espace logarithmique normalisé aux montants d'origine.

    Indispensable pour restituer un **montant attendu** au vérificateur : sans
    cette inversion, la prédiction resterait exprimée dans l'unité interne du
    modèle et afficherait des montants sans rapport avec la liasse.

    À noter : le *résidu* — donc le score et l'écart en pourcentage — est
    indifférent à cette normalisation, puisque la même échelle s'applique au
    déclaré et au prédit et se simplifie dans la différence. Seul l'affichage
    était affecté.
    """
    unite = echelle * 1e-3
    return np.sign(serie_log) * np.expm1(serie_log.abs()) * unite


def _preparer_variables(
    df: pd.DataFrame, variables: Sequence[str], colonne_secteur: str
) -> pd.DataFrame:
    """Construit la matrice explicative : log signé des postes + secteur encodé."""
    X = pd.DataFrame(index=df.index)
    for nom in variables:
        if nom in df.columns:
            X[f"{nom}_log"] = _log_signe(df[nom])
        else:
            X[f"{nom}_log"] = np.nan

    # Le secteur est déterminant : les normes de marge, de rotation et de
    # structure d'actif n'ont rien de commun entre l'hôtellerie et le négoce.
    if colonne_secteur in df.columns:
        codes = df[colonne_secteur].astype("category").cat.codes
        X["secteur_code"] = codes.replace(-1, np.nan)
    return X


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
def entrainer_reconstitution(
    df_modele: pd.DataFrame,
    cible: Cible,
    colonne_secteur: str = "secteur",
    n_plis: int = 5,
    random_state: int = 42,
) -> dict:
    """Entraîne un modèle de reconstitution et renvoie les prédictions hors
    échantillon.

    Returns
    -------
    dict avec :
        ``prediction``  — série des valeurs prédites (échelle d'origine)
        ``residu``      — écart en log : négatif = déclaré < attendu
        ``ecart_pct``   — même écart, en pourcentage de la valeur attendue
        ``r2``          — qualité d'ajustement hors échantillon
        ``importances`` — contribution de chaque variable
    """
    try:
        from sklearn.model_selection import GroupKFold
        from xgboost import XGBRegressor
    except ImportError as e:  # pragma: no cover - dépendance déclarée
        raise ModeleInsuffisant(
            "scikit-learn et xgboost sont requis : pip install -r requirements.txt"
        ) from e

    if cible.nom not in df_modele.columns:
        raise ModeleInsuffisant(f"Colonne cible `{cible.nom}` absente du fichier.")

    n_contribuables = df_modele["gvkey"].nunique()
    if n_contribuables < EFFECTIF_MINIMUM:
        raise ModeleInsuffisant(
            f"{n_contribuables} contribuables seulement — il en faut au moins "
            f"{EFFECTIF_MINIMUM} pour qu'un modèle apprenne autre chose que du bruit."
        )

    X = _preparer_variables(df_modele, cible.variables, colonne_secteur)
    y = _log_signe(df_modele[cible.nom])
    groupes = df_modele["gvkey"].values

    exploitable = y.notna() & X.notna().any(axis=1)
    if int(exploitable.sum()) < EFFECTIF_MINIMUM:
        raise ModeleInsuffisant(
            f"Trop peu de lignes exploitables pour la cible `{cible.nom}`."
        )

    prediction_log = pd.Series(np.nan, index=df_modele.index, dtype="float64")
    importances = np.zeros(X.shape[1])
    n_modeles = 0

    # GroupKFold : tous les exercices d'un même contribuable restent du même
    # côté du découpage. Sans cela, 2023 prédirait trivialement 2024.
    n_plis_effectif = int(min(n_plis, pd.Series(groupes[exploitable.values]).nunique()))
    if n_plis_effectif < 2:
        raise ModeleInsuffisant("Pas assez de contribuables distincts pour valider.")

    decoupage = GroupKFold(n_splits=n_plis_effectif)
    index_exploitable = np.where(exploitable.values)[0]

    for i_train, i_test in decoupage.split(
        X.iloc[index_exploitable], y.iloc[index_exploitable],
        groups=groupes[index_exploitable],
    ):
        pos_train = index_exploitable[i_train]
        pos_test = index_exploitable[i_test]

        modele = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_lambda=2.0,
            random_state=random_state,
            n_jobs=2,
            verbosity=0,
        )
        modele.fit(X.iloc[pos_train].values, y.iloc[pos_train].values)
        prediction_log.iloc[pos_test] = modele.predict(X.iloc[pos_test].values)
        importances += modele.feature_importances_
        n_modeles += 1

    if n_modeles:
        importances /= n_modeles

    # Résidu en log = écart relatif. Négatif ⇒ le contribuable déclare moins
    # que ce que sa liasse laisse attendre.
    residu = y - prediction_log

    valide = residu.notna()
    if valide.sum() > 1:
        variance_totale = float(((y[valide] - y[valide].mean()) ** 2).sum())
        variance_residuelle = float((residu[valide] ** 2).sum())
        r2 = 1 - variance_residuelle / variance_totale if variance_totale > 0 else np.nan
    else:
        r2 = np.nan

    echelle_cible = _echelle_robuste(df_modele[cible.nom])

    return {
        "cible": cible,
        "prediction": _inverse_log_signe(prediction_log, echelle_cible),
        "residu": residu,
        "ecart_pct": (np.expm1(residu) * 100).clip(-99, 500),
        "r2": r2,
        "importances": pd.Series(importances, index=X.columns).sort_values(
            ascending=False
        ),
        "n_lignes_predites": int(valide.sum()),
    }


# ---------------------------------------------------------------------------
# Score global
# ---------------------------------------------------------------------------
def scorer_reconstitution(
    df_modele: pd.DataFrame,
    cibles: Sequence[Cible] = CIBLES,
    colonne_secteur: str = "secteur",
    n_plis: int = 5,
    seuil_alerte: float = -0.20,
) -> pd.DataFrame:
    """Score chaque dossier par l'écart entre déclaré et reconstitué.

    **Seuls les écarts NÉGATIFS comptent.** Déclarer plus que la norme n'est pas
    un risque fiscal : une entreprise très rentable n'a pas à être inquiétée.
    Retenir la valeur absolue de l'écart, comme le ferait une détection
    d'anomalies naïve, remplirait la liste de travail des meilleurs
    contribuables du portefeuille.

    Parameters
    ----------
    seuil_alerte :
        Résidu en deçà duquel une cible est comptée comme « sous-déclarée »
        (−0,20 ≈ 18 % de moins qu'attendu).

    Returns
    -------
    DataFrame trié par risque décroissant, avec pour chaque cible les colonnes
    ``prevu_<cible>``, ``ecart_<cible>_pct``, ``residu_<cible>``, plus
    ``score_reconstitution``, ``rang_reconstitution``, ``n_cibles_sous_declarees``.
    L'attribut ``.attrs["diagnostics"]`` porte le R² et les importances.
    """
    resultat = df_modele.copy()
    score = pd.Series(0.0, index=resultat.index)
    n_sous_declarees = pd.Series(0, index=resultat.index)
    diagnostics: dict[str, dict] = {}
    poids_total = 0.0

    for cible in cibles:
        try:
            sortie = entrainer_reconstitution(
                resultat, cible, colonne_secteur=colonne_secteur, n_plis=n_plis
            )
        except ModeleInsuffisant as e:
            diagnostics[cible.nom] = {"erreur": str(e)}
            continue

        residu = sortie["residu"]
        resultat[f"prevu_{cible.nom}"] = sortie["prediction"]
        resultat[f"residu_{cible.nom}"] = residu
        resultat[f"ecart_{cible.nom}_pct"] = sortie["ecart_pct"]

        # Seule la sous-déclaration alimente le score.
        manque = (-residu).clip(lower=0).fillna(0.0)
        score = score + cible.poids * manque
        n_sous_declarees = n_sous_declarees + (residu < seuil_alerte).fillna(False).astype(int)
        poids_total += cible.poids

        diagnostics[cible.nom] = {
            "libelle": cible.libelle,
            "r2": sortie["r2"],
            "n_lignes_predites": sortie["n_lignes_predites"],
            "importances": sortie["importances"],
        }

    if poids_total == 0:
        raise ModeleInsuffisant(
            "Aucune cible n'a pu être modélisée :\n- "
            + "\n- ".join(str(d.get("erreur", "")) for d in diagnostics.values())
        )

    resultat["score_reconstitution"] = score / poids_total
    resultat["n_cibles_sous_declarees"] = n_sous_declarees.astype(int)

    resultat = resultat.sort_values(
        "score_reconstitution", ascending=False, kind="mergesort"
    )
    resultat["rang_reconstitution"] = np.arange(1, len(resultat) + 1)
    resultat["percentile_reconstitution"] = (
        resultat["score_reconstitution"].rank(pct=True) * 100
    )
    resultat.attrs["diagnostics"] = diagnostics
    return resultat


def expliquer_reconstitution(
    df_score: pd.DataFrame, index_ligne: int
) -> pd.DataFrame:
    """Détaille, pour un dossier, l'écart sur chaque grandeur reconstituée.

    C'est la restitution destinée au vérificateur : montant déclaré, montant
    attendu, écart, et ce que cet écart suggère d'examiner.
    """
    ligne = df_score.iloc[index_ligne]
    lignes = []

    for cible in CIBLES:
        colonne_prevu = f"prevu_{cible.nom}"
        if colonne_prevu not in df_score.columns:
            continue
        declare = ligne.get(cible.nom)
        prevu = ligne.get(colonne_prevu)
        ecart = ligne.get(f"ecart_{cible.nom}_pct")
        if pd.isna(prevu) or pd.isna(declare):
            continue
        lignes.append({
            "grandeur": cible.libelle,
            "declare": float(declare),
            "attendu": float(prevu),
            "ecart_pct": float(ecart) if pd.notna(ecart) else np.nan,
            "_tri": float(ecart) if pd.notna(ecart) else 0.0,
            "lecture": cible.interpretation if (pd.notna(ecart) and ecart < -20)
                       else "Écart non significatif.",
        })

    if not lignes:
        return pd.DataFrame(columns=["grandeur", "declare", "attendu",
                                     "ecart_pct", "lecture"])

    return (
        pd.DataFrame(lignes)
        .sort_values("_tri")          # les plus fortes sous-déclarations d'abord
        .drop(columns="_tri")
        .reset_index(drop=True)
    )
