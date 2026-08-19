"""Règle de rupture intra-contribuable — ciblage transparent, sans modèle appris.

Pourquoi ce module existe
-------------------------
L'évaluation du modèle JAR2020 sur un portefeuille marocain simulé
(`scripts/evaluer_jeu_test_marocain.py`) donne un résultat sans appel : le score
appris **classe par taille d'entreprise, pas par anomalie** — corrélation de
+0,50 avec le total de bilan contre −0,11 avec l'anomalie effective.

L'explication est structurelle. JAR2020 ne contient que des sociétés
**américaines cotées** : leur total de bilan médian avoisine 105 millions de
dollars, quand celui d'un portefeuille de direction régionale marocaine se situe
autour de **1 million**. Les entreprises marocaines se massent donc dans la
queue basse de la distribution d'entraînement, là où le modèle n'a presque rien
appris. Et comme 48 de ses 54 variables sont des variables de **niveau**, il
n'a guère d'autre choix que de trier par ordre de grandeur.

Ce module propose l'alternative complémentaire : une règle qui n'utilise que des
grandeurs **invariantes d'échelle**, et qui compare chaque contribuable **à
lui-même** d'un exercice à l'autre plutôt qu'à une population étrangère.

Le principe
-----------
Une entreprise saine a des ratios stables. Sa marge brute, son taux effectif
d'imposition, son ratio de créances évoluent lentement. Une rupture soudaine et
simultanée sur plusieurs de ces ratios, sans mouvement sectoriel correspondant,
est exactement ce qu'un vérificateur cherche.

La règle mesure donc, pour chaque indicateur suivi, l'écart relatif entre
l'exercice courant et l'exercice précédent **du même contribuable**, puis les
agrège. Aucun apprentissage, aucun paramètre ajusté sur des données : le score
est entièrement lisible et opposable.

Ce que cette règle vaut, et ce qu'elle ne vaut pas
-------------------------------------------------
⚠️ Elle a été éprouvée sur un jeu **synthétique dont nous avons planté les
anomalies sous forme de distorsions de ratios**. Qu'une règle de détection de
distorsions de ratios les retrouve n'a rien de surprenant : cette validation est
**circulaire** et ne prouve pas que la règle détecterait de la fraude réelle.

Le constat sur le modèle JAR2020, lui, ne l'est pas : le biais de taille
découle de l'écart entre distributions et ne dépend pas de la façon dont les
anomalies ont été construites. C'est une asymétrie importante à énoncer.

Cette règle est donc proposée comme **point de départ opérationnel et étalon de
comparaison**, dans l'attente d'un réentraînement sur données marocaines
étiquetées — pas comme un dispositif validé.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

EPS = 1e-9


@dataclass(frozen=True)
class Indicateur:
    """Un ratio suivi d'un exercice à l'autre.

    Attributes
    ----------
    nom :
        Identifiant court, utilisé dans les colonnes de sortie.
    libelle :
        Formulation destinée au vérificateur.
    interpretation :
        Ce qu'une rupture sur cet indicateur suggère d'examiner. C'est cette
        phrase qui rend une alerte actionnable.
    """

    nom: str
    libelle: str
    interpretation: str


INDICATEURS: tuple[Indicateur, ...] = (
    Indicateur(
        "marge_brute",
        "Marge brute — (CA − achats consommés) / CA",
        "Une chute de marge brute sans explication sectorielle oriente vers une "
        "minoration du chiffre d'affaires ou une majoration des achats.",
    ),
    Indicateur(
        "taux_is_effectif",
        "Taux effectif d'imposition — IS / résultat avant impôt",
        "Un taux effectif qui s'effondre alors que l'activité se maintient "
        "oriente vers des charges non justifiées ou des reports déficitaires "
        "contestables.",
    ),
    Indicateur(
        "is_sur_ca",
        "Charge d'impôt rapportée au chiffre d'affaires",
        "Complète le taux effectif : reste lisible même lorsque le résultat "
        "avant impôt est proche de zéro, cas où le taux effectif s'affole.",
    ),
    Indicateur(
        "creances_sur_ca",
        "Créances clients rapportées au chiffre d'affaires",
        "Des créances qui progressent bien plus vite que le chiffre d'affaires "
        "orientent vers des ventes constatées mais jamais encaissées, ou des "
        "clients fictifs.",
    ),
    Indicateur(
        "stocks_sur_ca",
        "Stocks rapportés au chiffre d'affaires",
        "Une progression des stocks sans rapport avec l'activité oriente vers "
        "une surévaluation d'inventaire ou un décalage de résultat.",
    ),
    Indicateur(
        "accruals",
        "Accruals — (résultat net − dotations) / total de bilan",
        "Un résultat qui repose sur des écritures d'ajustement plus que sur des "
        "flux économiques.",
    ),
    Indicateur(
        "rotation_actif",
        "Rotation de l'actif — CA / total de bilan",
        "Une rotation qui se dégrade brutalement signale un chiffre d'affaires "
        "déclaré en décalage avec l'appareil productif mobilisé.",
    ),
    Indicateur(
        "poids_fournisseurs",
        "Dettes fournisseurs rapportées aux achats",
        "Un décalage entre dettes fournisseurs et achats déclarés peut trahir "
        "des achats fictifs ou des achats sans facture.",
    ),
)

NOMS_INDICATEURS: tuple[str, ...] = tuple(i.nom for i in INDICATEURS)

# Au-delà de ce seuil, l'écart relatif est écrêté : un ratio partant d'une base
# quasi nulle produit des variations de plusieurs milliers de pour cent qui
# écraseraient tous les autres indicateurs. On veut une rupture *simultanée sur
# plusieurs ratios*, pas un indicateur unique qui déraille.
ECRETAGE = 3.0


def calculer_indicateurs(df_modele: pd.DataFrame) -> pd.DataFrame:
    """Calcule les ratios suivis à partir des 28 variables comptables.

    Parameters
    ----------
    df_modele :
        Sortie de `src.liasse_mapping.mapper_liasse` — colonnes `gvkey`,
        `fyear` et variables comptables Compustat.

    Notes
    -----
    Tous les ratios sont **invariants d'échelle** : ils gardent le même sens en
    dirhams, en millions de dirhams ou en millions de dollars. C'est la propriété
    qui permet à cette règle d'échapper au biais de taille.
    """
    d = df_modele.copy()

    def _sur(numerateur: pd.Series, denominateur: pd.Series) -> pd.Series:
        return numerateur / denominateur.abs().clip(lower=EPS)

    resultats = pd.DataFrame(index=d.index)
    resultats["marge_brute"] = _sur(d["sale"] - d["cogs"], d["sale"])
    resultats["taux_is_effectif"] = _sur(d["txt"], d["ib"] + d["txt"])
    resultats["is_sur_ca"] = _sur(d["txt"], d["sale"])
    resultats["creances_sur_ca"] = _sur(d["rect"], d["sale"])
    resultats["stocks_sur_ca"] = _sur(d["invt"], d["sale"])
    resultats["accruals"] = _sur(d["ni"] - d["dp"], d["at"])
    resultats["rotation_actif"] = _sur(d["sale"], d["at"])
    resultats["poids_fournisseurs"] = _sur(d["ap"], d["cogs"])

    return resultats.clip(-50, 50)


def scorer_ruptures(
    df_modele: pd.DataFrame,
    indicateurs: Sequence[str] = NOMS_INDICATEURS,
    ecretage: float = ECRETAGE,
) -> pd.DataFrame:
    """Score chaque ligne par l'ampleur de sa rupture intra-contribuable.

    Returns
    -------
    DataFrame trié par risque décroissant, avec :
    `score_rupture` (somme des écarts relatifs écrêtés), `rang_rupture`,
    `percentile_rupture`, `n_indicateurs_rompus` (écart > 30 %), et une colonne
    `ecart_<indicateur>` par ratio suivi.

    Les lignes sans exercice précédent (première année d'observation d'un
    contribuable) obtiennent un score nul : **absence d'historique n'est pas
    absence d'anomalie**, et l'appelant doit les traiter à part plutôt que de
    conclure qu'elles sont saines.
    """
    manquantes = [c for c in ("gvkey", "fyear") if c not in df_modele.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes {manquantes} absentes : impossible de comparer un "
            "contribuable à lui-même d'un exercice à l'autre."
        )

    ordre = "__ordre_entree__"
    d = df_modele.copy()
    d[ordre] = np.arange(len(d))

    ratios = calculer_indicateurs(d)
    d = pd.concat([d, ratios], axis=1)
    d = d.sort_values(["gvkey", "fyear"])

    score = pd.Series(0.0, index=d.index)
    n_rompus = pd.Series(0, index=d.index)
    colonnes_ecart = {}

    for nom in indicateurs:
        precedent = d.groupby("gvkey")[nom].shift(1)
        ecart = ((d[nom] - precedent) / precedent.abs().clip(lower=1e-6)).abs()
        ecart = ecart.clip(upper=ecretage)
        colonnes_ecart[f"ecart_{nom}"] = ecart
        score = score + ecart.fillna(0.0)
        n_rompus = n_rompus + (ecart > 0.30).fillna(False).astype(int)

    d = pd.concat([d, pd.DataFrame(colonnes_ecart, index=d.index)], axis=1)
    d["score_rupture"] = score
    d["n_indicateurs_rompus"] = n_rompus
    d["a_un_historique"] = d.groupby("gvkey")["fyear"].transform("size").gt(1) & (
        d.groupby("gvkey").cumcount() > 0
    )

    # Restauration de l'ordre d'entrée (même précaution que dans
    # `src.inference.prepare_features` : le tri interne par (gvkey, fyear) ne
    # doit jamais transparaître en sortie).
    d = d.sort_values(ordre).drop(columns=ordre)

    # Le percentile se calcule sur l'ensemble, avant classement.
    d["percentile_rupture"] = d["score_rupture"].rank(pct=True) * 100

    # Le rang est attribué APRÈS le tri, et non par `rank()` : sur un
    # portefeuille sain, les scores sont massivement à égalité (tous nuls), et
    # `rank(method="first")` les départagerait par position d'entrée — le rang
    # affiché ne correspondrait alors plus à la ligne affichée.
    d = d.sort_values("score_rupture", ascending=False, kind="mergesort")
    d["rang_rupture"] = np.arange(1, len(d) + 1)
    return d


def scorer_ecarts_sectoriels(
    df_modele: pd.DataFrame,
    colonne_secteur: str = "secteur",
    indicateurs: Sequence[str] = NOMS_INDICATEURS,
    effectif_minimum: int = 5,
) -> pd.DataFrame:
    """Score chaque ligne par son écart aux autres contribuables du même secteur.

    Complément indispensable à la rupture temporelle : un contribuable qui
    minore son résultat **depuis toujours** n'a aucune rupture à détecter, mais
    se distingue de ses pairs. Inversement, un secteur entier qui traverse une
    mauvaise année produit des ruptures partout sans qu'aucune fraude n'existe.
    Les deux lectures sont donc nécessaires et complémentaires.

    La norme sectorielle est calculée **à partir du fichier lui-même** : c'est
    ce qui rend la méthode utilisable sans aucune donnée externe. Plus le
    portefeuille déposé est large, plus la norme est fiable — un argument
    supplémentaire pour travailler sur l'ensemble du ressort plutôt que sur une
    poignée de dossiers.

    L'écart est mesuré en **écarts absolus médians** (MAD) plutôt qu'en écarts-
    types : la moyenne et l'écart-type sont eux-mêmes tirés par les valeurs
    extrêmes qu'on cherche justement à détecter.

    Parameters
    ----------
    effectif_minimum :
        En deçà de ce nombre de lignes, un secteur n'a pas de norme calculable
        et ses contribuables reçoivent un score nul — **et non un score faible**,
        ce qui les réputerait sains à tort. La colonne `secteur_comparable`
        permet de les identifier et de les traiter à part.
    """
    d = df_modele.copy()
    ratios = calculer_indicateurs(d)
    d = pd.concat([d, ratios], axis=1)

    if colonne_secteur not in d.columns:
        # Sans secteur déclaré, tout le portefeuille forme une strate unique.
        d = d.assign(**{colonne_secteur: "(non renseigné)"})

    # La comparaison se fait à secteur ET exercice donnés : comparer un
    # contribuable de 2024 à des pairs de 2021 mélangerait l'écart sectoriel
    # avec la conjoncture.
    groupes = [colonne_secteur, "fyear"] if "fyear" in d.columns else [colonne_secteur]

    score = pd.Series(0.0, index=d.index)
    n_atypiques = pd.Series(0, index=d.index)
    colonnes_ecart = {}

    for nom in indicateurs:
        mediane = d.groupby(groupes)[nom].transform("median")
        # MAD : écart absolu médian, robuste aux valeurs extrêmes
        mad = d.groupby(groupes)[nom].transform(
            lambda s: (s - s.median()).abs().median()
        )
        # Un secteur très homogène donne un MAD nul ou quasi nul. Écarter
        # simplement l'indicateur serait le pire choix : c'est précisément dans
        # un secteur homogène qu'un dossier déviant est le plus significatif, et
        # on le laisserait passer. On retient donc un plancher proportionnel à
        # la médiane — un écart de 5 % de la norme sectorielle devient l'unité
        # de mesure à défaut de dispersion observée.
        plancher = (mediane.abs() * 0.05).clip(lower=1e-6)
        mad = mad.where(mad > plancher, plancher)
        ecart = ((d[nom] - mediane) / mad).abs().clip(upper=ECRETAGE * 2)
        colonnes_ecart[f"ecart_sect_{nom}"] = ecart
        score = score + ecart.fillna(0.0)
        n_atypiques = n_atypiques + (ecart > 3.0).fillna(False).astype(int)

    effectifs = d.groupby(groupes)[indicateurs[0]].transform("size")
    comparable = effectifs >= effectif_minimum

    d = pd.concat([d, pd.DataFrame(colonnes_ecart, index=d.index)], axis=1)
    d["score_secteur"] = score.where(comparable, 0.0)
    d["n_ratios_atypiques"] = n_atypiques.where(comparable, 0).astype(int)
    d["secteur_comparable"] = comparable
    d["effectif_secteur"] = effectifs.astype(int)

    return d


def scorer_global(
    df_modele: pd.DataFrame,
    poids_rupture: float = 0.5,
    colonne_secteur: str = "secteur",
) -> pd.DataFrame:
    """Combine les deux lectures en un classement unique.

    Les deux scores n'ayant pas la même échelle, ils sont ramenés en rangs
    relatifs (0 à 1) avant d'être pondérés — sans quoi celui dont les valeurs
    sont numériquement les plus grandes dominerait le classement pour une
    raison purement arithmétique.

    Parameters
    ----------
    poids_rupture :
        Part du score temporel dans le score final (0 = uniquement sectoriel,
        1 = uniquement temporel). Par défaut les deux pèsent également.
    """
    if not 0.0 <= poids_rupture <= 1.0:
        raise ValueError("`poids_rupture` doit être compris entre 0 et 1.")

    ruptures = scorer_ruptures(df_modele)
    secteurs = scorer_ecarts_sectoriels(df_modele, colonne_secteur=colonne_secteur)

    colonnes_secteur = [
        c for c in secteurs.columns
        if c.startswith("ecart_sect_")
        or c in ("score_secteur", "n_ratios_atypiques", "secteur_comparable",
                 "effectif_secteur")
    ]
    fusion = ruptures.merge(
        secteurs[["gvkey", "fyear", *colonnes_secteur]],
        on=["gvkey", "fyear"], how="left", validate="one_to_one",
    )

    rang_rupture = fusion["score_rupture"].rank(pct=True)
    rang_secteur = fusion["score_secteur"].rank(pct=True)
    fusion["score_global"] = (
        poids_rupture * rang_rupture + (1 - poids_rupture) * rang_secteur
    )

    fusion = fusion.sort_values("score_global", ascending=False, kind="mergesort")
    fusion["rang_global"] = np.arange(1, len(fusion) + 1)
    fusion["percentile_global"] = fusion["score_global"].rank(pct=True) * 100
    return fusion


def motif_principal(df_score: pd.DataFrame, index_ligne: int) -> str:
    """Une phrase, en français courant, résumant pourquoi un dossier remonte.

    Destinée à figurer directement dans la liste de travail : un chef de
    brigade doit comprendre le signalement sans ouvrir le détail.
    """
    ligne = df_score.iloc[index_ligne]

    candidats = []
    for nom in NOMS_INDICATEURS:
        col_temps = f"ecart_{nom}"
        if col_temps in df_score.columns and pd.notna(ligne.get(col_temps)):
            candidats.append((float(ligne[col_temps]), nom, "temps"))
        col_sect = f"ecart_sect_{nom}"
        if col_sect in df_score.columns and pd.notna(ligne.get(col_sect)):
            # Ramené à la même échelle que l'écart temporel pour comparaison
            candidats.append((float(ligne[col_sect]) / 3.0, nom, "secteur"))

    if not candidats:
        return "Aucun exercice de comparaison disponible"

    ampleur, nom, origine = max(candidats)
    if ampleur < 0.30:
        return "Aucun écart notable"

    court = {
        "marge_brute": "marge brute",
        "taux_is_effectif": "taux d'imposition effectif",
        "is_sur_ca": "charge d'impôt rapportée au chiffre d'affaires",
        "creances_sur_ca": "créances clients",
        "stocks_sur_ca": "stocks",
        "accruals": "écritures d'ajustement du résultat",
        "rotation_actif": "rotation de l'actif",
        "poids_fournisseurs": "dettes fournisseurs",
    }[nom]

    if origine == "temps":
        return f"Rupture sur {court} ({ampleur * 100:.0f} % vs exercice précédent)"
    # Formulation neutre en genre et en nombre : « stocks », « marge brute » et
    # « créances clients » ne s'accordent pas de la même façon.
    return f"Écart au secteur sur {court}"


def expliquer_rupture(
    df_score: pd.DataFrame, index_ligne: int, top_n: int = 5
) -> pd.DataFrame:
    """Détaille les indicateurs qui ont fait remonter un dossier.

    Contrairement à SHAP, aucune approximation n'intervient ici : l'explication
    **est** le calcul. C'est l'avantage décisif d'une règle transparente dans un
    contexte administratif où l'alerte doit pouvoir être discutée.
    """
    ligne = df_score.iloc[index_ligne]
    par_nom = {i.nom: i for i in INDICATEURS}

    lignes = []
    for nom in NOMS_INDICATEURS:
        colonne = f"ecart_{nom}"
        if colonne not in df_score.columns:
            continue
        ecart = ligne[colonne]
        if pd.isna(ecart):
            continue
        indicateur = par_nom[nom]
        lignes.append({
            "indicateur": indicateur.libelle,
            "valeur_exercice": round(float(ligne[nom]), 4) if nom in ligne else np.nan,
            "ecart_vs_exercice_precedent": f"{ecart * 100:.0f} %",
            "_tri": float(ecart),
            "a_examiner": indicateur.interpretation,
        })

    if not lignes:
        return pd.DataFrame(columns=[
            "indicateur", "valeur_exercice", "ecart_vs_exercice_precedent", "a_examiner",
        ])

    resultat = pd.DataFrame(lignes).sort_values("_tri", ascending=False)
    return resultat.head(top_n).drop(columns="_tri").reset_index(drop=True)
