"""Passerelle entre la **liasse fiscale marocaine** (CGNC) et le format d'entrée
du modèle, hérité du dataset JAR2020 (variables Compustat, entreprises américaines).

Pourquoi ce module existe
-------------------------
Le modèle a été entraîné sur 28 variables comptables nommées selon la
nomenclature Compustat (``act``, ``ceq``, ``sale``, ...). Ces libellés n'ont
aucun sens pour un vérificateur de la Direction Régionale des Impôts : lui
dispose de la **liasse fiscale** déposée par le contribuable — bilan (tableaux
n°1 et n°2), compte de produits et charges (tableau n°3), établis selon le
**Code Général de Normalisation Comptable** marocain.

Ce module établit la correspondance poste à poste entre les deux référentiels,
de sorte qu'un fichier extrait du système d'information de la DGI puisse être
scoré sans réentraînement. C'est la pièce qui rend le dispositif transposable
au contexte marocain : sans elle, le modèle reste un exercice sur données
américaines.

Trois précautions méthodologiques, toutes matérialisées dans le code
--------------------------------------------------------------------
1. **Les correspondances ne sont pas toutes exactes.** Certaines sont directes
   (``at`` = TOTAL ACTIF), d'autres sont des agrégats reconstruits
   (``lct`` = passif circulant + trésorerie-passif), d'autres encore sont des
   approximations assumées (``ib``). Chaque entrée de ``LIASSE_MAPPING`` porte
   un champ ``fidelite`` qui le dit explicitement — un tableau de mapping qui
   prétendrait à l'exactitude partout serait malhonnête.

2. **Quatre variables n'existent pas dans la liasse.** ``prcc_f`` (cours de
   bourse), ``csho`` (nombre d'actions), ``dltis`` et ``sstk`` (flux du tableau
   de financement) ne sont pas des postes du bilan ou du CPC. Elles sont
   laissées à ``NaN`` — XGBoost les traite nativement — et le module rapporte
   le taux de couverture obtenu, au lieu de combler silencieusement les trous
   avec des zéros, ce qui fausserait les scores.

3. **L'échelle compte.** Les 28 variables de niveau entrent dans le modèle en
   log signé : une liasse libellée en dirhams et un état financier Compustat
   libellé en millions de dollars ne se placent pas au même endroit de la
   distribution d'entraînement. La conversion est donc explicite et paramétrée
   (``unite``, ``mad_par_usd``). Les 6 ratios fiscaux métier, eux, sont
   invariants d'échelle : ils restent valides quelle que soit la devise.

Usage
-----
>>> from src.liasse_mapping import lire_liasse, mapper_liasse
>>> df_liasse = lire_liasse("exports_dgi/liasses_2023.csv")
>>> df_modele, rapport = mapper_liasse(df_liasse, unite="dirhams")
>>> from src.inference import score_dataframe
>>> scores = score_dataframe(df_modele)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import RAW_ACCOUNTING_VARS

# ---------------------------------------------------------------------------
# Colonnes identifiantes attendues dans un export de liasses
# ---------------------------------------------------------------------------
# `identifiant_fiscal` (IF) joue le rôle de `gvkey` : il identifie le
# contribuable de façon stable dans le temps, ce qui est la condition pour
# calculer les variations d'un exercice à l'autre. L'ICE conviendrait aussi ;
# ce qui compte est la stabilité de la clé, pas sa nature.
COL_ID = "identifiant_fiscal"
COL_EXERCICE = "exercice"

# ---------------------------------------------------------------------------
# Unités
# ---------------------------------------------------------------------------
# JAR2020 exprime les montants en MILLIONS DE DOLLARS. Une liasse marocaine est
# déposée en dirhams (parfois en milliers de dirhams pour les grandes
# entreprises). Le facteur ci-dessous ramène la saisie à l'unité d'entraînement.
UNITES_SUPPORTEES = {
    "dirhams": 1.0,
    "milliers_dirhams": 1e3,
    "millions_dirhams": 1e6,
}

# Taux de change de référence. Valeur par défaut indicative : le taux à retenir
# est celui publié par Bank Al-Maghrib pour la date de clôture de l'exercice
# considéré, et doit être passé explicitement pour tout travail sérieux.
MAD_PAR_USD_DEFAUT = 10.0


# ---------------------------------------------------------------------------
# Description d'une correspondance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Correspondance:
    """Une variable du modèle et son origine dans la liasse fiscale marocaine.

    Attributes
    ----------
    variable :
        Nom de la variable côté modèle (nomenclature Compustat).
    libelle_us :
        Définition Compustat d'origine, pour tracer la correspondance.
    poste_liasse :
        Poste ou combinaison de postes de la liasse marocaine.
    tableau :
        Tableau de la liasse où le trouver (Bilan actif, Bilan passif, CPC, ...).
    comptes_cgnc :
        Comptes du plan comptable marocain concernés, à titre indicatif.
    colonnes_source :
        Colonnes attendues dans le CSV d'entrée pour reconstituer la variable.
    calcul :
        Fonction de reconstruction à partir du DataFrame source. ``None`` si la
        variable est simplement recopiée depuis ``colonnes_source[0]``.
    fidelite :
        ``"directe"``    — le poste marocain correspond exactement ;
        ``"agregat"``    — reconstruction par somme/différence de postes exacts ;
        ``"approximation"`` — le concept diffère, l'écart est documenté ;
        ``"indisponible"``  — absent de la liasse, laissé à NaN.
    remarque :
        Justification de la correspondance, et de son écart le cas échéant.
    """

    variable: str
    libelle_us: str
    poste_liasse: str
    tableau: str
    comptes_cgnc: str
    colonnes_source: tuple[str, ...]
    fidelite: str
    remarque: str
    calcul: Callable[[pd.DataFrame], pd.Series] | None = field(default=None, repr=False)


def _col(df: pd.DataFrame, nom: str) -> pd.Series:
    """Colonne numérique, ou série de NaN si la colonne est absente.

    Absence != zéro. Un poste non renseigné doit rester NaN et non 0, sans quoi
    on affirmerait au modèle que l'entreprise n'a, par exemple, aucun stock —
    une information que le fichier ne donne pas.
    """
    if nom not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[nom], errors="coerce")


def _somme(df: pd.DataFrame, *noms: str) -> pd.Series:
    """Somme de postes, NaN seulement si TOUS les postes sont absents/vides.

    ``pd.Series.add(fill_value=0)`` traiterait un poste manquant comme nul, ce
    qui est le comportement voulu ici : dans une liasse, un poste laissé vide
    vaut zéro (ex. « titres et valeurs de placement » non renseigné). Mais si
    aucun des postes de l'agrégat n'est disponible, le résultat doit rester NaN.
    """
    cols = [_col(df, n) for n in noms]
    total = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in cols:
        total = total.add(c, fill_value=0.0)
    tout_absent = pd.concat(cols, axis=1).isna().all(axis=1)
    return total.mask(tout_absent)


# ---------------------------------------------------------------------------
# LA TABLE DE CORRESPONDANCE
# ---------------------------------------------------------------------------
# Ordre : celui de RAW_ACCOUNTING_VARS, pour que la lecture du tableau suive la
# liste officielle des variables du modèle.
LIASSE_MAPPING: tuple[Correspondance, ...] = (
    Correspondance(
        variable="act",
        libelle_us="Current Assets — Total",
        poste_liasse="TOTAL II (actif circulant hors trésorerie) + TOTAL III (trésorerie-actif)",
        tableau="Bilan — Actif",
        comptes_cgnc="3xxx, 51xx",
        colonnes_source=("total_actif_circulant", "tresorerie_actif"),
        fidelite="agregat",
        remarque=(
            "Le CGNC sépare l'actif circulant de la trésorerie-actif, là où la "
            "notion américaine de « current assets » englobe les deux. On somme "
            "donc les totaux II et III du bilan actif."
        ),
        calcul=lambda d: _somme(d, "total_actif_circulant", "tresorerie_actif"),
    ),
    Correspondance(
        variable="ap",
        libelle_us="Accounts Payable — Trade",
        poste_liasse="Fournisseurs et comptes rattachés",
        tableau="Bilan — Passif (dettes du passif circulant)",
        comptes_cgnc="4411, 4415",
        colonnes_source=("fournisseurs_comptes_rattaches",),
        fidelite="directe",
        remarque="Correspondance exacte : dette commerciale d'exploitation.",
    ),
    Correspondance(
        variable="at",
        libelle_us="Assets — Total",
        poste_liasse="TOTAL GÉNÉRAL ACTIF (I + II + III)",
        tableau="Bilan — Actif",
        comptes_cgnc="classes 2, 3, 5 (actif)",
        colonnes_source=("total_actif",),
        fidelite="directe",
        remarque="Correspondance exacte. Retenir la colonne NET du bilan actif.",
    ),
    Correspondance(
        variable="ceq",
        libelle_us="Common/Ordinary Equity — Total",
        poste_liasse="TOTAL DES CAPITAUX PROPRES (A)",
        tableau="Bilan — Passif (financement permanent)",
        comptes_cgnc="11xx",
        colonnes_source=("total_capitaux_propres",),
        fidelite="directe",
        remarque=(
            "Les capitaux propres assimilés (B : subventions d'investissement, "
            "provisions réglementées) sont volontairement exclus : ils ne sont pas "
            "du capital actionnarial, et Compustat les classerait hors `ceq`."
        ),
    ),
    Correspondance(
        variable="che",
        libelle_us="Cash and Short-Term Investments",
        poste_liasse="TOTAL III (trésorerie-actif) + Titres et valeurs de placement",
        tableau="Bilan — Actif",
        comptes_cgnc="51xx, 350x",
        colonnes_source=("tresorerie_actif", "titres_valeurs_placement"),
        fidelite="agregat",
        remarque=(
            "« Cash and short-term investments » réunit la trésorerie et les "
            "placements liquides, que le CGNC range dans deux rubriques distinctes."
        ),
        calcul=lambda d: _somme(d, "tresorerie_actif", "titres_valeurs_placement"),
    ),
    Correspondance(
        variable="cogs",
        libelle_us="Cost of Goods Sold",
        poste_liasse="Achats revendus de marchandises + Achats consommés de matières et fournitures",
        tableau="CPC — charges d'exploitation",
        comptes_cgnc="611, 612",
        colonnes_source=("achats_revendus_marchandises", "achats_consommes_matieres"),
        fidelite="agregat",
        remarque=(
            "Le CPC marocain n'affiche pas de « coût des ventes » : il est "
            "reconstruit à partir des deux postes d'achats. Les autres charges "
            "externes (613/614) sont exclues, comme le fait Compustat qui les "
            "classe en charges opérationnelles hors coût des ventes."
        ),
        calcul=lambda d: _somme(d, "achats_revendus_marchandises", "achats_consommes_matieres"),
    ),
    Correspondance(
        variable="csho",
        libelle_us="Common Shares Outstanding",
        poste_liasse="Nombre de titres composant le capital social (ETIC, tableau B4)",
        tableau="ETIC — état des informations complémentaires",
        comptes_cgnc="1111",
        colonnes_source=("nombre_actions",),
        fidelite="approximation",
        remarque=(
            "Figure dans l'ETIC (tableau des capitaux propres) et non dans le "
            "bilan lui-même. À défaut : capital social / valeur nominale du titre. "
            "Non renseigné pour la plupart des sociétés non cotées."
        ),
    ),
    Correspondance(
        variable="dlc",
        libelle_us="Debt in Current Liabilities",
        poste_liasse="TOTAL V (trésorerie-passif) + part des dettes de financement à moins d'un an",
        tableau="Bilan — Passif",
        comptes_cgnc="55xx, 14xx (échéances < 1 an)",
        colonnes_source=("tresorerie_passif", "dettes_financement_moins_1an"),
        fidelite="agregat",
        remarque=(
            "La trésorerie-passif (crédits d'escompte, crédits de trésorerie, "
            "soldes bancaires créditeurs) constitue l'essentiel de la dette à "
            "court terme. L'échéancier des dettes de financement figure à l'ETIC "
            "(tableau B7) ; s'il n'est pas disponible, la colonne peut être omise."
        ),
        calcul=lambda d: _somme(d, "tresorerie_passif", "dettes_financement_moins_1an"),
    ),
    Correspondance(
        variable="dltis",
        libelle_us="Long-Term Debt Issuance",
        poste_liasse="Augmentation des dettes de financement (ressources stables)",
        tableau="Tableau de financement (tableau n°5)",
        comptes_cgnc="flux — compte 14xx",
        colonnes_source=("augmentation_dettes_financement",),
        fidelite="approximation",
        remarque=(
            "Variable de FLUX, absente du bilan. Le tableau de financement la "
            "fournit, mais il n'est exigible que des entreprises soumises au "
            "modèle normal. Laisser vide si indisponible."
        ),
    ),
    Correspondance(
        variable="dltt",
        libelle_us="Long-Term Debt — Total",
        poste_liasse="Dettes de financement (C)",
        tableau="Bilan — Passif (financement permanent)",
        comptes_cgnc="14xx (141 emprunts obligataires, 148 autres dettes)",
        colonnes_source=("dettes_financement",),
        fidelite="directe",
        remarque="Correspondance exacte : dettes à plus d'un an d'échéance à l'origine.",
    ),
    Correspondance(
        variable="dp",
        libelle_us="Depreciation and Amortization",
        poste_liasse="Dotations d'exploitation aux amortissements et aux provisions",
        tableau="CPC — charges d'exploitation",
        comptes_cgnc="619 (6191, 6192, 6193, 6194, 6195, 6196)",
        colonnes_source=("dotations_exploitation",),
        fidelite="approximation",
        remarque=(
            "Le poste marocain agrège amortissements ET provisions, là où `dp` ne "
            "vise que les dotations aux amortissements. Si l'ETIC (tableau B) "
            "permet d'isoler les seuls amortissements (6191/6192/6193), le faire."
        ),
    ),
    Correspondance(
        variable="ib",
        libelle_us="Income Before Extraordinary Items",
        poste_liasse="Résultat net de l'exercice − Résultat non courant",
        tableau="CPC",
        comptes_cgnc="résultat (8xxx)",
        colonnes_source=("resultat_net_exercice", "resultat_non_courant"),
        fidelite="approximation",
        remarque=(
            "`ib` est un résultat APRÈS IMPÔT hors éléments exceptionnels. Le "
            "« résultat courant » du CPC ne convient pas : il est AVANT impôt. On "
            "retranche donc le résultat non courant du résultat net. "
            "L'approximation porte sur l'effet d'impôt attaché au résultat non "
            "courant, que le CPC ne permet pas d'isoler."
        ),
        calcul=lambda d: _col(d, "resultat_net_exercice") - _col(d, "resultat_non_courant").fillna(0.0),
    ),
    Correspondance(
        variable="invt",
        libelle_us="Inventories — Total",
        poste_liasse="Stocks (F)",
        tableau="Bilan — Actif (actif circulant)",
        comptes_cgnc="31xx",
        colonnes_source=("stocks",),
        fidelite="directe",
        remarque="Correspondance exacte, colonne NET (après provisions pour dépréciation).",
    ),
    Correspondance(
        variable="ivao",
        libelle_us="Investment and Advances — Other",
        poste_liasse="Immobilisations financières (D)",
        tableau="Bilan — Actif (actif immobilisé)",
        comptes_cgnc="24x/25x (prêts immobilisés, titres de participation)",
        colonnes_source=("immobilisations_financieres",),
        fidelite="directe",
        remarque="Participations et créances financières à long terme.",
    ),
    Correspondance(
        variable="ivst",
        libelle_us="Short-Term Investments — Total",
        poste_liasse="Titres et valeurs de placement (H)",
        tableau="Bilan — Actif (actif circulant)",
        comptes_cgnc="350x",
        colonnes_source=("titres_valeurs_placement",),
        fidelite="directe",
        remarque="Correspondance exacte.",
    ),
    Correspondance(
        variable="lct",
        libelle_us="Current Liabilities — Total",
        poste_liasse="TOTAL IV (passif circulant) + TOTAL V (trésorerie-passif)",
        tableau="Bilan — Passif",
        comptes_cgnc="44xx, 45xx, 55xx",
        colonnes_source=("total_passif_circulant", "tresorerie_passif"),
        fidelite="agregat",
        remarque=(
            "Symétrique du traitement de `act` : la trésorerie-passif, isolée par "
            "le CGNC, fait partie des « current liabilities » au sens américain."
        ),
        calcul=lambda d: _somme(d, "total_passif_circulant", "tresorerie_passif"),
    ),
    Correspondance(
        variable="lt",
        libelle_us="Liabilities — Total",
        poste_liasse="TOTAL GÉNÉRAL PASSIF − Capitaux propres − Capitaux propres assimilés",
        tableau="Bilan — Passif",
        comptes_cgnc="14xx, 15xx, 44xx, 45xx, 55xx",
        colonnes_source=("total_actif", "total_capitaux_propres", "capitaux_propres_assimiles"),
        fidelite="agregat",
        remarque=(
            "Total des dettes, obtenu par différence — le bilan marocain n'affiche "
            "pas de sous-total « total des dettes ». Le total du passif étant égal "
            "au total de l'actif, on part de `total_actif`."
        ),
        calcul=lambda d: (
            _col(d, "total_actif")
            - _col(d, "total_capitaux_propres")
            - _col(d, "capitaux_propres_assimiles").fillna(0.0)
        ),
    ),
    Correspondance(
        variable="ni",
        libelle_us="Net Income (Loss)",
        poste_liasse="RÉSULTAT NET DE L'EXERCICE (XIII)",
        tableau="CPC — dernière ligne",
        comptes_cgnc="1191/1199",
        colonnes_source=("resultat_net_exercice",),
        fidelite="directe",
        remarque="Correspondance exacte. Doit coïncider avec le poste du bilan passif.",
    ),
    Correspondance(
        variable="ppegt",
        libelle_us="Property, Plant and Equipment — Total (Gross)",
        poste_liasse="Immobilisations corporelles — colonne BRUT",
        tableau="Bilan — Actif (actif immobilisé)",
        comptes_cgnc="23xx",
        colonnes_source=("immobilisations_corporelles_brut",),
        fidelite="directe",
        remarque=(
            "⚠️ Prendre la colonne **BRUT** du bilan actif, pas la colonne NET : "
            "`ppegt` est une valeur d'acquisition, avant amortissements. Le bilan "
            "marocain affiche les trois colonnes (Brut / Amortissements et "
            "provisions / Net), la correspondance est donc directe."
        ),
    ),
    Correspondance(
        variable="pstk",
        libelle_us="Preferred/Preference Stock (Capital) — Total",
        poste_liasse="— (sans équivalent)",
        tableau="—",
        comptes_cgnc="—",
        colonnes_source=("actions_privilegiees",),
        fidelite="indisponible",
        remarque=(
            "Le CGNC ne prévoit pas de catégorie d'actions de préférence au bilan. "
            "La loi 17-95 permet des actions à dividende prioritaire, mais elles "
            "sont rarissimes et non isolées comptablement. Valeur nulle en "
            "pratique pour la quasi-totalité des contribuables marocains."
        ),
    ),
    Correspondance(
        variable="re",
        libelle_us="Retained Earnings",
        poste_liasse="Réserve légale + Autres réserves + Report à nouveau + Résultats nets en instance d'affectation",
        tableau="Bilan — Passif (capitaux propres)",
        comptes_cgnc="114, 115, 116, 1169",
        colonnes_source=("reserve_legale", "autres_reserves", "report_a_nouveau",
                         "resultats_en_instance"),
        fidelite="agregat",
        remarque=(
            "Bénéfices accumulés non distribués. Le report à nouveau peut être "
            "négatif (pertes antérieures) : conserver le signe."
        ),
        calcul=lambda d: _somme(d, "reserve_legale", "autres_reserves",
                                "report_a_nouveau", "resultats_en_instance"),
    ),
    Correspondance(
        variable="rect",
        libelle_us="Receivables — Total",
        poste_liasse="Créances de l'actif circulant (G)",
        tableau="Bilan — Actif (actif circulant)",
        comptes_cgnc="342x (clients), 343x-349x (autres débiteurs)",
        colonnes_source=("creances_actif_circulant",),
        fidelite="directe",
        remarque=(
            "Poste global des créances, colonne NET. Il inclut les clients et les "
            "autres débiteurs, comme `rect` chez Compustat."
        ),
    ),
    Correspondance(
        variable="sale",
        libelle_us="Sales/Turnover (Net)",
        poste_liasse="CHIFFRE D'AFFAIRES (ventes de marchandises + ventes de biens et services produits)",
        tableau="CPC — produits d'exploitation",
        comptes_cgnc="711, 712",
        colonnes_source=("chiffre_affaires",),
        fidelite="directe",
        remarque=(
            "Le CPC affiche explicitement la ligne « Chiffre d'affaires ». Ne pas "
            "utiliser le total des produits d'exploitation, qui inclut la variation "
            "de stocks et les reprises."
        ),
    ),
    Correspondance(
        variable="sstk",
        libelle_us="Sale of Common and Preferred Stock",
        poste_liasse="Augmentations de capital, apports (ressources stables)",
        tableau="Tableau de financement (tableau n°5)",
        comptes_cgnc="flux — comptes 111x, 112x",
        colonnes_source=("augmentation_capital",),
        fidelite="approximation",
        remarque=(
            "Variable de FLUX, comme `dltis`. Disponible dans le tableau de "
            "financement du modèle normal. Laisser vide si indisponible."
        ),
    ),
    Correspondance(
        variable="txp",
        libelle_us="Income Taxes Payable",
        poste_liasse="État — impôts sur les résultats (dette)",
        tableau="Bilan — Passif (dettes du passif circulant)",
        comptes_cgnc="4453",
        colonnes_source=("etat_impots_sur_resultats_dette",),
        fidelite="directe",
        remarque=(
            "Impôt dû non encore décaissé. Le détail du poste « État — créditeur » "
            "figure à l'ETIC ; si seul le poste agrégé est disponible, la variable "
            "peut être laissée vide plutôt que renseignée avec le total « État »."
        ),
    ),
    Correspondance(
        variable="txt",
        libelle_us="Income Taxes — Total",
        poste_liasse="IMPÔTS SUR LES RÉSULTATS (XII)",
        tableau="CPC",
        comptes_cgnc="670 (6701 IS de l'exercice)",
        colonnes_source=("impots_sur_les_resultats",),
        fidelite="directe",
        remarque=(
            "Charge d'impôt de l'exercice. Variable centrale du dispositif : elle "
            "alimente trois des six ratios fiscaux métier (`tax_burden`, "
            "`effective_tax_rate`, `tax_to_sales`)."
        ),
    ),
    Correspondance(
        variable="xint",
        libelle_us="Interest and Related Expense — Total",
        poste_liasse="Charges d'intérêts",
        tableau="CPC — charges financières",
        comptes_cgnc="631",
        colonnes_source=("charges_interets",),
        fidelite="directe",
        remarque=(
            "Ne pas retenir le total des charges financières, qui englobe aussi "
            "les pertes de change et les dotations financières."
        ),
    ),
    Correspondance(
        variable="prcc_f",
        libelle_us="Price Close — Annual — Fiscal",
        poste_liasse="— (donnée de marché, hors liasse)",
        tableau="—",
        comptes_cgnc="—",
        colonnes_source=("cours_bourse_cloture",),
        fidelite="indisponible",
        remarque=(
            "Cours de clôture. Disponible pour les ~75 sociétés cotées à la Bourse "
            "de Casablanca, inexistant pour l'immense majorité des contribuables "
            "relevant d'une direction régionale. Laissé à NaN dans le cas général."
        ),
    ),
)

# Contrôle de cohérence au chargement du module : la table doit couvrir
# exactement les 28 variables attendues par le pipeline de features.
_VARS_MAPPEES = [c.variable for c in LIASSE_MAPPING]
assert _VARS_MAPPEES == list(RAW_ACCOUNTING_VARS) or sorted(_VARS_MAPPEES) == sorted(RAW_ACCOUNTING_VARS), (
    "LIASSE_MAPPING ne couvre pas exactement RAW_ACCOUNTING_VARS : "
    f"manquantes={set(RAW_ACCOUNTING_VARS) - set(_VARS_MAPPEES)}, "
    f"en trop={set(_VARS_MAPPEES) - set(RAW_ACCOUNTING_VARS)}"
)

# Colonnes du CSV liasse attendues, dans un ordre de lecture naturel
# (identifiants, bilan actif, bilan passif, CPC, hors liasse).
COLONNES_LIASSE: tuple[str, ...] = (
    COL_ID, COL_EXERCICE,
    # Bilan — actif
    "immobilisations_corporelles_brut", "immobilisations_financieres",
    "stocks", "creances_actif_circulant", "titres_valeurs_placement",
    "total_actif_circulant", "tresorerie_actif", "total_actif",
    # Bilan — passif
    "total_capitaux_propres", "capitaux_propres_assimiles",
    "reserve_legale", "autres_reserves", "report_a_nouveau", "resultats_en_instance",
    "dettes_financement", "dettes_financement_moins_1an", "provisions_durables",
    "fournisseurs_comptes_rattaches", "etat_impots_sur_resultats_dette",
    "total_passif_circulant", "tresorerie_passif",
    # CPC
    "chiffre_affaires", "achats_revendus_marchandises", "achats_consommes_matieres",
    "dotations_exploitation", "charges_interets", "impots_sur_les_resultats",
    "resultat_non_courant", "resultat_net_exercice",
    # Hors liasse (facultatif)
    "nombre_actions", "cours_bourse_cloture",
    "augmentation_dettes_financement", "augmentation_capital",
)

# Colonnes non comptables, facultatives : elles n'entrent pas dans le modèle mais
# sont recopiées telles quelles dans la sortie, pour que le vérificateur voie une
# raison sociale plutôt qu'un matricule dans le classement de risque.
COLONNES_DESCRIPTIVES: tuple[str, ...] = ("raison_sociale", "secteur", "commune")

# Sous-ensemble sans lequel le mapping n'a pas de sens : ce sont les postes qui
# alimentent les variables les plus contributives du modèle et les six ratios
# fiscaux métier. Le reste est souhaitable mais non bloquant.
COLONNES_LIASSE_ESSENTIELLES: tuple[str, ...] = (
    COL_ID, COL_EXERCICE,
    "total_actif", "total_capitaux_propres", "chiffre_affaires",
    "resultat_net_exercice", "impots_sur_les_resultats",
    "creances_actif_circulant", "stocks", "dotations_exploitation",
)


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------
def lire_liasse(chemin: str | Path, **kwargs) -> pd.DataFrame:
    """Lit un export de liasses fiscales et normalise les noms de colonnes.

    Tolère les variations d'écriture courantes dans les exports : majuscules,
    espaces, accents, tirets. « Chiffre d'Affaires » et « chiffre_affaires »
    désignent la même chose et doivent être acceptés tous les deux.
    """
    df = pd.read_csv(chemin, **kwargs)
    return normaliser_colonnes(df)


def normaliser_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les en-têtes : minuscules, sans accents, séparateur ``_``."""
    import unicodedata

    def _norm(nom: str) -> str:
        s = unicodedata.normalize("NFKD", str(nom))
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = s.lower().strip()
        for ch in (" ", "-", "'", "'", ".", "/"):
            s = s.replace(ch, "_")
        while "__" in s:
            s = s.replace("__", "_")
        return s.strip("_")

    return df.rename(columns={c: _norm(c) for c in df.columns})


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------
def mapper_liasse(
    df_liasse: pd.DataFrame,
    unite: str = "dirhams",
    mad_par_usd: float = MAD_PAR_USD_DEFAUT,
    convertir_en_usd: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Convertit un export de liasses fiscales en format d'entrée du modèle.

    Parameters
    ----------
    df_liasse :
        Un enregistrement par couple (contribuable, exercice), colonnes selon
        ``COLONNES_LIASSE``. Les colonnes absentes ne bloquent pas : les
        variables correspondantes seront NaN et comptées dans le rapport.
    unite :
        Unité de saisie des montants : ``"dirhams"``, ``"milliers_dirhams"`` ou
        ``"millions_dirhams"``.
    mad_par_usd :
        Taux de change à la clôture de l'exercice (cours Bank Al-Maghrib).
    convertir_en_usd :
        Si ``True`` (défaut), les montants sont ramenés en **millions de dollars**,
        unité du jeu d'entraînement. Passer à ``False`` revient à scorer en
        millions de dirhams : les 6 ratios fiscaux restent valides (invariants
        d'échelle) mais les 48 variables de niveau sont décalées par rapport à la
        distribution d'entraînement, ce qui biaise les scores.

    Returns
    -------
    (df_modele, rapport)
        ``df_modele`` a les colonnes ``gvkey``, ``fyear`` et les 28 variables
        comptables, prêt pour ``src.inference.score_dataframe``.
        ``rapport`` détaille la couverture obtenue variable par variable.
    """
    if unite not in UNITES_SUPPORTEES:
        raise ValueError(
            f"Unité inconnue : {unite!r}. Valeurs acceptées : {list(UNITES_SUPPORTEES)}"
        )

    df = normaliser_colonnes(df_liasse)

    manquantes_bloquantes = [c for c in (COL_ID, COL_EXERCICE) if c not in df.columns]
    if manquantes_bloquantes:
        raise ValueError(
            f"Colonnes identifiantes absentes : {', '.join(manquantes_bloquantes)}. "
            f"Elles sont indispensables au calcul des variables de trajectoire "
            f"(comparaison d'un contribuable à lui-même d'un exercice à l'autre)."
        )

    # --- Facteur d'échelle -------------------------------------------------
    # dirhams -> millions de dirhams -> millions de dollars
    facteur = UNITES_SUPPORTEES[unite] / 1e6
    if convertir_en_usd:
        if mad_par_usd <= 0:
            raise ValueError("`mad_par_usd` doit être strictement positif.")
        facteur = facteur / mad_par_usd

    # --- Reconstruction des 28 variables ----------------------------------
    out = pd.DataFrame(index=df.index)
    # `gvkey` doit être numérique : les features de trajectoire groupent dessus,
    # et un identifiant fiscal est de toute façon un entier.
    out["gvkey"] = pd.to_numeric(df[COL_ID], errors="coerce")
    if out["gvkey"].isna().any():
        # Repli : identifiant non numérique (ICE alphanumérique, matricule) —
        # on encode en catégories stables plutôt que d'échouer.
        out["gvkey"] = df[COL_ID].astype("category").cat.codes.astype("int64")
    out["fyear"] = pd.to_numeric(df[COL_EXERCICE], errors="coerce").astype("Int64")

    # Colonnes descriptives recopiées telles quelles (hors modèle)
    for c in COLONNES_DESCRIPTIVES:
        if c in df.columns:
            out[c] = df[c].values

    details = []
    for corr in LIASSE_MAPPING:
        if corr.calcul is not None:
            serie = corr.calcul(df)
        else:
            serie = _col(df, corr.colonnes_source[0])

        # `prcc_f` est un COURS UNITAIRE, pas un montant agrégé : il ne subit pas
        # la mise à l'échelle en millions, seulement la conversion de devise.
        if corr.variable == "prcc_f":
            serie = serie / mad_par_usd if convertir_en_usd else serie
        # `csho` est un NOMBRE DE TITRES ; Compustat l'exprime en millions.
        elif corr.variable == "csho":
            serie = serie / 1e6
        else:
            serie = serie * facteur

        out[corr.variable] = serie

        n_renseigne = int(serie.notna().sum())
        details.append({
            "variable": corr.variable,
            "poste_liasse": corr.poste_liasse,
            "tableau": corr.tableau,
            "fidelite": corr.fidelite,
            "lignes_renseignees": n_renseigne,
            "taux_couverture_pct": round(100 * n_renseigne / max(len(df), 1), 1),
        })

    # `pstk` : absence structurelle et non lacune de saisie. Le CGNC ne connaît
    # pas d'actions de préférence — la valeur correcte est 0, pas « inconnu ».
    if "actions_privilegiees" not in df.columns:
        out["pstk"] = 0.0
        for d in details:
            if d["variable"] == "pstk":
                d["lignes_renseignees"] = len(df)
                d["taux_couverture_pct"] = 100.0
                d["fidelite"] = "convention (0 — absent du CGNC)"

    # --- Rapport de couverture --------------------------------------------
    detail_df = pd.DataFrame(details)
    variables_vides = detail_df.loc[detail_df["lignes_renseignees"] == 0, "variable"].tolist()
    colonnes_reconnues = [c for c in COLONNES_LIASSE if c in df.columns]
    colonnes_ignorees = [
        c for c in df.columns
        if c not in COLONNES_LIASSE and c not in COLONNES_DESCRIPTIVES
        and not c.startswith("_")
    ]

    rapport = {
        "n_lignes": len(df),
        "n_contribuables": int(out["gvkey"].nunique()),
        "exercices": sorted(out["fyear"].dropna().unique().tolist()),
        "unite_saisie": unite,
        "mad_par_usd": mad_par_usd if convertir_en_usd else None,
        "devise_sortie": "millions USD" if convertir_en_usd else "millions MAD",
        "facteur_echelle": facteur,
        "n_variables_renseignees": int(28 - len(variables_vides)),
        "variables_vides": variables_vides,
        "taux_couverture_global_pct": round(float(detail_df["taux_couverture_pct"].mean()), 1),
        "colonnes_reconnues": colonnes_reconnues,
        "colonnes_ignorees": colonnes_ignorees,
        "detail": detail_df,
    }
    return out, rapport


# ---------------------------------------------------------------------------
# Diagnostics d'entrée
# ---------------------------------------------------------------------------
def verifier_liasse(df_liasse: pd.DataFrame) -> dict:
    """Contrôles de cohérence comptable AVANT scoring.

    Un fichier qui ne vérifie pas l'équilibre du bilan produira des scores
    ininterprétables : autant le dire tout de suite plutôt que de laisser
    l'utilisateur commenter des résultats construits sur une extraction fausse.
    Ces contrôles portent sur la **qualité de l'extraction**, ils ne constituent
    en rien un indice de fraude.
    """
    df = normaliser_colonnes(df_liasse)
    anomalies: list[str] = []
    avertissements: list[str] = []

    for col in (COL_ID, COL_EXERCICE):
        if col not in df.columns:
            anomalies.append(f"Colonne `{col}` absente — indispensable.")
        elif df[col].isna().any():
            anomalies.append(f"Colonne `{col}` : {int(df[col].isna().sum())} valeur(s) vide(s).")

    essentielles_absentes = [
        c for c in COLONNES_LIASSE_ESSENTIELLES if c not in df.columns
    ]
    if essentielles_absentes:
        avertissements.append(
            "Postes essentiels absents : " + ", ".join(essentielles_absentes)
            + ". Le scoring reste possible mais perd en pertinence."
        )

    # Équilibre du bilan : total actif = capitaux propres + assimilés + dettes.
    besoin = {"total_actif", "total_capitaux_propres", "total_passif_circulant",
              "dettes_financement", "tresorerie_passif"}
    if besoin.issubset(df.columns):
        passif = _somme(df, "total_capitaux_propres", "capitaux_propres_assimiles",
                        "dettes_financement", "provisions_durables",
                        "total_passif_circulant", "tresorerie_passif")
        actif = _col(df, "total_actif")
        ecart_rel = ((passif - actif).abs() / actif.abs().clip(lower=1)).fillna(0)
        n_desequilibre = int((ecart_rel > 0.01).sum())
        if n_desequilibre:
            avertissements.append(
                f"{n_desequilibre} ligne(s) où actif ≠ passif à plus de 1 % — "
                "vérifiez l'extraction (postes manquants ou colonne Brut/Net confondues)."
            )

    # Doublons (contribuable, exercice) : fausseraient les variations annuelles.
    if {COL_ID, COL_EXERCICE}.issubset(df.columns):
        n_dup = int(df.duplicated([COL_ID, COL_EXERCICE]).sum())
        if n_dup:
            anomalies.append(
                f"{n_dup} doublon(s) (identifiant_fiscal, exercice) — le calcul des "
                "variations d'un exercice à l'autre serait faux."
            )

    # Profondeur d'historique
    if {COL_ID, COL_EXERCICE}.issubset(df.columns) and not df.empty:
        exercices_par_contribuable = df.groupby(COL_ID)[COL_EXERCICE].nunique()
        part_mono = float((exercices_par_contribuable < 2).mean())
        if part_mono > 0.5:
            avertissements.append(
                f"{part_mono:.0%} des contribuables n'ont qu'un seul exercice : les "
                "20 variables de trajectoire seront majoritairement absentes. "
                "Fournir 3 exercices consécutifs par contribuable est l'idéal."
            )

    return {
        "scorable": not anomalies,
        "anomalies": anomalies,
        "avertissements": avertissements,
        "n_lignes": len(df),
    }


# ---------------------------------------------------------------------------
# Export du tableau de correspondance (pour le rapport et le dashboard)
# ---------------------------------------------------------------------------
def table_correspondance() -> pd.DataFrame:
    """Le tableau de mapping sous forme de DataFrame, prêt à être inséré dans
    le mémoire ou affiché dans le dashboard."""
    return pd.DataFrame([
        {
            "Variable modèle": c.variable,
            "Définition (Compustat)": c.libelle_us,
            "Poste de la liasse marocaine": c.poste_liasse,
            "Tableau": c.tableau,
            "Comptes CGNC": c.comptes_cgnc,
            "Fidélité": c.fidelite,
            "Remarque": c.remarque,
        }
        for c in LIASSE_MAPPING
    ])


def modele_csv_liasse(n_lignes: int = 0) -> pd.DataFrame:
    """Gabarit CSV vide (en-têtes seuls) au format liasse fiscale marocaine."""
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLONNES_LIASSE}).head(n_lignes)
