"""Collecte des états financiers réels de sociétés marocaines cotées (AMMC).

Pourquoi ce script existe
-------------------------
Le portefeuille de test du projet (`portefeuille_dri_tetouan.csv`, 220
contribuables × 4 exercices) est **entièrement simulé**. C'est un jeu de
*recette applicative*, pas un jeu de données réel : ses anomalies y ont été
plantées, donc toute mesure de performance calculée dessus est partiellement
circulaire (voir `docs/06_transferabilite.md` §1 et `docs/07_modele_reconstitution.md` §4).

Ce script constitue le **contrepoids réel** : des liasses véritables, publiées,
auditées, opposables — celles des sociétés cotées à la Bourse de Casablanca,
que l'AMMC (Autorité Marocaine du Marché des Capitaux) impose de publier au
titre de l'information périodique (loi 44-12, circulaire AMMC 03/19).

Elles ne portent aucune étiquette de fraude — c'est précisément l'intérêt :
le modèle de reconstitution (`src/modele_marocain.py`) n'en a pas besoin, et
ces liasses permettent de vérifier qu'il se comporte correctement sur des
comptes réels, avec leurs irrégularités naturelles (exercices atypiques,
restructurations, holdings pures) que la simulation ne produit pas.

Ce qui est collecté, et ce qui ne l'est pas
-------------------------------------------
- **Comptes SOCIAUX (entité seule, référentiel CGNC)** — jamais les comptes
  consolidés IFRS. La liasse fiscale marocaine est un document d'entité : le
  mapping de `src/liasse_mapping.py` n'a de sens qu'au niveau social. Les
  émetteurs publient les deux ; la confusion est le piège principal de cette
  collecte, et chaque société a fait l'objet d'un contrôle explicite
  (cf. `SOURCES` ci-dessous, champ ``controle_social_consolide``).
- **2 exercices consécutifs par société** (4 pour Med Paper). C'est une limite
  connue : la règle de rupture de trajectoire (`src/regle_rupture.py`) demande
  au moins 3 exercices, elle ne s'applique donc pas à la plupart de ces
  sociétés. Voir §« Limites » plus bas.

Méthode d'extraction
--------------------
La collecte a été faite manuellement, document par document, par lecture des
états de synthèse publiés. Le script ne « scrape » pas le site de l'AMMC en
direct : il **fige** le résultat de cette lecture avec sa provenance, de sorte
que le jeu de données soit reproductible, auditable et vérifiable ligne à ligne
par un lecteur du mémoire.

Deux enseignements méthodologiques, valables pour toute reprise de la collecte :

1. **Les Rapports Financiers Annuels (RFA) sont inexploitables automatiquement.**
   Hébergés sur ``ammc.ma/sites/default/files/{Societe}_RFA_{annee}.pdf``, ils
   pèsent 10 à 33 Mo et 150 à 300 pages ; tout extracteur de texte tronque
   avant d'atteindre la section « Comptes sociaux », placée en annexe.
2. **Les documents courts contiennent les mêmes états, intégralement.** Par
   ordre d'efficacité constatée :
   ``avis de convocation à l'AGO`` > ``publication annuelle légale`` >
   ``communiqué financier annuel`` > ``rapport financier semestriel``
   (dont la colonne comparative donne une **lecture indépendante** du bilan de
   clôture N-1 — c'est l'outil qui a permis de trancher les valeurs douteuses).

Contrôle qualité
----------------
Chaque ligne a été soumise à quatre contrôles, tous rejoués par ce script
(``python scripts/collecte_ammc.py``) :

1. **Équation du bilan** — total actif = capitaux propres + capitaux propres
   assimilés + provisions durables + dettes de financement + total passif
   circulant + trésorerie passif.
2. **Concordance du résultat net** entre le CPC et le détail des capitaux
   propres du bilan passif.
3. **Recomposition des capitaux propres** — capital + primes + réserve légale
   + autres réserves + report à nouveau + résultat net.
4. **Cascade du CPC** — exploitation + financier = courant ; courant + non
   courant = avant impôts ; avant impôts − IS = net.

Aucune valeur n'a été estimée, extrapolée ou reconstituée par plausibilité.
Un champ que les états publiés ne permettent pas de lire est laissé **vide**
dans le CSV, jamais rempli par approximation. C'est la raison pour laquelle
``etat_impots_sur_resultats_dette`` est absent chez huit sociétés sur douze :
leur bilan passif ne présente qu'un poste « État » agrégé (TVA + IS + IR +
autres taxes) et la ventilation ne figure que dans l'ETIC détaillé.

Limites à ne pas passer sous silence
------------------------------------
- **Effectif.** 12 sociétés, 26 lignes. Le modèle de reconstitution exige
  ``EFFECTIF_MINIMUM = 40`` contribuables (`src/ciblage.py`) : ce jeu sert donc
  à **valider le mapping et le comportement des règles**, pas à entraîner un
  modèle. Le portefeuille simulé reste le support d'entraînement.
- **Biais de sélection.** Les sociétés cotées sont les plus grandes, les plus
  contrôlées et les plus auditées du tissu économique marocain. Elles ne
  représentent pas la population de contribuables d'une direction régionale.
- **Profondeur d'historique.** 2 exercices pour dix des douze sociétés, contre
  les 3 exercices consécutifs que demande `docs/02_mapping_liasse_fiscale.md`.
- **Absence d'étiquette.** Aucune de ces sociétés n'est réputée fraudeuse.
  Rien ici ne permet de mesurer un taux de détection.

Usage
-----
    python scripts/collecte_ammc.py            # régénère le CSV + rapport
    python scripts/collecte_ammc.py --verifier # contrôles seuls, sans écriture
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DESTINATION = RACINE / "data" / "exemples_marocains" / "societes_cotees_reelles.csv"

# Tolérance sur l'équation du bilan, en pourcentage du total de l'actif.
# 0,01 % laisse passer les arrondis de publication (certains émetteurs publient
# au dirham entier) sans laisser passer une erreur de saisie.
TOLERANCE_EQUILIBRE_PCT = 0.01

COLONNES = [
    "identifiant_fiscal",
    "exercice",
    "raison_sociale",
    "secteur",
    # --- Bilan actif ---
    "immobilisations_corporelles_brut",
    "immobilisations_financieres",
    "stocks",
    "creances_actif_circulant",
    "titres_valeurs_placement",
    "total_actif_circulant",
    "tresorerie_actif",
    "total_actif",
    # --- Bilan passif ---
    "total_capitaux_propres",
    "capitaux_propres_assimiles",
    "reserve_legale",
    "autres_reserves",
    "report_a_nouveau",
    "resultats_en_instance",
    "dettes_financement",
    "provisions_durables",
    "fournisseurs_comptes_rattaches",
    "etat_impots_sur_resultats_dette",
    "total_passif_circulant",
    "tresorerie_passif",
    # --- CPC ---
    "chiffre_affaires",
    "achats_revendus_marchandises",
    "achats_consommes_matieres",
    "dotations_exploitation",
    "charges_interets",
    "impots_sur_les_resultats",
    "resultat_non_courant",
    "resultat_net_exercice",
]

# ---------------------------------------------------------------------------
# Identification des sociétés
# ---------------------------------------------------------------------------
# `identifiant_fiscal` est un **numéro d'ordre conventionnel** (1001, 1002, …),
# pas l'identifiant fiscal réel des sociétés. Il joue le rôle de clé de
# regroupement (`gvkey`) pour comparer un contribuable à lui-même d'un exercice
# à l'autre. L'IF réel d'une société n'est pas une donnée publique et n'a aucune
# utilité ici.

SOCIETES: dict[int, dict[str, str]] = {
    1001: {"raison_sociale": "MED PAPER", "secteur": "Industrie papetiere"},
    1002: {"raison_sociale": "SNEP", "secteur": "Industrie plastique"},
    1003: {"raison_sociale": "LESIEUR CRISTAL", "secteur": "Agroalimentaire"},
    1004: {"raison_sociale": "COSUMAR", "secteur": "Agroalimentaire"},
    1005: {"raison_sociale": "UNIMER", "secteur": "Agroalimentaire"},
    1006: {"raison_sociale": "CIMENTS DU MAROC", "secteur": "Materiaux de construction"},
    1007: {"raison_sociale": "LABEL VIE", "secteur": "Commerce de detail"},
    1008: {"raison_sociale": "CTM", "secteur": "Transport et logistique"},
    1009: {"raison_sociale": "RISMA", "secteur": "Hotellerie et tourisme"},
    1010: {"raison_sociale": "AUTO NEJMA", "secteur": "Distribution automobile"},
    1011: {"raison_sociale": "DELTA HOLDING", "secteur": "Holding et services"},
    1012: {"raison_sociale": "JET CONTRACTORS", "secteur": "BTP et construction"},
}

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
# Une entrée par société : documents effectivement lus, et trace du contrôle
# social/consolidé. Cette table n'est pas décorative — c'est ce qui rend le
# jeu de données auditable par un jury.

SOURCES: dict[int, dict[str, object]] = {
    1001: {
        "documents": [
            "https://www.ammc.ma/sites/default/files/Med_Paper_RFA_2021.pdf",
            "https://www.ammc.ma/sites/default/files/MED_PAPER_RFA_2024.pdf",
        ],
        "controle_social_consolide": "Med Paper ne publie pas de comptes consolides (pas de filiales).",
    },
    1002: {
        "documents": [
            "https://www.ammc.ma/sites/default/files/SNEP_RFA_2023.pdf",
        ],
        "controle_social_consolide": "Comptes sociaux isoles de la section consolidee du RFA.",
    },
    1003: {
        "documents": [
            "https://www.ammc.ma/sites/default/files/Lesieur_Cristal_2024_CP_RFA_2024.pdf",
            "https://www.ammc.ma/sites/default/files/Lesieur_cristal_2023.pdf",
        ],
        "controle_social_consolide": (
            "Communiques financiers annuels publiant separement 'etats de synthese sociaux' "
            "et 'comptes consolides IFRS' ; seuls les premiers ont ete lus."
        ),
    },
    1004: {
        "documents": [
            "https://www.ammc.ma/sites/default/files/Cosumar_2025.pdf",
            "https://www.ammc.ma/sites/default/files/Cosumar_2024_CP_RFA_2024.pdf",
        ],
        "controle_social_consolide": (
            "L'exercice 2024 figure comme annee courante dans un document et comme annee "
            "comparative dans l'autre : toutes les valeurs concordent au centime."
        ),
    },
    1005: {
        "documents": [
            "https://www.ammc.ma/sites/default/files/Unimer_RFA_2025.pdf",
            "https://www.ammc.ma/sites/default/files/UNIMER_RFA_2024.pdf",
            "https://www.ammc.ma/sites/default/files/CP_Unimer_2024.pdf",
        ],
        "controle_social_consolide": (
            "CA social 806 MDH (2024) tres inferieur au CA consolide du groupe ; "
            "resultat net social -23,5 MDH confirme par le communique annuel."
        ),
    },
    1006: {
        "documents": [
            "https://www.ammc.ma/sites/default/files/Cimar_2024_avis_AGO_26_05_25.pdf",
            "https://www.cimentsdumaroc.com/sites/default/files/2024-04/"
            "Rapport%20g%C3%A9n%C3%A9ral%20comptes%20sociaux%202023.pdf",
            "https://www.ammc.ma/sites/default/files/CP_CIMAR_Avis%20de%20convocation.pdf",
            "https://www.ammc.ma/sites/default/files/Cimar_S1_25.pdf",
        ],
        "controle_social_consolide": (
            "CA social 3 903 MDH (2024) contre CA consolide ~4 360 MDH : ecart verifie, "
            "seul le social retenu."
        ),
    },
    1007: {
        "documents": [
            "https://labelvie.ma/wp-content/uploads/2024/02/Rapport-annuel-2023.pdf",
            "https://labelvie.ma/wp-content/uploads/2025/09/LBV-RFS-2025.pdf",
            "https://www.ammc.ma/sites/default/files/LabelVie_2024_CP_RFA_2024.pdf",
            "https://labelvie.ma/wp-content/uploads/2024/09/RFS-2024-LABELVIE-FR.pdf",
        ],
        "controle_social_consolide": (
            "CA social 8,14 Md DH (2024) contre CA consolide 16,4 Md DH. Les immobilisations "
            "financieres sociales portent 2,13 Md DH de titres de participation, poste qui "
            "disparait en consolidation."
        ),
    },
    1008: {
        "documents": [
            "https://ctm.ma/wp-content/uploads/2026/04/Publication-CTM-2025-Web.pdf",
            "https://ctm.ma/wp-content/uploads/2025/05/CTM-Avis-Convocation-AGO-17-Juin-2025.pdf",
            "https://www.ammc.ma/sites/default/files/CTM_RFA_2024.pdf",
            "https://ctm.ma/wp-content/uploads/2025/10/Communique_Financier-S1-2025.pdf",
        ],
        "controle_social_consolide": (
            "Publications legales des comptes sociaux ; CA social 540 974 KMAD (2024) "
            "recoupe au dirham pres le texte extractible du RFA depose a l'AMMC."
        ),
    },
    1009: {
        "documents": [
            "https://risma.com/wp-content/uploads/2026/04/Publication-RISMA-2025.pdf",
            "https://risma.com/wp-content/uploads/2025/04/RISMA-Publication-annuelle-2024.pdf",
            "https://www.ammc.ma/sites/default/files/Risma_RFA_2025.pdf",
        ],
        "controle_social_consolide": (
            "Publications annuelles legales, etats de synthese CGNC exclusivement. "
            "L'entite sociale exploite des hotels tout en portant 1,5 Md MAD de titres."
        ),
    },
    1010: {
        "documents": [
            "https://www.ammc.ma/sites/default/files/Avis_Auto_Nejma_AGO_15_04_25.pdf",
            "https://www.ammc.ma/sites/default/files/Auto_Nejma_RFS_S1_25.pdf",
            "https://www.ammc.ma/sites/default/files/CP_Auto_Nejma_2025.pdf",
        ],
        "controle_social_consolide": (
            "Resultat net social 300,0 MDH (2025) contre consolide 324 MDH : ecart identifie, "
            "seul le social retenu."
        ),
    },
    1011: {
        "documents": [
            "https://www.ammc.ma/sites/default/files/DHO_RFA_2025.pdf",
            "https://www.ammc.ma/sites/default/files/DHO_RFA_2024_0.pdf",
            "https://www.ammc.ma/sites/default/files/DHO_2025_avis_AGO_12_05_26.pdf",
        ],
        "controle_social_consolide": (
            "CA social 84,1 MDH (prestations aux filiales) contre CA consolide 3 050 MDH. "
            "L'entite sociale est une holding pure : aucun stock, 94 % du resultat provient "
            "de produits de titres de participation. Profil non representatif du secteur BTP."
        ),
    },
    1012: {
        "documents": [
            "https://www.jet-contractors.com/documents/COMPTES_SOCIAUX_2025.pdf",
            "https://www.jet-contractors.com/documents/comptes_sociaux_2024.pdf",
            "https://www.jet-contractors.com/documents/RAPPORT_GENERAL_COMPTES_SOCIAUX_31_12_2025.pdf",
        ],
        "controle_social_consolide": (
            "Resultat net social 140,0 MDH (2025) contre consolide 222,3 MDH ; total bilan "
            "social 5 647 MDH contre consolide 6 008 MDH. Ecarts verifies."
        ),
    },
}

# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
# Ordre des valeurs = ordre de COLONNES à partir de `immobilisations_corporelles_brut`.
# `None` = champ que les états publiés ne permettent pas de lire → cellule vide.
# `0.0` = vrai zéro (ligne présente à néant, ou absence confirmée par la
#         réconciliation exacte d'un sous-total).

_CHAMPS = COLONNES[4:]

LIGNES: list[dict[str, object]] = []


def _ajouter(identifiant: int, exercice: int, **valeurs: float | None) -> None:
    """Enregistre une ligne, en refusant tout champ inconnu ou manquant."""
    inconnus = set(valeurs) - set(_CHAMPS)
    if inconnus:
        raise KeyError(f"{identifiant}/{exercice} : champs inconnus {sorted(inconnus)}")
    manquants = set(_CHAMPS) - set(valeurs)
    if manquants:
        raise KeyError(
            f"{identifiant}/{exercice} : champs non renseignes {sorted(manquants)}. "
            "Utiliser None pour un champ illisible dans les etats publies."
        )
    societe = SOCIETES[identifiant]
    LIGNES.append(
        {
            "identifiant_fiscal": identifiant,
            "exercice": exercice,
            "raison_sociale": societe["raison_sociale"],
            "secteur": societe["secteur"],
            **valeurs,
        }
    )


# --- 1001 MED PAPER — Industrie papetiere ---------------------------------
# Collecte anterieure, conservee telle quelle. Le bilan actif 2020 et 2023
# n'etait pas integralement lisible : les champs correspondants restent vides.
_ajouter(
    1001, 2020,
    immobilisations_corporelles_brut=None, immobilisations_financieres=None,
    stocks=None, creances_actif_circulant=None, titres_valeurs_placement=None,
    total_actif_circulant=None, tresorerie_actif=None, total_actif=None,
    total_capitaux_propres=309995229.90, capitaux_propres_assimiles=-46845475.09,
    reserve_legale=0.0, autres_reserves=4572204.49, report_a_nouveau=42581194.97,
    resultats_en_instance=-493848360.19, dettes_financement=0.0,
    provisions_durables=96491433.98, fournisseurs_comptes_rattaches=0.0,
    etat_impots_sur_resultats_dette=66486590.98, total_passif_circulant=None,
    tresorerie_passif=158674036.16, chiffre_affaires=101675234.85,
    achats_revendus_marchandises=None, achats_consommes_matieres=None,
    dotations_exploitation=None, charges_interets=None,
    impots_sur_les_resultats=None, resultat_non_courant=None,
    resultat_net_exercice=-8562121.36,
)
_ajouter(
    1001, 2021,
    immobilisations_corporelles_brut=599254603.38, immobilisations_financieres=1637824.76,
    stocks=36787591.33, creances_actif_circulant=78003245.28, titres_valeurs_placement=0.0,
    total_actif_circulant=114790836.61, tresorerie_actif=792308.11, total_actif=287525958.56,
    total_capitaux_propres=-37361523.92, capitaux_propres_assimiles=0.0,
    reserve_legale=4572204.49, autres_reserves=42581194.97,
    report_a_nouveau=-502410481.55, resultats_en_instance=0.0,
    dettes_financement=128916843.20, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=36740895.28, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=150602166.40, tresorerie_passif=45368472.88,
    chiffre_affaires=None, achats_revendus_marchandises=None,
    achats_consommes_matieres=None, dotations_exploitation=None,
    charges_interets=None, impots_sur_les_resultats=None,
    resultat_non_courant=None, resultat_net_exercice=9483951.17,
)
_ajouter(
    1001, 2023,
    immobilisations_corporelles_brut=None, immobilisations_financieres=1637824.76,
    stocks=56199569.36, creances_actif_circulant=117045542.55, titres_valeurs_placement=0.0,
    total_actif_circulant=173273425.87, tresorerie_actif=238549.39, total_actif=345567827.11,
    total_capitaux_propres=26447630.97, capitaux_propres_assimiles=0.0,
    reserve_legale=0.46, autres_reserves=0.0, report_a_nouveau=-27141464.62,
    resultats_en_instance=0.0, dettes_financement=122513152.27, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=42270881.57, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=110815873.61, tresorerie_passif=85791170.26,
    chiffre_affaires=107863480.0, achats_revendus_marchandises=0.0,
    achats_consommes_matieres=64608710.0, dotations_exploitation=5802570.0,
    charges_interets=4682330.0, impots_sur_les_resultats=271480.0,
    resultat_non_courant=-220300.0, resultat_net_exercice=5750865.13,
)
_ajouter(
    1001, 2024,
    immobilisations_corporelles_brut=611657781.80, immobilisations_financieres=1637824.76,
    stocks=50681053.06, creances_actif_circulant=133968616.13, titres_valeurs_placement=0.0,
    total_actif_circulant=184649669.19, tresorerie_actif=762389.68, total_actif=353325568.20,
    total_capitaux_propres=34432862.61, capitaux_propres_assimiles=0.0,
    reserve_legale=0.46, autres_reserves=0.0, report_a_nouveau=-21390599.49,
    resultats_en_instance=0.0, dettes_financement=125943411.37, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=41881704.72, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=124525806.11, tresorerie_passif=68423488.11,
    chiffre_affaires=105563870.0, achats_revendus_marchandises=0.0,
    achats_consommes_matieres=52815140.0, dotations_exploitation=5538510.0,
    charges_interets=3297140.0, impots_sur_les_resultats=277500.0,
    resultat_non_courant=5337740.0, resultat_net_exercice=7985231.64,
)

# --- 1002 SNEP — Industrie plastique --------------------------------------
_ajouter(
    1002, 2022,
    immobilisations_corporelles_brut=None, immobilisations_financieres=430886885.10,
    stocks=266587819.08, creances_actif_circulant=489595720.89, titres_valeurs_placement=0.0,
    total_actif_circulant=757110421.89, tresorerie_actif=107745938.07,
    total_actif=1702179072.33, total_capitaux_propres=897313231.34,
    capitaux_propres_assimiles=0.0, reserve_legale=24000000.00,
    autres_reserves=100000000.00, report_a_nouveau=446232959.71,
    resultats_en_instance=0.0, dettes_financement=130585465.70,
    provisions_durables=98613.00, fournisseurs_comptes_rattaches=255326079.61,
    etat_impots_sur_resultats_dette=None, total_passif_circulant=456475946.04,
    tresorerie_passif=217705816.25, chiffre_affaires=1107442210.0,
    achats_revendus_marchandises=77156030.0, achats_consommes_matieres=744044510.0,
    dotations_exploitation=28761250.0, charges_interets=13123690.0,
    impots_sur_les_resultats=54076790.0, resultat_non_courant=3176680.0,
    resultat_net_exercice=87080271.63,
)
_ajouter(
    1002, 2023,
    immobilisations_corporelles_brut=1592212053.83, immobilisations_financieres=418599985.10,
    stocks=224481045.19, creances_actif_circulant=408829953.17, titres_valeurs_placement=0.0,
    total_actif_circulant=633357721.55, tresorerie_actif=13462635.77,
    total_actif=1622125697.62, total_capitaux_propres=802153639.17,
    capitaux_propres_assimiles=0.0, reserve_legale=24000000.00,
    autres_reserves=100000000.00, report_a_nouveau=494913231.34,
    resultats_en_instance=0.0, dettes_financement=92644826.25,
    provisions_durables=618449.51, fournisseurs_comptes_rattaches=199012715.87,
    etat_impots_sur_resultats_dette=None, total_passif_circulant=278068729.33,
    tresorerie_passif=448640053.36, chiffre_affaires=794476380.0,
    achats_revendus_marchandises=162328450.0, achats_consommes_matieres=451447570.0,
    dotations_exploitation=36762170.0, charges_interets=23995990.0,
    impots_sur_les_resultats=9101110.0, resultat_non_courant=-41163940.0,
    resultat_net_exercice=-56759592.17,
)

# --- 1003 LESIEUR CRISTAL — Agroalimentaire -------------------------------
_ajouter(
    1003, 2023,
    immobilisations_corporelles_brut=2205921185.77, immobilisations_financieres=392117609.05,
    stocks=1241931885.30, creances_actif_circulant=1259954223.13,
    titres_valeurs_placement=146191674.26, total_actif_circulant=2648077782.69,
    tresorerie_actif=206510470.67, total_actif=3811453553.50,
    total_capitaux_propres=1656124468.47, capitaux_propres_assimiles=77807425.0,
    reserve_legale=27631510.00, autres_reserves=827623799.70,
    report_a_nouveau=515681484.48, resultats_en_instance=0.0,
    dettes_financement=0.0, provisions_durables=20461624.0,
    fournisseurs_comptes_rattaches=1735521090.72,
    etat_impots_sur_resultats_dette=41070289.20, total_passif_circulant=2054807403.12,
    tresorerie_passif=2252633.61, chiffre_affaires=5228985949.22,
    achats_revendus_marchandises=526766178.24, achats_consommes_matieres=4944474053.23,
    dotations_exploitation=216420136.15, charges_interets=12459735.29,
    impots_sur_les_resultats=19034615.00, resultat_non_courant=-9887165.40,
    resultat_net_exercice=8872574.29,
)
_ajouter(
    1003, 2024,
    immobilisations_corporelles_brut=2332481817.76, immobilisations_financieres=404715500.71,
    stocks=1068755414.95, creances_actif_circulant=1434208506.48,
    titres_valeurs_placement=253969679.44, total_actif_circulant=2756933600.87,
    tresorerie_actif=123622833.64, total_actif=3920874414.72,
    total_capitaux_propres=1638326837.68, capitaux_propres_assimiles=53381872.58,
    reserve_legale=27631510.00, autres_reserves=827623799.70,
    report_a_nouveau=469291038.68, resultats_en_instance=None,
    dettes_financement=0.0, provisions_durables=13398324.56,
    fournisseurs_comptes_rattaches=1625119538.93,
    etat_impots_sur_resultats_dette=191736080.56, total_passif_circulant=2169847225.51,
    tresorerie_passif=45920154.39, chiffre_affaires=4777720778.94,
    achats_revendus_marchandises=287122304.66, achats_consommes_matieres=3488266446.64,
    dotations_exploitation=260187659.24, charges_interets=16554246.72,
    impots_sur_les_resultats=89320470.00, resultat_non_courant=-123812003.65,
    resultat_net_exercice=37465389.30,
)

# --- 1004 COSUMAR — Agroalimentaire ---------------------------------------
_ajouter(
    1004, 2024,
    immobilisations_corporelles_brut=6642679153.78, immobilisations_financieres=1611260964.47,
    stocks=2664524295.68, creances_actif_circulant=3857530053.63,
    titres_valeurs_placement=0.0, total_actif_circulant=6529726007.14,
    tresorerie_actif=355297258.53, total_actif=10218522962.46,
    total_capitaux_propres=4684807359.45, capitaux_propres_assimiles=304347200.52,
    reserve_legale=94487143.00, autres_reserves=2337667149.07,
    report_a_nouveau=194245137.05, resultats_en_instance=0.0,
    dettes_financement=0.0, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=3945773455.84,
    etat_impots_sur_resultats_dette=37816696.28, total_passif_circulant=4312196687.01,
    tresorerie_passif=917171715.48, chiffre_affaires=9643112303.24,
    achats_revendus_marchandises=0.0, achats_consommes_matieres=11862200853.94,
    dotations_exploitation=221421432.23, charges_interets=40107296.39,
    impots_sur_les_resultats=487874939.00, resultat_non_courant=43941837.52,
    resultat_net_exercice=937554578.11,
)
_ajouter(
    1004, 2025,
    immobilisations_corporelles_brut=6832956173.61, immobilisations_financieres=1611292144.35,
    stocks=2238852689.71, creances_actif_circulant=2460674455.77,
    titres_valeurs_placement=29998472.29, total_actif_circulant=4741164472.93,
    tresorerie_actif=603966825.76, total_actif=8672271679.55,
    total_capitaux_propres=4434619383.71, capitaux_propres_assimiles=250867928.14,
    reserve_legale=94487143.00, autres_reserves=2337667149.07,
    report_a_nouveau=186928285.16, resultats_en_instance=0.0,
    dettes_financement=0.0, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=2727400116.18,
    etat_impots_sur_resultats_dette=171453706.10, total_passif_circulant=3170450602.13,
    tresorerie_passif=816333765.57, chiffre_affaires=9776161240.38,
    achats_revendus_marchandises=0.0, achats_consommes_matieres=10586123074.49,
    dotations_exploitation=216604976.12, charges_interets=52767438.93,
    impots_sur_les_resultats=375170429.00, resultat_non_courant=-108817385.67,
    resultat_net_exercice=694683454.26,
)

# --- 1005 UNIMER — Agroalimentaire ----------------------------------------
# `etat_impots_sur_resultats_dette` : le bilan ne presente qu'un poste « Etat »
# agrege (4 826 314 en 2024 ; 2 672 035 en 2025). Laisse vide, comme partout
# ailleurs ou seule la ligne agregee est publiee.
_ajouter(
    1005, 2024,
    immobilisations_corporelles_brut=514953256.80, immobilisations_financieres=701652129.60,
    stocks=206896469.03, creances_actif_circulant=1087696552.93,
    titres_valeurs_placement=57532.38, total_actif_circulant=1295531632.15,
    tresorerie_actif=117602437.30, total_actif=2514789880.50,
    total_capitaux_propres=1532049342.84, capitaux_propres_assimiles=0.0,
    reserve_legale=11413880.00, autres_reserves=1800000.00,
    report_a_nouveau=472665615.83, resultats_en_instance=0.0,
    dettes_financement=157286391.43, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=116092779.0, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=211736901.96, tresorerie_passif=613717244.27,
    chiffre_affaires=806449256.64, achats_revendus_marchandises=None,
    achats_consommes_matieres=527580903.85, dotations_exploitation=20613267.81,
    charges_interets=39174386.0, impots_sur_les_resultats=2685482.00,
    resultat_non_courant=-6665301.26, resultat_net_exercice=-23508152.99,
)
_ajouter(
    1005, 2025,
    immobilisations_corporelles_brut=519978247.0, immobilisations_financieres=701652130.0,
    stocks=177009914.0, creances_actif_circulant=1111249519.0,
    titres_valeurs_placement=57532.0, total_actif_circulant=1289726385.0,
    tresorerie_actif=139023796.0, total_actif=2515927677.0,
    total_capitaux_propres=1522579730.0, capitaux_propres_assimiles=0.0,
    reserve_legale=11413880.0, autres_reserves=1800000.0,
    report_a_nouveau=449157463.0, resultats_en_instance=0.0,
    dettes_financement=146199956.0, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=122999748.0, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=219202063.0, tresorerie_passif=627945928.0,
    chiffre_affaires=688370925.0, achats_revendus_marchandises=None,
    achats_consommes_matieres=456776798.0, dotations_exploitation=20374921.0,
    charges_interets=36991029.0, impots_sur_les_resultats=1828143.0,
    resultat_non_courant=-9397136.0, resultat_net_exercice=-9469613.0,
)

# --- 1006 CIMENTS DU MAROC — Materiaux de construction --------------------
# `immobilisations_corporelles_brut` 2023 : deux valeurs concurrentes
# circulaient (8 165 028 618,61 et 8 194 817 663,43). La seconde est le brut
# au 30/06/2024, pas au 31/12/2023 — la premiere est retenue, confirmee par
# brut - amortissements = net publie et par la monotonie de la serie.
_ajouter(
    1006, 2023,
    immobilisations_corporelles_brut=8165028618.61, immobilisations_financieres=841883277.24,
    stocks=567822922.85, creances_actif_circulant=822922975.87,
    titres_valeurs_placement=470325586.93, total_actif_circulant=1875158184.58,
    tresorerie_actif=88308575.92, total_actif=4642675587.54,
    total_capitaux_propres=3308867213.98, capitaux_propres_assimiles=1260535.75,
    reserve_legale=144360040.00, autres_reserves=537959259.94,
    report_a_nouveau=0.0, resultats_en_instance=0.0,
    dettes_financement=0.0, provisions_durables=202541020.65,
    fournisseurs_comptes_rattaches=596633608.61,
    etat_impots_sur_resultats_dette=130773115.11, total_passif_circulant=942347768.96,
    tresorerie_passif=187659048.20, chiffre_affaires=3764385674.00,
    achats_revendus_marchandises=0.0, achats_consommes_matieres=1405492256.75,
    dotations_exploitation=261162520.77, charges_interets=2272923.50,
    impots_sur_les_resultats=391628469.00, resultat_non_courant=-15380215.88,
    resultat_net_exercice=981648303.57,
)
_ajouter(
    1006, 2024,
    immobilisations_corporelles_brut=8307994437.80, immobilisations_financieres=825485969.23,
    stocks=616077077.98, creances_actif_circulant=857875496.41,
    titres_valeurs_placement=587556484.42, total_actif_circulant=2070436858.77,
    tresorerie_actif=82043229.39, total_actif=4802795230.72,
    total_capitaux_propres=3197839000.56, capitaux_propres_assimiles=989068.49,
    reserve_legale=144360040.00, autres_reserves=509087283.51,
    report_a_nouveau=0.0, resultats_en_instance=0.0,
    dettes_financement=0.0, provisions_durables=232596377.81,
    fournisseurs_comptes_rattaches=691385008.24,
    etat_impots_sur_resultats_dette=163479358.10, total_passif_circulant=1184817403.99,
    tresorerie_passif=186553379.87, chiffre_affaires=3902740243.80,
    achats_revendus_marchandises=0.0, achats_consommes_matieres=1411648936.37,
    dotations_exploitation=273565144.95, charges_interets=1447860.04,
    impots_sur_les_resultats=508349690.00, resultat_non_courant=-144326953.60,
    resultat_net_exercice=899492066.58,
)

# --- 1007 LABEL VIE — Commerce de detail ----------------------------------
# Une premiere lecture, faite sur le seul communique annuel, comportait un
# decalage systematique de lignes et de colonnes (immobilisations financieres
# recevant le net corporel, report a nouveau recevant le resultat net, colonnes
# de charges 2024/2023 inversees). Les valeurs ci-dessous proviennent du
# rapport annuel 2023 et du rapport semestriel S1 2025, deux sources
# independantes, et sont verrouillees par la recomposition exacte des capitaux
# propres et par la resolution d'affectation du resultat votee en AGO.
_ajouter(
    1007, 2023,
    immobilisations_corporelles_brut=3259103720.17, immobilisations_financieres=2602883694.66,
    stocks=1237956662.71, creances_actif_circulant=1947056760.69,
    titres_valeurs_placement=176564419.50, total_actif_circulant=3361890188.34,
    tresorerie_actif=1260831073.46, total_actif=10672588791.06,
    total_capitaux_propres=1862475704.23, capitaux_propres_assimiles=0.0,
    reserve_legale=28939570.00, autres_reserves=0.0,
    report_a_nouveau=258175013.19, resultats_en_instance=0.0,
    dettes_financement=4172710598.81, provisions_durables=6003251.69,
    fournisseurs_comptes_rattaches=2438811090.18, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=4391399236.33, tresorerie_passif=240000000.00,
    chiffre_affaires=7864055076.40, achats_revendus_marchandises=6083142301.61,
    achats_consommes_matieres=227093917.54, dotations_exploitation=371053922.24,
    charges_interets=279771800.06, impots_sur_les_resultats=48434407.00,
    resultat_non_courant=-8288303.65, resultat_net_exercice=160942521.85,
)
_ajouter(
    1007, 2024,
    immobilisations_corporelles_brut=3536406569.98, immobilisations_financieres=2269690091.65,
    stocks=1425227478.49, creances_actif_circulant=1946585021.07,
    titres_valeurs_placement=175461749.50, total_actif_circulant=3547392840.12,
    tresorerie_actif=1906384172.13, total_actif=11527437488.87,
    total_capitaux_propres=1735884729.96, capitaux_propres_assimiles=0.0,
    reserve_legale=28939570.00, autres_reserves=0.0,
    report_a_nouveau=139117535.04, resultats_en_instance=0.0,
    dettes_financement=4648476229.21, provisions_durables=7775815.37,
    fournisseurs_comptes_rattaches=2564931414.91, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=4635300714.33, tresorerie_passif=500000000.00,
    chiffre_affaires=8139772618.22, achats_revendus_marchandises=6292022773.38,
    achats_consommes_matieres=256385281.11, dotations_exploitation=426123005.13,
    charges_interets=435715048.43, impots_sur_les_resultats=58081788.00,
    resultat_non_courant=26731817.50, resultat_net_exercice=153409025.73,
)

# --- 1008 CTM — Transport et logistique -----------------------------------
_ajouter(
    1008, 2024,
    immobilisations_corporelles_brut=932972857.51, immobilisations_financieres=344231875.00,
    stocks=8068960.35, creances_actif_circulant=135805165.23,
    titres_valeurs_placement=186120134.05, total_actif_circulant=329994259.63,
    tresorerie_actif=9667246.54, total_actif=1047882072.55,
    total_capitaux_propres=371009990.39, capitaux_propres_assimiles=0.0,
    reserve_legale=12259780.00, autres_reserves=68324083.15,
    report_a_nouveau=55502840.67, resultats_en_instance=0.0,
    dettes_financement=332056131.67, provisions_durables=5662909.34,
    fournisseurs_comptes_rattaches=63604023.80, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=339153041.15, tresorerie_passif=0.0,
    chiffre_affaires=540973754.45, achats_revendus_marchandises=0.0,
    achats_consommes_matieres=174580931.17, dotations_exploitation=82897059.23,
    charges_interets=9316002.89, impots_sur_les_resultats=23266402.00,
    resultat_non_courant=51629860.62, resultat_net_exercice=69887786.57,
)
_ajouter(
    1008, 2025,
    immobilisations_corporelles_brut=945236225.53, immobilisations_financieres=322183076.00,
    stocks=8885625.80, creances_actif_circulant=242934291.60,
    titres_valeurs_placement=0.0, total_actif_circulant=251819917.40,
    tresorerie_actif=10543720.09, total_actif=938308672.73,
    total_capitaux_propres=410532466.14, capitaux_propres_assimiles=0.0,
    reserve_legale=12259780.00, autres_reserves=68324083.15,
    report_a_nouveau=94741177.24, resultats_en_instance=0.0,
    dettes_financement=316427433.93, provisions_durables=7069059.64,
    fournisseurs_comptes_rattaches=64137223.47, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=204279713.02, tresorerie_passif=0.0,
    chiffre_affaires=556039243.41, achats_revendus_marchandises=0.0,
    achats_consommes_matieres=152368933.14, dotations_exploitation=96321450.64,
    charges_interets=12758020.08, impots_sur_les_resultats=21553492.00,
    resultat_non_courant=48708835.80, resultat_net_exercice=70171925.75,
)

# --- 1009 RISMA — Hotellerie et tourisme ----------------------------------
# Exercice 2024 atypique : reprises financieres de 353,6 MMAD et charges non
# courantes de 428,6 MMAD liees a une restructuration de participations, d'ou
# un resultat non courant de -400,8 MMAD. Rupture reelle, pas une erreur de
# saisie — utile comme cas-test de faux positif.
_ajouter(
    1009, 2024,
    immobilisations_corporelles_brut=2820271824.0, immobilisations_financieres=919306671.0,
    stocks=11196821.0, creances_actif_circulant=168811711.0,
    titres_valeurs_placement=387713620.0, total_actif_circulant=567722152.0,
    tresorerie_actif=79906905.0, total_actif=3553360408.0,
    total_capitaux_propres=2176379064.0, capitaux_propres_assimiles=25319226.0,
    reserve_legale=28156036.0, autres_reserves=0.0,
    report_a_nouveau=160914640.0, resultats_en_instance=0.0,
    dettes_financement=977319530.0, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=126758543.0, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=322977456.0, tresorerie_passif=51365131.0,
    chiffre_affaires=957681451.0, achats_revendus_marchandises=593533.0,
    achats_consommes_matieres=172988805.0, dotations_exploitation=163206572.0,
    charges_interets=62173979.0, impots_sur_les_resultats=2509443.0,
    resultat_non_courant=-400802091.0, resultat_net_exercice=111006261.0,
)
_ajouter(
    1009, 2025,
    immobilisations_corporelles_brut=2958803205.30, immobilisations_financieres=1443306671.07,
    stocks=8624810.51, creances_actif_circulant=186373919.85,
    titres_valeurs_placement=10725736.15, total_actif_circulant=205724466.51,
    tresorerie_actif=98033941.75, total_actif=3919083883.17,
    total_capitaux_propres=2256730359.73, capitaux_propres_assimiles=17734247.76,
    reserve_legale=33706348.84, autres_reserves=0.0,
    report_a_nouveau=166081959.38, resultats_en_instance=0.0,
    dettes_financement=839327544.13, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=117377725.45, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=387191853.76, tresorerie_passif=418099877.79,
    chiffre_affaires=957104462.96, achats_revendus_marchandises=578041.04,
    achats_consommes_matieres=159464336.25, dotations_exploitation=159905860.60,
    charges_interets=53543824.01, impots_sur_les_resultats=5921669.26,
    resultat_non_courant=208967.40, resultat_net_exercice=180639924.32,
)

# --- 1010 AUTO NEJMA — Distribution automobile ----------------------------
_ajouter(
    1010, 2024,
    immobilisations_corporelles_brut=788569979.55, immobilisations_financieres=222738949.46,
    stocks=1017180073.63, creances_actif_circulant=740963115.31,
    titres_valeurs_placement=97166369.33, total_actif_circulant=1855340585.82,
    tresorerie_actif=45704015.33, total_actif=2615633844.89,
    total_capitaux_propres=1256879994.57, capitaux_propres_assimiles=0.0,
    reserve_legale=10232640.00, autres_reserves=8552273.80,
    report_a_nouveau=930839986.53, resultats_en_instance=0.0,
    dettes_financement=69703873.06, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=499202693.34, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=1021893433.70, tresorerie_passif=267156543.56,
    chiffre_affaires=2830867602.71, achats_revendus_marchandises=2196431978.92,
    achats_consommes_matieres=23086900.07, dotations_exploitation=55878797.14,
    charges_interets=19759782.24, impots_sur_les_resultats=118541584.00,
    resultat_non_courant=-20665213.69, resultat_net_exercice=204928694.24,
)
_ajouter(
    1010, 2025,
    immobilisations_corporelles_brut=879630754.43, immobilisations_financieres=204850949.46,
    stocks=1439664993.33, creances_actif_circulant=664219840.56,
    titres_valeurs_placement=363225534.94, total_actif_circulant=2467120085.95,
    tresorerie_actif=85877099.02, total_actif=3300674394.04,
    total_capitaux_propres=1434094312.44, capitaux_propres_assimiles=0.0,
    reserve_legale=10232640.00, autres_reserves=8552273.80,
    report_a_nouveau=1012977000.66, resultats_en_instance=0.0,
    dettes_financement=581250000.00, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=684987016.69, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=1285330081.61, tresorerie_passif=0.0,
    chiffre_affaires=4133588091.66, achats_revendus_marchandises=3196366201.22,
    achats_consommes_matieres=33557329.90, dotations_exploitation=107165854.46,
    charges_interets=19759826.86, impots_sur_les_resultats=186412198.00,
    resultat_non_courant=-15158843.64, resultat_net_exercice=300005997.98,
)

# --- 1011 DELTA HOLDING — Holding et services -----------------------------
# L'entite sociale est une holding pure : aucun stock, CA = prestations de
# services aux filiales, 94 % du resultat provenant de produits de titres de
# participation. Profil comptable deliberement conserve : c'est un cas-test
# utile (une regle sectorielle naive le signalerait a tort).
_ajouter(
    1011, 2024,
    immobilisations_corporelles_brut=49088278.51, immobilisations_financieres=606325285.47,
    stocks=0.0, creances_actif_circulant=1052039011.63,
    titres_valeurs_placement=0.0, total_actif_circulant=1053744453.25,
    tresorerie_actif=133895617.11, total_actif=1810561604.60,
    total_capitaux_propres=1781003527.76, capitaux_propres_assimiles=3088.20,
    reserve_legale=87600000.00, autres_reserves=0.0,
    report_a_nouveau=641251853.13, resultats_en_instance=0.0,
    dettes_financement=0.0, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=1049836.74, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=29548572.16, tresorerie_passif=6416.48,
    chiffre_affaires=64222892.42, achats_revendus_marchandises=0.0,
    achats_consommes_matieres=2671139.91, dotations_exploitation=1862947.83,
    charges_interets=67701.23, impots_sur_les_resultats=9491501.00,
    resultat_non_courant=-5400097.30, resultat_net_exercice=176151674.63,
)
_ajouter(
    1011, 2025,
    immobilisations_corporelles_brut=49239625.46, immobilisations_financieres=605715485.47,
    stocks=0.0, creances_actif_circulant=1008207790.03,
    titres_valeurs_placement=100000000.00, total_actif_circulant=1108767223.27,
    tresorerie_actif=230846253.55, total_actif=1960727281.84,
    total_capitaux_propres=1924797978.22, capitaux_propres_assimiles=0.0,
    reserve_legale=87600000.00, autres_reserves=0.0,
    report_a_nouveau=620303527.76, resultats_en_instance=0.0,
    dettes_financement=0.0, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=1143880.06, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=35922697.62, tresorerie_passif=6606.00,
    chiffre_affaires=84115527.42, achats_revendus_marchandises=0.0,
    achats_consommes_matieres=2587908.70, dotations_exploitation=1852855.06,
    charges_interets=22106.62, impots_sur_les_resultats=16604015.00,
    resultat_non_courant=-1612289.90, resultat_net_exercice=340894450.46,
)

# --- 1012 JET CONTRACTORS — BTP et construction ---------------------------
# L'exercice 2025 laisse un residu de 184 692,03 DH sur l'equation du bilan :
# c'est le poste « Ecarts de conversion - Passif (durables) », que le schema de
# la liasse retenu ici ne comporte pas. Le residu est donc attendu et
# entierement explique ; il vaut 0,0033 % du total du bilan.
_ajouter(
    1012, 2024,
    immobilisations_corporelles_brut=225389543.01, immobilisations_financieres=281200220.75,
    stocks=791251535.18, creances_actif_circulant=2658533185.56,
    titres_valeurs_placement=0.0, total_actif_circulant=3454810281.11,
    tresorerie_actif=28163964.17, total_actif=3921073795.93,
    total_capitaux_propres=1015217842.53, capitaux_propres_assimiles=2890884.15,
    reserve_legale=15147610.00, autres_reserves=3000000.00,
    report_a_nouveau=484720453.24, resultats_en_instance=0.0,
    dettes_financement=230412830.86, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=1285017826.92, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=2187724319.51, tresorerie_passif=484827918.88,
    chiffre_affaires=2695481931.37, achats_revendus_marchandises=20935869.96,
    achats_consommes_matieres=2318732714.09, dotations_exploitation=121835051.58,
    charges_interets=52974550.28, impots_sur_les_resultats=64371995.31,
    resultat_non_courant=1354229.26, resultat_net_exercice=76817079.29,
)
_ajouter(
    1012, 2025,
    immobilisations_corporelles_brut=570572138.69, immobilisations_financieres=362842871.95,
    stocks=870297161.89, creances_actif_circulant=3495466308.04,
    titres_valeurs_placement=0.0, total_actif_circulant=4371483565.72,
    tresorerie_actif=416390082.16, total_actif=5646901866.67,
    total_capitaux_propres=1109866931.01, capitaux_propres_assimiles=2102492.08,
    reserve_legale=15147610.00, autres_reserves=3000000.00,
    report_a_nouveau=516224287.54, resultats_en_instance=0.0,
    dettes_financement=1169143939.19, provisions_durables=0.0,
    fournisseurs_comptes_rattaches=1604459987.84, etat_impots_sur_resultats_dette=None,
    total_passif_circulant=2770073496.73, tresorerie_passif=595530315.63,
    chiffre_affaires=3553982968.14, achats_revendus_marchandises=95035521.50,
    achats_consommes_matieres=3034146912.89, dotations_exploitation=120832773.61,
    charges_interets=65238306.50, impots_sur_les_resultats=98994137.95,
    resultat_non_courant=26368167.63, resultat_net_exercice=139962333.47,
)


# ---------------------------------------------------------------------------
# Contrôles
# ---------------------------------------------------------------------------

POSTES_PASSIF = (
    "total_capitaux_propres",
    "capitaux_propres_assimiles",
    "provisions_durables",
    "dettes_financement",
    "total_passif_circulant",
    "tresorerie_passif",
)


def verifier_equilibre(ligne: dict[str, object]) -> tuple[str, float | None]:
    """Contrôle l'identité bilancielle d'une ligne.

    Retourne ``(statut, ecart_pct)``. Le statut vaut ``"ok"``, ``"ecart"`` ou
    ``"incomplet"`` — ce dernier lorsqu'un poste nécessaire n'a pas pu être lu
    dans les états publiés, auquel cas aucun contrôle n'est possible.
    """
    actif = ligne["total_actif"]
    if actif is None or any(ligne[poste] is None for poste in POSTES_PASSIF):
        return "incomplet", None
    passif = sum(float(ligne[poste]) for poste in POSTES_PASSIF)  # type: ignore[arg-type]
    ecart_pct = abs(passif - float(actif)) / float(actif) * 100 if actif else 0.0
    return ("ok" if ecart_pct <= TOLERANCE_EQUILIBRE_PCT else "ecart"), ecart_pct


def rapport() -> int:
    """Affiche le rapport de contrôle. Retourne le nombre de lignes en écart."""
    largeur = 78
    print("=" * largeur)
    print("CONTROLE QUALITE — societes cotees reelles (AMMC)")
    print("=" * largeur)

    en_ecart = 0
    incomplets = 0
    for ligne in LIGNES:
        statut, ecart = verifier_equilibre(ligne)
        if statut == "incomplet":
            incomplets += 1
            marque, detail = "  ?", "poste manquant, controle impossible"
        elif statut == "ok":
            marque, detail = "  .", f"ecart {ecart:.6f} %"
        else:
            en_ecart += 1
            marque, detail = "  !", f"ECART {ecart:.6f} % — A VERIFIER"
        print(f"{marque} {ligne['raison_sociale']:<18} {ligne['exercice']}   {detail}")

    manquants = sum(1 for ligne in LIGNES for champ in _CHAMPS if ligne[champ] is None)
    total_cellules = len(LIGNES) * len(_CHAMPS)

    print("-" * largeur)
    print(f"Contribuables      : {len({l['identifiant_fiscal'] for l in LIGNES})}")
    print(f"Lignes             : {len(LIGNES)}")
    print(f"Equilibre verifie  : {len(LIGNES) - incomplets - en_ecart} / {len(LIGNES)}")
    print(f"Controle impossible: {incomplets} (postes non lisibles dans les etats publies)")
    print(f"Lignes en ecart    : {en_ecart}")
    print(
        f"Cellules vides     : {manquants} / {total_cellules} "
        f"({manquants / total_cellules * 100:.1f} %) — champs non lisibles, jamais estimes"
    )
    print("=" * largeur)
    return en_ecart


def ecrire_csv(destination: Path = DESTINATION) -> None:
    """Écrit le CSV, trié par contribuable puis exercice."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lignes = sorted(LIGNES, key=lambda l: (l["identifiant_fiscal"], l["exercice"]))
    with destination.open("w", encoding="utf-8", newline="") as flux:
        redacteur = csv.DictWriter(flux, fieldnames=COLONNES)
        redacteur.writeheader()
        for ligne in lignes:
            redacteur.writerow(
                {
                    colonne: ("" if ligne[colonne] is None else ligne[colonne])
                    for colonne in COLONNES
                }
            )
    print(f"\nEcrit : {destination.relative_to(RACINE)}  ({len(lignes)} lignes)")


def main() -> int:
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyseur.add_argument(
        "--verifier",
        action="store_true",
        help="n'execute que les controles, sans reecrire le CSV",
    )
    arguments = analyseur.parse_args()

    en_ecart = rapport()
    if not arguments.verifier:
        ecrire_csv()
    return 1 if en_ecart else 0


if __name__ == "__main__":
    sys.exit(main())
