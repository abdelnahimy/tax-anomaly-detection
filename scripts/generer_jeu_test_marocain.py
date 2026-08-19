"""Génère un portefeuille simulé de contribuables de la DRI de Tétouan.

⚠️ DONNÉES ENTIÈREMENT FICTIVES — AUCUN CONTRIBUABLE RÉEL
=========================================================
Aucune entreprise, aucun identifiant fiscal, aucun montant de ce jeu ne
correspond à une réalité. Les données de l'administration fiscale sont couvertes
par le secret professionnel (article 246 du CGI) et n'ont pas été — ni ne
pouvaient être — utilisées. Voir `docs/01_cadrage_methodologique.md` §2.

À quoi sert ce jeu
------------------
À **tester l'application** : vérifier que la chaîne liasse → mapping → scoring →
restitution fonctionne à une échelle réaliste, que le classement se comporte
comme attendu, et que les diagnostics remontent les bonnes informations.

Il est calibré pour cela : ~220 contribuables sur 4 exercices, 14 secteurs
représentatifs du tissu économique de la région Tanger-Tétouan-Al Hoceïma, et
4 classes de taille — de la TPE à la grande entreprise. Un budget de contrôle
de 5 % y représente une quarantaine de dossiers, ce qui rend le raisonnement
« top k % » concret.

Les profils d'anomalie
----------------------
Une fraction des contribuables porte une **anomalie plantée**, construite à
partir des schémas que recherche effectivement un vérificateur : minoration du
chiffre d'affaires, majoration des charges, créances fictives, etc. La clé de
correction est écrite dans un **fichier séparé** (`cle_anomalies.csv`), pour que
le portefeuille soumis à l'application ne contienne aucune fuite.

⚠️ **Mise en garde méthodologique, à ne jamais omettre.** Les anomalies sont
plantées par nous, puis détectées par le modèle. Si celui-ci les retrouve, cela
démontre que **le signal traverse correctement la chaîne de traitement** — pas
que le modèle sait détecter la fraude fiscale marocaine. Une démonstration sur
données synthétiques n'est pas une validation. Les profils ci-dessous ont été
définis à partir de la doctrine du contrôle fiscal, et non à partir de la liste
des variables du modèle, ce qui limite la circularité sans la supprimer.

Contrainte de construction
--------------------------
Le bilan de chaque ligne est **équilibré par construction** : les capitaux
propres sont calculés en solde. Le fichier passe donc les contrôles de
`src.liasse_mapping.verifier_liasse()` sans aucun avertissement, y compris pour
les lignes portant une anomalie — une anomalie fiscale ne déséquilibre pas un
bilan, elle en déplace le contenu.

Usage
-----
    python scripts/generer_jeu_test_marocain.py
    python scripts/generer_jeu_test_marocain.py --contribuables 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.liasse_mapping import mapper_liasse, verifier_liasse

GRAINE = 20240817
EXERCICES = (2021, 2022, 2023, 2024)
N_CONTRIBUABLES = 220
PART_ANOMALIES = 0.08  # deliberately higher than a real fraud rate — see below

DOSSIER_SORTIE = Path(__file__).resolve().parent.parent / "data" / "exemples_marocains"

# ---------------------------------------------------------------------------
# Référentiel sectoriel
# ---------------------------------------------------------------------------
# Chaque secteur a sa propre physionomie comptable. C'est ce qui rend le jeu
# utile : un modèle qui se contenterait d'apprendre le secteur plutôt que
# l'anomalie se verrait immédiatement dans l'évaluation par strate.
#
# marge      : résultat net / chiffre d'affaires
# rotation   : chiffre d'affaires / total de bilan
# immob      : part de l'actif immobilisé net
# stocks     : part des stocks dans l'actif
# creances   : part des créances dans l'actif
# taux_is    : taux effectif d'imposition observé
SECTEURS = {
    "Industrie agroalimentaire":   dict(marge=(0.04, 0.09), rotation=(1.0, 1.5), immob=(0.38, 0.52), stocks=(0.14, 0.24), creances=(0.16, 0.26), taux_is=(0.26, 0.33)),
    "Peche et transformation":     dict(marge=(0.03, 0.10), rotation=(1.2, 1.8), immob=(0.34, 0.48), stocks=(0.10, 0.20), creances=(0.14, 0.24), taux_is=(0.24, 0.32)),
    "Textile et habillement":      dict(marge=(0.02, 0.07), rotation=(1.1, 1.6), immob=(0.30, 0.44), stocks=(0.20, 0.30), creances=(0.18, 0.28), taux_is=(0.25, 0.32)),
    "Batiment et travaux publics": dict(marge=(0.03, 0.08), rotation=(0.8, 1.3), immob=(0.26, 0.40), stocks=(0.10, 0.20), creances=(0.30, 0.42), taux_is=(0.26, 0.33)),
    "Commerce de gros":            dict(marge=(0.015, 0.045), rotation=(1.8, 3.0), immob=(0.14, 0.26), stocks=(0.24, 0.36), creances=(0.24, 0.36), taux_is=(0.26, 0.33)),
    "Commerce de detail":          dict(marge=(0.02, 0.06), rotation=(2.0, 3.5), immob=(0.18, 0.32), stocks=(0.26, 0.38), creances=(0.06, 0.16), taux_is=(0.25, 0.32)),
    "Hebergement et restauration": dict(marge=(0.05, 0.15), rotation=(0.4, 0.8), immob=(0.58, 0.74), stocks=(0.03, 0.09), creances=(0.08, 0.18), taux_is=(0.24, 0.32)),
    "Transport et logistique":     dict(marge=(0.03, 0.09), rotation=(0.9, 1.5), immob=(0.48, 0.64), stocks=(0.03, 0.09), creances=(0.20, 0.32), taux_is=(0.26, 0.33)),
    "Negoce de materiaux":         dict(marge=(0.03, 0.08), rotation=(1.5, 2.5), immob=(0.20, 0.34), stocks=(0.22, 0.34), creances=(0.22, 0.34), taux_is=(0.26, 0.33)),
    "Services aux entreprises":    dict(marge=(0.08, 0.18), rotation=(1.2, 2.0), immob=(0.14, 0.28), stocks=(0.01, 0.05), creances=(0.34, 0.48), taux_is=(0.27, 0.34)),
    "Promotion immobiliere":       dict(marge=(0.08, 0.20), rotation=(0.3, 0.7), immob=(0.16, 0.30), stocks=(0.36, 0.54), creances=(0.12, 0.24), taux_is=(0.24, 0.32)),
    "Sante et cliniques":          dict(marge=(0.06, 0.14), rotation=(0.6, 1.0), immob=(0.52, 0.68), stocks=(0.03, 0.08), creances=(0.16, 0.28), taux_is=(0.25, 0.32)),
    "Agriculture et agrumes":      dict(marge=(0.04, 0.12), rotation=(0.5, 0.9), immob=(0.50, 0.66), stocks=(0.10, 0.20), creances=(0.10, 0.20), taux_is=(0.14, 0.26)),
    "Industrie plastique":         dict(marge=(0.03, 0.09), rotation=(1.0, 1.6), immob=(0.40, 0.56), stocks=(0.14, 0.24), creances=(0.18, 0.28), taux_is=(0.26, 0.33)),
}

# Ressort de la DRI de Tétouan
COMMUNES = [
    "Tetouan", "Martil", "M'diq", "Fnideq", "Oued Laou", "Chefchaouen",
    "Ouezzane", "Larache", "Ksar El Kebir", "Bni Karrich", "Azla", "Jebha",
]

# Classes de taille : un portefeuille réel est très majoritairement composé de
# TPE et de petites entreprises. Le jeu doit refléter cette asymétrie.
# (libellé, chiffre d'affaires min, max, poids)
TAILLES = [
    ("TPE", 800_000, 10_000_000, 0.45),
    ("Petite entreprise", 10_000_000, 50_000_000, 0.33),
    ("Moyenne entreprise", 50_000_000, 200_000_000, 0.17),
    ("Grande entreprise", 200_000_000, 900_000_000, 0.05),
]

PREFIXES = [
    "AL", "DAR", "NOUR", "RIF", "DETROIT", "ATLAS", "MEDITERRANEE", "OUED",
    "JBEL", "BAB", "SALAM", "AMAL", "FATH", "MAGHREB", "ANDALUS", "TARIK",
    "ZITOUNE", "CHAMAL", "IBN BATOUTA", "KETAMA", "TALASSEMTANE", "GHOMARA",
]
SUFFIXES = ["SARL", "SA", "SARL AU", "SNC"]

MOTS_SECTEUR = {
    "Industrie agroalimentaire": ["CONSERVES", "AGRO", "OLIVES", "LAITERIE", "MINOTERIE"],
    "Peche et transformation": ["PECHERIES", "MAREE", "POISSONS", "CONGELATION"],
    "Textile et habillement": ["TEXTILE", "CONFECTION", "TISSAGE", "BONNETERIE"],
    "Batiment et travaux publics": ["CONSTRUCTION", "TRAVAUX", "BTP", "GENIE CIVIL"],
    "Commerce de gros": ["DISTRIBUTION", "GROS", "COMPTOIR", "NEGOCE"],
    "Commerce de detail": ["MARCHE", "SUPERETTE", "COMMERCE", "BOUTIQUE"],
    "Hebergement et restauration": ["HOTELIERE", "RESIDENCE", "RESTAURATION", "TOURISME"],
    "Transport et logistique": ["TRANSPORT", "LOGISTIQUE", "MESSAGERIE", "FRET"],
    "Negoce de materiaux": ["MATERIAUX", "CERAMIQUE", "QUINCAILLERIE", "SANITAIRE"],
    "Services aux entreprises": ["SERVICES", "CONSEIL", "INGENIERIE", "MAINTENANCE"],
    "Promotion immobiliere": ["IMMOBILIER", "PROMOTION", "RESIDENCES", "FONCIERE"],
    "Sante et cliniques": ["CLINIQUE", "POLYCLINIQUE", "LABORATOIRE", "IMAGERIE"],
    "Agriculture et agrumes": ["AGRICOLE", "AGRUMES", "DOMAINES", "PLANTATIONS"],
    "Industrie plastique": ["PLASTIQUE", "EMBALLAGE", "INJECTION", "POLYMERES"],
}

# ---------------------------------------------------------------------------
# Profils d'anomalie
# ---------------------------------------------------------------------------
# Définis à partir de la DOCTRINE DU CONTRÔLE FISCAL — les schémas que recherche
# un vérificateur — et non à partir de la liste des variables du modèle. Cette
# précaution limite la circularité de l'exercice : le modèle n'a pas été
# renseigné sur ce qu'il devait trouver.
PROFILS_ANOMALIE = {
    "minoration_ca": (
        "Minoration du chiffre d'affaires : ventes partiellement non déclarées, "
        "alors que les achats, les stocks et l'appareil productif restent "
        "cohérents avec l'activité réelle. Se trahit par une marge brute "
        "anormalement faible au regard du secteur."
    ),
    "majoration_charges": (
        "Majoration des charges déductibles (factures de complaisance, charges "
        "personnelles passées en frais généraux). Comprime le résultat imposable "
        "sans contrepartie à l'actif."
    ),
    "creances_fictives": (
        "Créances clients progressant très au-delà du chiffre d'affaires : "
        "ventes constatées mais jamais encaissées, ou clients fictifs destinés à "
        "soutenir un chiffre d'affaires de façade."
    ),
    "stocks_surevalues": (
        "Stocks dont la progression est sans rapport avec l'activité : "
        "surévaluation d'inventaire, ou rétention destinée à décaler la "
        "constatation du résultat."
    ),
    "is_anormalement_bas": (
        "Taux effectif d'imposition très inférieur à la norme sectorielle sans "
        "justification apparente : provisions non déductibles, reports "
        "déficitaires contestables, charges non justifiées."
    ),
    "accruals_eleves": (
        "Écart important entre le résultat comptable et la substance économique "
        "de l'exercice (accruals élevés) : le résultat déclaré repose sur des "
        "écritures d'ajustement plus que sur des flux."
    ),
    "rupture_trajectoire": (
        "Rupture brutale et inexpliquée de la trajectoire sur un exercice : "
        "effondrement de la marge ou saut du total de bilan, sans mouvement "
        "sectoriel correspondant."
    ),
}


# ---------------------------------------------------------------------------
# Construction d'une liasse
# ---------------------------------------------------------------------------
def _tirer_taille(rng: np.random.Generator) -> tuple[str, float]:
    poids = np.array([t[3] for t in TAILLES], dtype=float)
    index = rng.choice(len(TAILLES), p=poids / poids.sum())
    libelle, minimum, maximum, _ = TAILLES[index]
    # Distribution log-uniforme : beaucoup de petits, peu de gros, comme un
    # portefeuille réel.
    ca = float(np.exp(rng.uniform(np.log(minimum), np.log(maximum))))
    return libelle, ca


def _nom_entreprise(rng: np.random.Generator, secteur: str, commune: str) -> str:
    prefixe = rng.choice(PREFIXES)
    mot = rng.choice(MOTS_SECTEUR[secteur])
    suffixe = rng.choice(SUFFIXES)
    if rng.random() < 0.35:
        return f"{mot} {commune.upper()} {suffixe}"
    return f"{prefixe} {mot} {suffixe}"


def _signature_entreprise(rng: np.random.Generator, profil: dict) -> dict:
    """Tire une fois pour toutes la physionomie comptable d'un contribuable.

    Point important pour la qualité du jeu de test : la structure d'actif et le
    ratio d'achats d'une entreprise sont **stables dans le temps**. Les tirer à
    nouveau à chaque exercice créerait un bruit intra-entreprise si fort qu'il
    masquerait toute anomalie — et rendrait de surcroît le jeu irréaliste : le
    bilan d'une société ne se recompose pas au hasard d'une année sur l'autre.

    Les variables de trajectoire du modèle comparant un contribuable à
    lui-même, c'est précisément ce bruit-là qu'il fallait éliminer.
    """
    return {
        "part_immob": rng.uniform(*profil["immob"]),
        "part_stocks": rng.uniform(*profil["stocks"]),
        "part_creances": rng.uniform(*profil["creances"]),
        "ratio_achats": rng.uniform(0.55, 0.72),
        "part_dettes_fin": rng.uniform(0.05, 0.30),
        "part_fournisseurs": rng.uniform(0.10, 0.26),
        "vetuste": rng.uniform(1.45, 2.05),
        "taux_amortissement": rng.uniform(0.06, 0.11),
    }


def _construire_liasse(
    rng: np.random.Generator,
    profil: dict,
    signature: dict,
    ca: float,
    marge: float,
    rotation: float,
    taux_is: float,
    ajustements: dict,
) -> dict:
    """Construit une liasse équilibrée à partir de paramètres économiques.

    `ajustements` porte les multiplicateurs d'anomalie. Les capitaux propres
    sont calculés **en solde**, ce qui garantit l'équilibre du bilan quelles que
    soient les distorsions appliquées — une anomalie fiscale déplace le contenu
    d'un bilan, elle ne le déséquilibre pas.
    """
    ca_declare = ca * ajustements.get("ca", 1.0)
    actif_total = ca / rotation  # l'appareil productif reflète l'activité RÉELLE

    def _jitter(base: float, ampleur: float = 0.05) -> float:
        """Variation annuelle modérée autour de la signature de l'entreprise."""
        return base * rng.uniform(1 - ampleur, 1 + ampleur)

    # --- ACTIF -------------------------------------------------------------
    part_immob = _jitter(signature["part_immob"])
    part_stocks = _jitter(signature["part_stocks"]) * ajustements.get("stocks", 1.0)
    part_creances = _jitter(signature["part_creances"]) * ajustements.get("creances", 1.0)

    # Les parts ne peuvent pas dépasser l'actif : on renormalise si besoin.
    total_parts = part_immob + part_stocks + part_creances
    if total_parts > 0.94:
        facteur = 0.94 / total_parts
        part_immob *= facteur
        part_stocks *= facteur
        part_creances *= facteur

    immob_net = actif_total * part_immob
    immob_financieres = actif_total * rng.uniform(0.002, 0.025)
    immob_brut = immob_net * signature["vetuste"]  # avant amortissements
    stocks = actif_total * part_stocks
    creances = actif_total * part_creances
    titres_placement = actif_total * rng.uniform(0.0, 0.02)

    actif_circulant = stocks + creances + titres_placement
    tresorerie_actif = actif_total - immob_net - actif_circulant
    if tresorerie_actif < actif_total * 0.005:
        manque = actif_total * 0.01 - tresorerie_actif
        stocks = max(stocks - manque, actif_total * 0.005)
        actif_circulant = stocks + creances + titres_placement
        tresorerie_actif = actif_total - immob_net - actif_circulant

    # --- CPC ---------------------------------------------------------------
    resultat_net = ca_declare * marge * ajustements.get("resultat", 1.0)
    impots = max(resultat_net, 0.0) * taux_is * ajustements.get("is", 1.0)

    # Les achats suivent l'activité RÉELLE : c'est précisément l'incohérence
    # qui trahit une minoration du chiffre d'affaires.
    achats_totaux = ca * _jitter(signature["ratio_achats"], 0.04) * ajustements.get("achats", 1.0)
    achats_marchandises = achats_totaux * rng.uniform(0.55, 0.80)
    achats_matieres = achats_totaux - achats_marchandises

    dotations = (immob_brut * _jitter(signature["taux_amortissement"], 0.05)
                 * ajustements.get("dotations", 1.0))
    resultat_non_courant = ca_declare * rng.uniform(-0.008, 0.012)

    # --- PASSIF ------------------------------------------------------------
    cp_assimiles = actif_total * rng.uniform(0.0, 0.025)
    dettes_financement = actif_total * _jitter(signature["part_dettes_fin"], 0.08)
    dettes_financement_ct = dettes_financement * rng.uniform(0.10, 0.32)
    provisions_durables = actif_total * rng.uniform(0.0, 0.025)
    tresorerie_passif = actif_total * rng.uniform(0.005, 0.07)
    fournisseurs = actif_total * _jitter(signature["part_fournisseurs"], 0.06)
    etat_is_dette = impots * rng.uniform(0.15, 0.55)
    autres_dettes = actif_total * rng.uniform(0.02, 0.09)
    passif_circulant = fournisseurs + etat_is_dette + autres_dettes

    # Capitaux propres EN SOLDE → équilibre garanti
    capitaux_propres = (actif_total - cp_assimiles - dettes_financement
                        - provisions_durables - passif_circulant - tresorerie_passif)

    plancher = actif_total * 0.10
    if capitaux_propres < plancher:
        manque = plancher - capitaux_propres
        dettes_financement = max(dettes_financement - manque, actif_total * 0.02)
        capitaux_propres = (actif_total - cp_assimiles - dettes_financement
                            - provisions_durables - passif_circulant - tresorerie_passif)

    capital_social = capitaux_propres * rng.uniform(0.30, 0.58)
    reserve_legale = min(capital_social * 0.10, capitaux_propres * 0.10)
    autres_reserves = capitaux_propres * rng.uniform(0.02, 0.12)
    report_a_nouveau = (capitaux_propres - capital_social - reserve_legale
                        - autres_reserves - resultat_net)

    charges_interets = dettes_financement * rng.uniform(0.035, 0.070)

    return {
        # Bilan — actif
        "immobilisations_corporelles_brut": immob_brut,
        "immobilisations_financieres": immob_financieres,
        "stocks": stocks,
        "creances_actif_circulant": creances,
        "titres_valeurs_placement": titres_placement,
        "total_actif_circulant": actif_circulant,
        "tresorerie_actif": tresorerie_actif,
        "total_actif": actif_total,
        # Bilan — passif
        "total_capitaux_propres": capitaux_propres,
        "capitaux_propres_assimiles": cp_assimiles,
        "reserve_legale": reserve_legale,
        "autres_reserves": autres_reserves,
        "report_a_nouveau": report_a_nouveau,
        "resultats_en_instance": 0.0,
        "dettes_financement": dettes_financement,
        "dettes_financement_moins_1an": dettes_financement_ct,
        "provisions_durables": provisions_durables,
        "fournisseurs_comptes_rattaches": fournisseurs,
        "etat_impots_sur_resultats_dette": etat_is_dette,
        "total_passif_circulant": passif_circulant,
        "tresorerie_passif": tresorerie_passif,
        # CPC
        "chiffre_affaires": ca_declare,
        "achats_revendus_marchandises": achats_marchandises,
        "achats_consommes_matieres": achats_matieres,
        "dotations_exploitation": dotations,
        "charges_interets": charges_interets,
        "impots_sur_les_resultats": impots,
        "resultat_non_courant": resultat_non_courant,
        "resultat_net_exercice": resultat_net,
    }


def _ajustements_du_profil(profil_nom: str, rng: np.random.Generator) -> dict:
    """Traduit un profil d'anomalie en distorsions comptables."""
    if profil_nom == "minoration_ca":
        # CA amputé, appareil productif et achats inchangés
        return {"ca": rng.uniform(0.62, 0.80), "resultat": rng.uniform(0.35, 0.60)}
    if profil_nom == "majoration_charges":
        return {"achats": rng.uniform(1.15, 1.32), "resultat": rng.uniform(0.30, 0.55)}
    if profil_nom == "creances_fictives":
        return {"creances": rng.uniform(1.6, 2.4)}
    if profil_nom == "stocks_surevalues":
        return {"stocks": rng.uniform(1.7, 2.6)}
    if profil_nom == "is_anormalement_bas":
        return {"is": rng.uniform(0.15, 0.40)}
    if profil_nom == "accruals_eleves":
        # Résultat gonflé sans dotations correspondantes
        return {"resultat": rng.uniform(1.5, 2.2), "dotations": rng.uniform(0.35, 0.60)}
    if profil_nom == "rupture_trajectoire":
        return {"ca": rng.uniform(0.55, 0.72), "resultat": rng.uniform(0.20, 0.45),
                "creances": rng.uniform(1.3, 1.8)}
    return {}


# ---------------------------------------------------------------------------
# Génération du portefeuille
# ---------------------------------------------------------------------------
def generer(n_contribuables: int = N_CONTRIBUABLES) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(GRAINE)
    noms_secteurs = list(SECTEURS)
    noms_profils = list(PROFILS_ANOMALIE)

    # Les porteurs d'anomalie sont tirés au hasard, mais les profils leur sont
    # attribués en ROTATION plutôt qu'aléatoirement : un tirage aléatoire sur si
    # peu de contribuables donne des effectifs très déséquilibrés (un profil à
    # 15 lignes, un autre à 1), ce qui interdit toute évaluation par profil.
    n_anomalies = max(len(noms_profils), int(round(n_contribuables * PART_ANOMALIES)))
    tirage = rng.choice(n_contribuables, size=n_anomalies, replace=False).tolist()
    profil_par_indice = {
        indice: noms_profils[rang % len(noms_profils)]
        for rang, indice in enumerate(tirage)
    }
    indices_anomalies = set(tirage)

    lignes: list[dict] = []
    cle: list[dict] = []

    for i in range(n_contribuables):
        identifiant = 15_000_000 + i * 137 + int(rng.integers(0, 90))
        secteur = str(rng.choice(noms_secteurs))
        commune = str(rng.choice(COMMUNES))
        profil = SECTEURS[secteur]
        taille, ca_initial = _tirer_taille(rng)
        raison_sociale = _nom_entreprise(rng, secteur, commune)

        croissance = rng.uniform(-0.04, 0.16)
        marge_base = rng.uniform(*profil["marge"])
        rotation_base = rng.uniform(*profil["rotation"])
        taux_is_base = rng.uniform(*profil["taux_is"])
        signature = _signature_entreprise(rng, profil)

        # L'anomalie n'affecte pas toute l'histoire du contribuable : elle
        # apparaît à partir d'un exercice donné, comme dans un dossier réel.
        profil_anomalie = None
        exercice_debut = None
        if i in indices_anomalies:
            profil_anomalie = profil_par_indice[i]
            # Jamais dès le premier exercice : le contribuable doit avoir un
            # historique « propre » auquel se comparer, sans quoi les variables
            # de trajectoire n'ont rien à détecter.
            exercice_debut = int(rng.choice(EXERCICES[1:]))

        for exercice in EXERCICES:
            decalage = exercice - EXERCICES[0]
            # Bruit annuel volontairement modéré : au-delà, la variation
            # naturelle d'une entreprise sur elle-même masquerait l'anomalie
            # plantée, et le jeu ne testerait plus rien.
            ca = ca_initial * (1 + croissance) ** decalage * rng.uniform(0.98, 1.02)
            marge = marge_base * rng.uniform(0.93, 1.07)
            rotation = rotation_base * rng.uniform(0.97, 1.03)
            taux_is = taux_is_base * rng.uniform(0.97, 1.03)

            anomalie_active = (
                profil_anomalie is not None and exercice >= exercice_debut
            )
            ajustements = (
                _ajustements_du_profil(profil_anomalie, rng) if anomalie_active else {}
            )

            liasse = _construire_liasse(
                rng, profil, signature, ca, marge, rotation, taux_is, ajustements
            )
            lignes.append({
                "identifiant_fiscal": identifiant,
                "exercice": exercice,
                "raison_sociale": raison_sociale,
                "secteur": secteur,
                "commune": commune,
                **liasse,
            })
            cle.append({
                "identifiant_fiscal": identifiant,
                "exercice": exercice,
                "raison_sociale": raison_sociale,
                "secteur": secteur,
                "taille": taille,
                "anomalie": int(anomalie_active),
                "profil_anomalie": profil_anomalie if anomalie_active else "",
                "premier_exercice_anomalie": exercice_debut if anomalie_active else "",
            })

    portefeuille = pd.DataFrame(lignes)
    montants = portefeuille.select_dtypes("number").columns.drop(
        ["identifiant_fiscal", "exercice"]
    )
    portefeuille[montants] = portefeuille[montants].round(2)
    portefeuille = portefeuille.sort_values(
        ["identifiant_fiscal", "exercice"]
    ).reset_index(drop=True)

    correction = pd.DataFrame(cle).sort_values(
        ["identifiant_fiscal", "exercice"]
    ).reset_index(drop=True)

    return portefeuille, correction


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--contribuables", type=int, default=N_CONTRIBUABLES,
                         help=f"nombre de contribuables (défaut : {N_CONTRIBUABLES})")
    arguments = parseur.parse_args()

    portefeuille, correction = generer(arguments.contribuables)

    # Le jeu de test doit être irréprochable sur la forme : un fichier qui
    # déclenche lui-même des avertissements ne permet pas de tester l'outil.
    diagnostic = verifier_liasse(portefeuille)
    if not diagnostic["scorable"]:
        raise SystemExit("Anomalies bloquantes :\n- " + "\n- ".join(diagnostic["anomalies"]))
    if diagnostic["avertissements"]:
        raise SystemExit("Avertissements :\n- " + "\n- ".join(diagnostic["avertissements"]))

    _, rapport = mapper_liasse(portefeuille, unite="dirhams")

    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    chemin_portefeuille = DOSSIER_SORTIE / "portefeuille_dri_tetouan.csv"
    chemin_cle = DOSSIER_SORTIE / "cle_anomalies.csv"

    portefeuille.to_csv(chemin_portefeuille, index=False)
    correction.to_csv(chemin_cle, index=False)

    # Échantillon réduit, pour une prise en main rapide du dashboard.
    premiers = portefeuille["identifiant_fiscal"].drop_duplicates().head(12)
    echantillon = portefeuille[portefeuille["identifiant_fiscal"].isin(premiers)]
    chemin_echantillon = DOSSIER_SORTIE / "echantillon_demo.csv"
    echantillon.to_csv(chemin_echantillon, index=False)

    n_lignes_anomalie = int(correction["anomalie"].sum())
    n_contrib_anomalie = correction.loc[correction["anomalie"] == 1,
                                        "identifiant_fiscal"].nunique()

    print(f"Portefeuille : {chemin_portefeuille}")
    print(f"  {portefeuille['identifiant_fiscal'].nunique()} contribuables "
          f"× {portefeuille['exercice'].nunique()} exercices = {len(portefeuille)} lignes")
    print(f"  {len(SECTEURS)} secteurs · {portefeuille['commune'].nunique()} communes")
    print("  Bilan équilibré, aucun avertissement")
    print(f"  Couverture après mapping : {rapport['taux_couverture_global_pct']} % "
          f"({rapport['n_variables_renseignees']}/28 variables)")
    print(f"\nClé de correction : {chemin_cle}")
    print(f"  {n_contrib_anomalie} contribuables porteurs d'une anomalie "
          f"({n_contrib_anomalie / portefeuille['identifiant_fiscal'].nunique():.1%})")
    print(f"  {n_lignes_anomalie} lignes concernées "
          f"({n_lignes_anomalie / len(portefeuille):.1%})")
    print("\n  Répartition par profil :")
    repartition = correction.loc[correction["anomalie"] == 1, "profil_anomalie"].value_counts()
    for nom, compte in repartition.items():
        print(f"    {nom:<24} {compte:>3} lignes")
    print(f"\nÉchantillon de démonstration : {chemin_echantillon} ({len(echantillon)} lignes)")
    print("\n⚠️  Données entièrement fictives — aucun contribuable réel.")


if __name__ == "__main__":
    main()
