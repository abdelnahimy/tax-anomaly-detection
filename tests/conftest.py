"""Configuration pytest : rend le package `src` importable et fournit les
fixtures partagées.

Lancement de la suite depuis la racine du projet :

    python -m pytest              # tout
    python -m pytest -m "not slow"  # sans les tests qui chargent le modèle
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: test nécessitant le modèle entraîné (models/xgboost_v2_best.pkl)",
    )


@pytest.fixture(scope="session")
def racine() -> Path:
    return RACINE


@pytest.fixture
def liasse_minimale() -> pd.DataFrame:
    """Deux contribuables × trois exercices, bilan équilibré à la construction.

    Les montants sont ronds et vérifiables à la main : chaque test peut donc
    affirmer une valeur attendue exacte plutôt qu'une tolérance floue.
    """
    lignes = []
    for identifiant, base in ((11111111, 1_000_000.0), (22222222, 5_000_000.0)):
        for decalage, exercice in enumerate((2021, 2022, 2023)):
            k = base * (1 + 0.10 * decalage)
            lignes.append({
                "identifiant_fiscal": identifiant,
                "exercice": exercice,
                # --- actif ---
                "immobilisations_corporelles_brut": 80 * k,
                "immobilisations_financieres": 2 * k,
                "stocks": 15 * k,
                "creances_actif_circulant": 25 * k,
                "titres_valeurs_placement": 3 * k,
                "total_actif_circulant": 43 * k,   # stocks + créances + TVP
                "tresorerie_actif": 7 * k,
                "total_actif": 100 * k,
                # --- passif (somme = total_actif) ---
                "total_capitaux_propres": 40 * k,
                "capitaux_propres_assimiles": 2 * k,
                "reserve_legale": 4 * k,
                "autres_reserves": 6 * k,
                "report_a_nouveau": 10 * k,
                "resultats_en_instance": 1 * k,
                "dettes_financement": 20 * k,
                "dettes_financement_moins_1an": 5 * k,
                "provisions_durables": 3 * k,
                "fournisseurs_comptes_rattaches": 18 * k,
                "etat_impots_sur_resultats_dette": 2 * k,
                "total_passif_circulant": 30 * k,
                "tresorerie_passif": 5 * k,
                # --- CPC ---
                "chiffre_affaires": 120 * k,
                "achats_revendus_marchandises": 50 * k,
                "achats_consommes_matieres": 20 * k,
                "dotations_exploitation": 8 * k,
                "charges_interets": 1 * k,
                "impots_sur_les_resultats": 4 * k,
                "resultat_non_courant": 1 * k,
                "resultat_net_exercice": 12 * k,
            })
    return pd.DataFrame(lignes)


@pytest.fixture
def jar2020_minimal() -> pd.DataFrame:
    """Petit jeu au format JAR2020 brut, pour tester le pipeline d'inférence."""
    from src.config import RAW_ACCOUNTING_VARS

    rng = np.random.default_rng(7)
    lignes = []
    for gvkey in (1001, 1002, 1003, 1004):
        for exercice in (2010, 2011, 2012):
            ligne = {"gvkey": gvkey, "fyear": exercice}
            for i, var in enumerate(RAW_ACCOUNTING_VARS):
                ligne[var] = float(rng.uniform(1, 500) + i)
            lignes.append(ligne)
    return pd.DataFrame(lignes)
