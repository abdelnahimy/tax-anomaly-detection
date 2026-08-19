"""Tests de l'orchestrateur (`src.ciblage`).

L'enjeu principal ici n'est pas la qualité des scores sous-jacents — déjà
couverte par `test_regle_rupture.py` et `test_modele_marocain.py` — mais le
comportement de l'orchestrateur lui-même : bascule propre quand le volet IA
n'est pas disponible, cohérence du classement produit, et lisibilité du motif
affiché à la brigade.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ciblage import METHODES, analyser_portefeuille
from src.liasse_mapping import mapper_liasse
from src.modele_marocain import EFFECTIF_MINIMUM


def _en_modele(liasse: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return mapper_liasse(liasse, **kwargs)[0]


@pytest.fixture
def petit_portefeuille() -> pd.DataFrame:
    """Sous le seuil `EFFECTIF_MINIMUM` : le volet IA doit être écarté."""
    lignes = []
    for i in range(5):
        k = 1_000_000.0
        lignes.append({
            "identifiant_fiscal": 70000 + i, "exercice": 2023,
            "secteur": "Commerce de gros",
            "total_actif": 100 * k, "total_actif_circulant": 43 * k,
            "tresorerie_actif": 7 * k, "stocks": 15 * k,
            "creances_actif_circulant": 25 * k, "titres_valeurs_placement": 3 * k,
            "immobilisations_corporelles_brut": 80 * k,
            "immobilisations_financieres": 2 * k,
            "total_capitaux_propres": 40 * k, "capitaux_propres_assimiles": 2 * k,
            "reserve_legale": 4 * k, "autres_reserves": 6 * k,
            "report_a_nouveau": 10 * k, "resultats_en_instance": 0.0,
            "dettes_financement": 20 * k, "provisions_durables": 3 * k,
            "fournisseurs_comptes_rattaches": 18 * k,
            "etat_impots_sur_resultats_dette": 2 * k,
            "total_passif_circulant": 30 * k, "tresorerie_passif": 5 * k,
            "chiffre_affaires": 120 * k, "achats_revendus_marchandises": 50 * k,
            "achats_consommes_matieres": 20 * k, "dotations_exploitation": 8 * k,
            "charges_interets": 1 * k, "impots_sur_les_resultats": 4 * k,
            "resultat_non_courant": 1 * k, "resultat_net_exercice": 12 * k,
        })
    return _en_modele(pd.DataFrame(lignes))


@pytest.fixture(scope="module")
def grand_portefeuille() -> pd.DataFrame:
    chemin = (
        Path(__file__).resolve().parent.parent
        / "data" / "exemples_marocains" / "portefeuille_dri_tetouan.csv"
    )
    if not chemin.exists():
        pytest.skip("jeu de test absent — python scripts/generer_jeu_test_marocain.py")
    return _en_modele(pd.read_csv(chemin), unite="dirhams")


class TestMethodeInconnue:

    def test_leve_value_error(self, petit_portefeuille):
        with pytest.raises(ValueError, match="Méthode inconnue"):
            analyser_portefeuille(petit_portefeuille, methode="nimporte_quoi")


class TestBasculeSiIaIndisponible:

    def test_petit_portefeuille_bascule_sur_les_regles(self, petit_portefeuille):
        assert petit_portefeuille["gvkey"].nunique() < EFFECTIF_MINIMUM

        classement, rapport = analyser_portefeuille(petit_portefeuille, methode="combine")

        assert rapport["ia_disponible"] is False
        assert rapport["methode_appliquee"] == "regles"
        assert "ia_indisponible_motif" in rapport
        # Le score effectivement utilisé doit rester celui des règles.
        pd.testing.assert_series_equal(
            classement["score_ciblage"], classement["score_global"],
            check_names=False,
        )

    def test_methode_ia_seule_sur_petit_portefeuille_leve_une_erreur_explicite(
        self, petit_portefeuille
    ):
        """Demander explicitement l'IA sans l'avoir n'a pas de repli sensé —
        contrairement à `combine`, où les règles prennent le relais."""
        classement, rapport = analyser_portefeuille(petit_portefeuille, methode="ia")
        assert rapport["ia_disponible"] is False
        # `score_ciblage` ne doit référencer aucune colonne IA absente.
        assert classement["score_ciblage"].notna().all()


class TestClassementProduit:

    def test_classement_est_trie_et_range(self, grand_portefeuille):
        classement, rapport = analyser_portefeuille(grand_portefeuille, methode="combine")
        assert rapport["methode_appliquee"] == "combine"
        assert rapport["ia_disponible"] is True
        assert classement["score_ciblage"].is_monotonic_decreasing
        assert classement["rang"].tolist() == list(range(1, len(classement) + 1))
        assert rapport["n_lignes"] == len(classement)
        assert rapport["n_contribuables"] == classement["gvkey"].nunique()

    def test_les_trois_methodes_produisent_des_classements_valides(
        self, grand_portefeuille
    ):
        for methode in METHODES:
            classement, rapport = analyser_portefeuille(
                grand_portefeuille, methode=methode
            )
            assert len(classement) == len(grand_portefeuille)
            assert classement["score_ciblage"].notna().all()
            assert rapport["methode_demandee"] == methode

    def test_chaque_ligne_porte_un_motif(self, grand_portefeuille):
        classement, _ = analyser_portefeuille(grand_portefeuille, methode="combine")
        assert classement["motif"].map(lambda m: isinstance(m, str) and len(m) > 10).all()

    def test_motif_ia_bien_accorde_en_genre(self, grand_portefeuille):
        """« Charge d'impôt » est féminin : le motif doit dire « inférieure »,
        pas « inférieur »."""
        classement, _ = analyser_portefeuille(grand_portefeuille, methode="combine")
        motifs_impot = classement[
            classement["motif"].str.startswith("Charge d'impôt", na=False)
        ]["motif"]
        for motif in motifs_impot:
            assert "inférieure" in motif

    def test_poids_regles_extreme_neutralise_le_volet_ia(self, grand_portefeuille):
        """poids_regles=1 doit redonner exactement le rang des règles seules."""
        classement, _ = analyser_portefeuille(
            grand_portefeuille, methode="combine", poids_regles=1.0
        )
        np.testing.assert_allclose(
            classement["score_ciblage"].values,
            classement["score_global"].rank(pct=True).values,
            rtol=1e-9,
        )
