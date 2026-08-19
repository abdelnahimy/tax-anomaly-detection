"""Tests du modèle de reconstitution (`src.modele_marocain`).

Deux catégories de risques guident ces tests :

1. **Le biais de taille.** C'est le défaut qui disqualifie le modèle JAR2020
   sur données marocaines (voir `docs/06_transferabilite.md`) : le score finit
   par mesurer la taille de l'entreprise plutôt que son anomalie. Le module
   neutralise ce biais en normalisant chaque colonne par son propre ordre de
   grandeur avant le log signé — on vérifie ici que cette neutralisation tient,
   y compris quand l'unité de saisie change (dirhams vs millions de dirhams).

2. **La régression de l'inversion d'échelle.** `entrainer_reconstitution`
   prédit dans un espace log normalisé ; sans ré-application de l'échelle du
   train, le montant « attendu » affiché au vérificateur n'a plus de rapport
   avec le montant « déclaré » (bug corrigé : les deux se retrouvaient dans des
   ordres de grandeur différents, ex. attendu = 1,1 milliard pour un déclaré de
   30 634 MAD). On vérifie que déclaré et attendu restent du même ordre de
   grandeur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.liasse_mapping import mapper_liasse
from src.modele_marocain import (
    CIBLES,
    EFFECTIF_MINIMUM,
    ModeleInsuffisant,
    _echelle_robuste,
    _inverse_log_signe,
    _log_signe,
    expliquer_reconstitution,
    scorer_reconstitution,
)


def _en_modele(liasse: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return mapper_liasse(liasse, **kwargs)[0]


@pytest.fixture(scope="module")
def portefeuille_reel() -> pd.DataFrame:
    """Le portefeuille simulé de 220 contribuables livré avec le projet."""
    from pathlib import Path

    chemin = (
        Path(__file__).resolve().parent.parent
        / "data" / "exemples_marocains" / "portefeuille_dri_tetouan.csv"
    )
    if not chemin.exists():
        pytest.skip("jeu de test absent — python scripts/generer_jeu_test_marocain.py")
    return _en_modele(pd.read_csv(chemin), unite="dirhams")


# ---------------------------------------------------------------------------
# Transformations d'échelle
# ---------------------------------------------------------------------------
class TestEchelle:

    def test_echelle_robuste_est_la_mediane_absolue(self):
        serie = pd.Series([10.0, -20.0, 30.0, -40.0, 1000.0])
        # médiane des valeurs absolues : 10, 20, 30, 40, 1000 -> 30
        assert _echelle_robuste(serie) == pytest.approx(30.0)

    def test_echelle_robuste_ignore_les_zeros(self):
        serie = pd.Series([0.0, 0.0, 5.0, 10.0, 15.0])
        # les zéros sont exclus avant la médiane : 5, 10, 15 -> 10
        assert _echelle_robuste(serie) == pytest.approx(10.0)

    def test_echelle_robuste_repli_sur_1_si_rien_dexploitable(self):
        assert _echelle_robuste(pd.Series([np.nan, np.nan])) == 1.0
        assert _echelle_robuste(pd.Series([], dtype="float64")) == 1.0
        assert _echelle_robuste(pd.Series([0.0, 0.0])) == 1.0

    def test_inverse_log_signe_est_linverse_exact_de_log_signe(self):
        """Sans cette réciprocité, le montant « attendu » affiché au
        vérificateur ne correspond à rien : c'est le bug qui vient d'être
        corrigé (l'inversion manquait après la normalisation d'échelle)."""
        valeurs = pd.Series([1_234_567.0, -890_123.0, 45.0, -3_000_000.0, 0.0])
        echelle = _echelle_robuste(valeurs)
        aller = _log_signe(valeurs)
        retour = _inverse_log_signe(aller, echelle)
        np.testing.assert_allclose(retour.values, valeurs.values, rtol=1e-6, atol=1e-6)

    def test_log_signe_invariant_dechelle_apres_inversion(self):
        """Que la colonne soit en dirhams ou en centimes, l'aller-retour doit
        retrouver la valeur d'origine — dans l'unité de la colonne fournie."""
        dirhams = pd.Series([100_000.0, 5_000_000.0, -250_000.0])
        centimes = dirhams * 100

        for valeurs in (dirhams, centimes):
            echelle = _echelle_robuste(valeurs)
            retour = _inverse_log_signe(_log_signe(valeurs), echelle)
            np.testing.assert_allclose(retour.values, valeurs.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# Garde-fou sur l'effectif
# ---------------------------------------------------------------------------
class TestEffectifMinimum:

    def test_leve_modele_insuffisant_sous_le_seuil(self, portefeuille_stable=None):
        from src.liasse_mapping import mapper_liasse as _map

        # Un tout petit portefeuille, très inférieur à EFFECTIF_MINIMUM.
        lignes = []
        for i in range(5):
            k = 1_000_000.0
            lignes.append({
                "identifiant_fiscal": 80000 + i, "exercice": 2023,
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
        modele = _map(pd.DataFrame(lignes))[0]
        assert modele["gvkey"].nunique() < EFFECTIF_MINIMUM

        with pytest.raises(ModeleInsuffisant):
            scorer_reconstitution(modele)


# ---------------------------------------------------------------------------
# Sur le portefeuille livré
# ---------------------------------------------------------------------------
class TestReconstitutionSurPortefeuilleReel:

    def test_score_croissant_pour_chaque_contribuable(self, portefeuille_reel):
        resultat = scorer_reconstitution(portefeuille_reel)
        assert len(resultat) == len(portefeuille_reel)
        assert resultat["rang_reconstitution"].tolist() == list(
            range(1, len(resultat) + 1)
        )

    def test_r2_hors_echantillon_est_correct(self, portefeuille_reel):
        """Un R² proche de 0 ou négatif signalerait un modèle qui n'apprend
        rien ; un R² suspicieusement proche de 1 signalerait une fuite (cible
        visible dans les variables explicatives)."""
        resultat = scorer_reconstitution(portefeuille_reel)
        for nom, diag in resultat.attrs["diagnostics"].items():
            r2 = diag.get("r2")
            assert r2 is not None and not np.isnan(r2), f"R² manquant pour {nom}"
            assert 0.5 < r2 < 0.999, f"R² suspect pour {nom} : {r2}"

    def test_attendu_du_meme_ordre_de_grandeur_que_declare(self, portefeuille_reel):
        """Régression du bug d'inversion d'échelle : le montant reconstitué
        doit être comparable au montant déclaré, jamais décalé de plusieurs
        ordres de grandeur."""
        resultat = scorer_reconstitution(portefeuille_reel)
        for cible in CIBLES:
            colonne_prevu = f"prevu_{cible.nom}"
            if colonne_prevu not in resultat.columns:
                continue
            declare = resultat[cible.nom].abs()
            attendu = resultat[colonne_prevu].abs()
            valide = (declare > 1e-9) & attendu.notna()
            ratio = (attendu[valide] / declare[valide]).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            assert len(ratio) > 0
            # Le ratio attendu/déclaré doit rester dans un domaine raisonnable —
            # avant correction, il grimpait à des facteurs de 10 000 ou plus.
            assert ratio.median() < 50, (
                f"cible `{cible.nom}` : ratio médian attendu/déclaré = "
                f"{ratio.median():.1f} — l'inversion d'échelle est probablement cassée"
            )

    def test_ecart_pct_coherent_avec_declare_et_attendu(self, portefeuille_reel):
        """L'écart affiché doit correspondre à l'écart réel entre déclaré et
        attendu, pas à un artefact de l'espace log interne."""
        resultat = scorer_reconstitution(portefeuille_reel).reset_index(drop=True)
        cible = CIBLES[0]
        colonne_prevu, colonne_ecart = f"prevu_{cible.nom}", f"ecart_{cible.nom}_pct"

        sous = resultat[[cible.nom, colonne_prevu, colonne_ecart]].dropna()
        sous = sous[sous[colonne_prevu].abs() > 1e-9]
        ecart_recalcule = (
            (sous[cible.nom] - sous[colonne_prevu]) / sous[colonne_prevu] * 100
        ).clip(-99, 500)
        np.testing.assert_allclose(
            sous[colonne_ecart].values, ecart_recalcule.values, rtol=0.05, atol=2.0
        )

    def test_expliquer_reconstitution_donne_des_montants_positifs(
        self, portefeuille_reel
    ):
        resultat = scorer_reconstitution(portefeuille_reel).reset_index(drop=True)
        pire = resultat["score_reconstitution"].idxmax()
        detail = expliquer_reconstitution(resultat, pire)
        assert not detail.empty
        for colonne in ("declare", "attendu"):
            assert (detail[colonne] >= 0).all()

    def test_score_invariant_dechelle(self, portefeuille_reel):
        """Le score de reconstitution ne doit pas dépendre de l'unité de
        saisie : c'est le défaut précis qui disqualifie JAR2020 au Maroc, et le
        modèle marocain ne doit pas le reproduire."""
        from pathlib import Path

        chemin = (
            Path(__file__).resolve().parent.parent
            / "data" / "exemples_marocains" / "portefeuille_dri_tetouan.csv"
        )
        brut = pd.read_csv(chemin)

        en_dirhams = _en_modele(brut, unite="dirhams")
        en_millions = _en_modele(brut, unite="millions_dirhams")

        a = scorer_reconstitution(en_dirhams).set_index(["gvkey", "fyear"])[
            "score_reconstitution"
        ].sort_index()
        b = scorer_reconstitution(en_millions).set_index(["gvkey", "fyear"])[
            "score_reconstitution"
        ].sort_index()

        # Les modèles sont ré-entraînés séparément (XGBoost n'est pas
        # déterministe au bit près d'une échelle à l'autre) : on compare donc
        # les RANGS plutôt que les valeurs exactes, avec une tolérance large.
        correlation = a.rank().corr(b.rank())
        assert correlation > 0.90, (
            f"corrélation de rang dirhams/millions = {correlation:.2f} — "
            "le score dépend de l'unité de saisie"
        )
