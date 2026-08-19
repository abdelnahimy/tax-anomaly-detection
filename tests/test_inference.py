"""Tests du pipeline d'inférence : des données brutes au score de risque.

Deux invariants sont testés ici avec insistance, parce qu'ils ont déjà produit
un bug réel et silencieux :

1. **L'ordre des COLONNES.** Les modèles ont été entraînés sur des tableaux
   NumPy (`df[features].values`) et ne portent pas les noms de colonnes en
   interne. L'ordre est donc la seule chose qui relie une valeur à la bonne
   variable. Une permutation ne lèverait aucune erreur : elle produirait des
   scores faux.

2. **L'ordre des LIGNES.** `add_temporal_features` et `add_fiscal_features`
   trient par (gvkey, fyear) pour calculer les variations. Sans restauration
   explicite de l'ordre d'entrée, l'appelant attribue les scores aux mauvaises
   lignes — 16 418 lignes sur 16 423 étaient concernées avant correction.

Ces deux défauts sont invisibles à l'œil nu et catastrophiques : d'où des tests
qui les traquent spécifiquement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, RAW_ACCOUNTING_VARS
from src.inference import (
    REQUIRED_INPUT_COLS,
    InferenceError,
    load_model,
    prepare_features,
    score_dataframe,
    score_with_diagnostics,
    temporal_coverage,
    validate_input,
)

MODELE_PRESENT = (MODELS_DIR / "xgboost_v2_best.pkl").exists()
besoin_modele = pytest.mark.skipif(
    not MODELE_PRESENT,
    reason="models/xgboost_v2_best.pkl absent — exécutez 05_modeling_supervised_v2.ipynb",
)


# ---------------------------------------------------------------------------
# Validation des entrées
# ---------------------------------------------------------------------------
class TestValidationEntree:

    def test_fichier_complet_est_valide(self, jar2020_minimal):
        assert validate_input(jar2020_minimal) == []

    def test_colonnes_manquantes_signalees(self, jar2020_minimal):
        problemes = validate_input(jar2020_minimal.drop(columns=["sale", "at"]))
        assert problemes
        assert "sale" in problemes[0] and "at" in problemes[0]

    def test_fichier_vide_signale(self, jar2020_minimal):
        problemes = validate_input(jar2020_minimal.iloc[0:0])
        assert any("aucune ligne" in p for p in problemes)

    def test_identifiant_vide_signale(self, jar2020_minimal):
        troue = jar2020_minimal.copy()
        troue.loc[0, "gvkey"] = np.nan
        assert any("gvkey" in p for p in validate_input(troue))

    def test_liste_des_colonnes_requises_coherente(self):
        assert REQUIRED_INPUT_COLS[:2] == ["gvkey", "fyear"]
        assert set(REQUIRED_INPUT_COLS[2:]) == set(RAW_ACCOUNTING_VARS)


# ---------------------------------------------------------------------------
# Invariant n° 1 — l'ordre des colonnes
# ---------------------------------------------------------------------------
@besoin_modele
class TestOrdreDesColonnes:

    def test_ordre_exactement_celui_des_metadonnees(self, jar2020_minimal):
        _, meta = load_model()
        X = prepare_features(jar2020_minimal, meta["features"])
        assert list(X.columns) == list(meta["features"])

    def test_nombre_de_colonnes_attendu_par_le_modele(self, jar2020_minimal):
        modele, meta = load_model()
        X = prepare_features(jar2020_minimal, meta["features"])
        assert X.shape[1] == modele.n_features_in_

    def test_permuter_les_colonnes_dentree_ne_change_rien(self, jar2020_minimal):
        """La sélection doit se faire par NOM depuis les métadonnées, jamais par
        position dans le fichier d'entrée."""
        _, meta = load_model()
        melange = jar2020_minimal.sample(frac=1, axis=1, random_state=3)
        attendu = prepare_features(jar2020_minimal, meta["features"])
        obtenu = prepare_features(melange, meta["features"])
        pd.testing.assert_frame_equal(attendu, obtenu[attendu.columns], rtol=1e-9)

    def test_variable_absente_creee_en_nan(self, jar2020_minimal):
        """Mieux vaut une colonne NaN — que XGBoost sait traiter — qu'un échec."""
        _, meta = load_model()
        ampute = jar2020_minimal.drop(columns=["prcc_f"])
        X = prepare_features(ampute, meta["features"])
        assert "prcc_f_log" in X.columns
        assert X["prcc_f_log"].isna().all()


# ---------------------------------------------------------------------------
# Invariant n° 2 — l'ordre des lignes
# ---------------------------------------------------------------------------
@besoin_modele
class TestOrdreDesLignes:

    def test_ordre_dentree_restaure(self, jar2020_minimal):
        """Le tri interne par (gvkey, fyear) ne doit pas transparaître en sortie."""
        _, meta = load_model()
        desordre = jar2020_minimal.sample(frac=1, random_state=11).reset_index(drop=True)
        X = prepare_features(desordre, meta["features"])
        np.testing.assert_allclose(
            X["at_log"].values,
            np.sign(desordre["at"]) * np.log1p(desordre["at"].abs()),
            rtol=1e-9,
            err_msg="les lignes ont été réordonnées sans être restaurées",
        )

    def test_score_suit_la_ligne_quel_que_soit_lordre(self, jar2020_minimal):
        """Le test qui aurait attrapé le bug historique : scorer un tableau puis
        le même tableau mélangé doit donner à chaque couple (gvkey, fyear) le
        même score."""
        reference = score_dataframe(jar2020_minimal).set_index(["gvkey", "fyear"])
        melange = jar2020_minimal.sample(frac=1, random_state=5).reset_index(drop=True)
        obtenu = score_dataframe(melange).set_index(["gvkey", "fyear"])

        commun = reference.index.intersection(obtenu.index)
        assert len(commun) == len(jar2020_minimal)
        np.testing.assert_allclose(
            reference.loc[commun, "risk_score"].values,
            obtenu.loc[commun, "risk_score"].values,
            rtol=1e-9,
            err_msg="un score a été attribué à la mauvaise ligne",
        )

    def test_index_non_trivial_supporte(self, jar2020_minimal):
        """Un index dupliqué ou exotique ne doit pas casser la restauration :
        c'est pourquoi le module utilise un compteur positionnel, pas l'index."""
        _, meta = load_model()
        bizarre = jar2020_minimal.copy()
        bizarre.index = [7] * len(bizarre)
        X = prepare_features(bizarre, meta["features"])
        assert len(X) == len(bizarre)

    def test_nombre_de_lignes_preserve(self, jar2020_minimal):
        _, meta = load_model()
        assert len(prepare_features(jar2020_minimal, meta["features"])) == len(jar2020_minimal)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@besoin_modele
class TestScoring:

    def test_sortie_complete_et_classee(self, jar2020_minimal):
        resultat = score_dataframe(jar2020_minimal)
        assert len(resultat) == len(jar2020_minimal)
        for colonne in ("risk_score", "risk_rank", "risk_percentile"):
            assert colonne in resultat.columns
        assert resultat["risk_score"].is_monotonic_decreasing
        assert resultat["risk_rank"].tolist() == list(range(1, len(resultat) + 1))

    def test_scores_dans_lintervalle_unitaire(self, jar2020_minimal):
        scores = score_dataframe(jar2020_minimal)["risk_score"]
        assert scores.between(0, 1).all()

    def test_colonnes_dorigine_conservees(self, jar2020_minimal):
        """Le vérificateur doit retrouver ses données dans le classement."""
        enrichi = jar2020_minimal.assign(raison_sociale="TEST SARL")
        assert "raison_sociale" in score_dataframe(enrichi).columns

    def test_entree_invalide_rejetee_explicitement(self, jar2020_minimal):
        with pytest.raises(InferenceError, match="invalides"):
            score_dataframe(jar2020_minimal.drop(columns=["sale"]))

    def test_diagnostics_complets(self, jar2020_minimal):
        diag = score_with_diagnostics(jar2020_minimal)
        assert diag["n_rows"] == len(jar2020_minimal)
        assert diag["n_features"] == 54
        assert 0 <= diag["coverage"]["available_pct"] <= 100
        assert diag["top_5pct_threshold"] >= diag["scored"]["risk_score"].min()


# ---------------------------------------------------------------------------
# Couverture temporelle
# ---------------------------------------------------------------------------
@besoin_modele
class TestCouvertureTemporelle:

    def test_couverture_nulle_sans_historique(self, jar2020_minimal):
        """Un seul exercice par contribuable : les 20 variables de trajectoire
        sont toutes vides, et cela doit être rapporté et non masqué."""
        _, meta = load_model()
        un_exercice = jar2020_minimal[jar2020_minimal["fyear"] == 2012]
        X = prepare_features(un_exercice, meta["features"])
        couverture = temporal_coverage(X, meta["temporal_features"])
        assert couverture["available_pct"] == pytest.approx(0.0)
        assert couverture["rows_with_history"] == 0

    def test_couverture_croit_avec_lhistorique(self, jar2020_minimal):
        _, meta = load_model()
        deux = jar2020_minimal[jar2020_minimal["fyear"] <= 2011]
        trois = jar2020_minimal
        c2 = temporal_coverage(prepare_features(deux, meta["features"]),
                               meta["temporal_features"])
        c3 = temporal_coverage(prepare_features(trois, meta["features"]),
                               meta["temporal_features"])
        assert c3["available_pct"] > c2["available_pct"]
        assert c3["rows_with_history"] > c2["rows_with_history"]


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------
class TestChargementModele:

    @besoin_modele
    def test_metadonnees_coherentes_avec_le_modele(self):
        modele, meta = load_model()
        assert meta["n_features"] == len(meta["features"]) == modele.n_features_in_
        assert len(meta["temporal_features"]) == 20
        assert len(meta["fiscal_features"]) == 6

    @besoin_modele
    def test_toutes_les_variables_derivent_des_28_postes(self):
        """Aucune variable du modèle ne doit venir d'ailleurs que des 28 postes
        comptables — sans quoi le mapping de la liasse serait incomplet."""
        _, meta = load_model()
        from src.feature_engineering import FISCAL_FEATURE_NAMES

        for variable in meta["features"]:
            if variable in FISCAL_FEATURE_NAMES:
                continue
            base = variable
            for suffixe in ("_log_d1", "_log_d2", "_log"):
                if base.endswith(suffixe):
                    base = base[: -len(suffixe)]
                    break
            assert base in RAW_ACCOUNTING_VARS, f"variable orpheline : {variable}"

    def test_message_explicite_si_modele_absent(self, tmp_path):
        with pytest.raises(InferenceError, match="introuvable"):
            load_model(models_dir=tmp_path)
