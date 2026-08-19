"""Tests de la passerelle liasse fiscale marocaine → variables du modèle.

Ce qui est vérifié ici n'est pas cosmétique. Une correspondance fausse — un
poste pris pour un autre, une somme mal reconstruite, une conversion d'unité
oubliée — ne provoque **aucune erreur** : elle produit des scores plausibles et
faux. C'est précisément le genre de défaut qu'un jury ne verra pas et qui
invaliderait pourtant tout le travail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_ACCOUNTING_VARS
from src.liasse_mapping import (
    COLONNES_LIASSE,
    LIASSE_MAPPING,
    UNITES_SUPPORTEES,
    _somme,
    mapper_liasse,
    modele_csv_liasse,
    normaliser_colonnes,
    table_correspondance,
    verifier_liasse,
)


# ---------------------------------------------------------------------------
# Intégrité de la table de correspondance
# ---------------------------------------------------------------------------
class TestTableCorrespondance:

    def test_couvre_exactement_les_28_variables(self):
        """Ni variable oubliée, ni variable en trop : le modèle en attend 28."""
        variables = [c.variable for c in LIASSE_MAPPING]
        assert sorted(variables) == sorted(RAW_ACCOUNTING_VARS)
        assert len(variables) == len(set(variables)), "doublon dans le mapping"

    def test_chaque_correspondance_est_documentee(self):
        """Une correspondance sans justification n'est pas défendable."""
        for c in LIASSE_MAPPING:
            assert c.remarque.strip(), f"{c.variable} : remarque vide"
            assert c.poste_liasse.strip(), f"{c.variable} : poste non renseigné"
            assert c.colonnes_source, f"{c.variable} : aucune colonne source"

    def test_niveaux_de_fidelite_valides(self):
        attendus = {"directe", "agregat", "approximation", "indisponible"}
        for c in LIASSE_MAPPING:
            assert c.fidelite in attendus, f"{c.variable} : fidélité « {c.fidelite} »"

    def test_colonnes_source_declarees_dans_le_format_dentree(self):
        """Toute colonne citée par une correspondance doit figurer dans le
        gabarit, sinon l'utilisateur ne saura jamais qu'il doit la fournir."""
        connues = set(COLONNES_LIASSE)
        for c in LIASSE_MAPPING:
            for source in c.colonnes_source:
                if c.fidelite == "indisponible" and source == "actions_privilegiees":
                    continue  # convention interne, pas un poste à saisir
                assert source in connues, f"{c.variable} : `{source}` absent du gabarit"

    def test_export_dataframe(self):
        table = table_correspondance()
        assert len(table) == 28
        assert not table["Poste de la liasse marocaine"].isna().any()

    def test_gabarit_csv_a_les_bonnes_colonnes(self):
        assert list(modele_csv_liasse().columns) == list(COLONNES_LIASSE)


# ---------------------------------------------------------------------------
# Normalisation des en-têtes
# ---------------------------------------------------------------------------
class TestNormalisationColonnes:

    @pytest.mark.parametrize("saisi,attendu", [
        ("Chiffre d'Affaires", "chiffre_d_affaires"),
        ("TOTAL ACTIF", "total_actif"),
        ("Résultat Net Exercice", "resultat_net_exercice"),
        ("total-passif-circulant", "total_passif_circulant"),
        ("  Stocks  ", "stocks"),
        ("Trésorerie Actif", "tresorerie_actif"),
    ])
    def test_variantes_decriture(self, saisi, attendu):
        df = pd.DataFrame({saisi: [1.0]})
        assert list(normaliser_colonnes(df).columns) == [attendu]

    def test_ne_modifie_pas_lentree(self):
        df = pd.DataFrame({"TOTAL ACTIF": [1.0]})
        normaliser_colonnes(df)
        assert list(df.columns) == ["TOTAL ACTIF"]


# ---------------------------------------------------------------------------
# Sémantique de l'absence : NaN n'est pas zéro
# ---------------------------------------------------------------------------
class TestSemantiqueDeLabsence:

    def test_somme_ignore_les_postes_absents(self):
        """Un poste vide dans une liasse vaut zéro : la somme reste calculable."""
        df = pd.DataFrame({"a": [10.0, 20.0], "b": [np.nan, 5.0]})
        assert _somme(df, "a", "b").tolist() == [10.0, 25.0]

    def test_somme_est_nan_si_tous_les_postes_manquent(self):
        """En revanche, si aucun poste de l'agrégat n'est disponible, on ne peut
        rien affirmer — surtout pas zéro."""
        df = pd.DataFrame({"a": [np.nan, 1.0], "b": [np.nan, 2.0]})
        resultat = _somme(df, "a", "b")
        assert np.isnan(resultat.iloc[0])
        assert resultat.iloc[1] == 3.0

    def test_colonne_absente_donne_nan_pas_zero(self, liasse_minimale):
        """Le point le plus important du module : combler par des zéros
        affirmerait au modèle une information fausse."""
        sans_stocks = liasse_minimale.drop(columns=["stocks"])
        modele, _ = mapper_liasse(sans_stocks)
        assert modele["invt"].isna().all(), "un poste absent a été comblé par 0"

    def test_variables_hors_liasse_restent_vides(self, liasse_minimale):
        modele, rapport = mapper_liasse(liasse_minimale)
        assert set(rapport["variables_vides"]) == {"csho", "dltis", "sstk", "prcc_f"}
        for variable in ("csho", "dltis", "sstk", "prcc_f"):
            assert modele[variable].isna().all()

    def test_pstk_vaut_zero_par_convention(self, liasse_minimale):
        """Absence structurelle et non lacune : le CGNC ne connaît pas les
        actions de préférence, donc la valeur correcte est 0, pas « inconnu »."""
        modele, _ = mapper_liasse(liasse_minimale)
        assert (modele["pstk"] == 0).all()


# ---------------------------------------------------------------------------
# Exactitude arithmétique des correspondances
# ---------------------------------------------------------------------------
class TestExactitudeDesCorrespondances:

    @staticmethod
    def _mapper_sans_conversion(df):
        """Mapping à l'échelle 1 : les valeurs de sortie sont directement
        comparables aux postes saisis."""
        return mapper_liasse(df, unite="millions_dirhams", convertir_en_usd=False)[0]

    def test_correspondances_directes(self, liasse_minimale):
        modele = self._mapper_sans_conversion(liasse_minimale)
        directes = {
            "at": "total_actif",
            "ceq": "total_capitaux_propres",
            "sale": "chiffre_affaires",
            "ni": "resultat_net_exercice",
            "txt": "impots_sur_les_resultats",
            "invt": "stocks",
            "rect": "creances_actif_circulant",
            "dltt": "dettes_financement",
            "ap": "fournisseurs_comptes_rattaches",
            "txp": "etat_impots_sur_resultats_dette",
            "xint": "charges_interets",
            "ppegt": "immobilisations_corporelles_brut",
            "ivao": "immobilisations_financieres",
            "ivst": "titres_valeurs_placement",
            "dp": "dotations_exploitation",
        }
        for variable, poste in directes.items():
            attendu = liasse_minimale[poste].values
            obtenu = modele[variable].values
            np.testing.assert_allclose(obtenu, attendu, rtol=1e-9,
                                       err_msg=f"{variable} ≠ {poste}")

    def test_act_somme_actif_circulant_et_tresorerie(self, liasse_minimale):
        modele = self._mapper_sans_conversion(liasse_minimale)
        attendu = (liasse_minimale["total_actif_circulant"]
                   + liasse_minimale["tresorerie_actif"])
        np.testing.assert_allclose(modele["act"], attendu, rtol=1e-9)

    def test_lct_somme_passif_circulant_et_tresorerie_passif(self, liasse_minimale):
        modele = self._mapper_sans_conversion(liasse_minimale)
        attendu = (liasse_minimale["total_passif_circulant"]
                   + liasse_minimale["tresorerie_passif"])
        np.testing.assert_allclose(modele["lct"], attendu, rtol=1e-9)

    def test_che_somme_tresorerie_et_titres(self, liasse_minimale):
        modele = self._mapper_sans_conversion(liasse_minimale)
        attendu = (liasse_minimale["tresorerie_actif"]
                   + liasse_minimale["titres_valeurs_placement"])
        np.testing.assert_allclose(modele["che"], attendu, rtol=1e-9)

    def test_cogs_somme_des_deux_postes_dachats(self, liasse_minimale):
        modele = self._mapper_sans_conversion(liasse_minimale)
        attendu = (liasse_minimale["achats_revendus_marchandises"]
                   + liasse_minimale["achats_consommes_matieres"])
        np.testing.assert_allclose(modele["cogs"], attendu, rtol=1e-9)

    def test_lt_est_le_total_des_dettes(self, liasse_minimale):
        """`lt` = total actif − capitaux propres − capitaux propres assimilés."""
        modele = self._mapper_sans_conversion(liasse_minimale)
        attendu = (liasse_minimale["total_actif"]
                   - liasse_minimale["total_capitaux_propres"]
                   - liasse_minimale["capitaux_propres_assimiles"])
        np.testing.assert_allclose(modele["lt"], attendu, rtol=1e-9)

    def test_ib_retranche_le_resultat_non_courant(self, liasse_minimale):
        modele = self._mapper_sans_conversion(liasse_minimale)
        attendu = (liasse_minimale["resultat_net_exercice"]
                   - liasse_minimale["resultat_non_courant"])
        np.testing.assert_allclose(modele["ib"], attendu, rtol=1e-9)

    def test_re_somme_les_quatre_postes_de_reserves(self, liasse_minimale):
        modele = self._mapper_sans_conversion(liasse_minimale)
        attendu = (liasse_minimale["reserve_legale"]
                   + liasse_minimale["autres_reserves"]
                   + liasse_minimale["report_a_nouveau"]
                   + liasse_minimale["resultats_en_instance"])
        np.testing.assert_allclose(modele["re"], attendu, rtol=1e-9)

    def test_report_a_nouveau_negatif_conserve_son_signe(self):
        """Des pertes antérieures doivent rester des pertes."""
        df = pd.DataFrame([{
            "identifiant_fiscal": 1, "exercice": 2023,
            "reserve_legale": 100.0, "autres_reserves": 50.0,
            "report_a_nouveau": -400.0, "resultats_en_instance": 0.0,
        }])
        modele, _ = mapper_liasse(df, unite="millions_dirhams", convertir_en_usd=False)
        assert modele["re"].iloc[0] == pytest.approx(-250.0)


# ---------------------------------------------------------------------------
# Unités et devise
# ---------------------------------------------------------------------------
class TestConversionDunites:

    @pytest.mark.parametrize("unite,facteur_attendu", [
        ("dirhams", 1 / 1e6),
        ("milliers_dirhams", 1e3 / 1e6),
        ("millions_dirhams", 1.0),
    ])
    def test_mise_a_lechelle_en_millions(self, liasse_minimale, unite, facteur_attendu):
        modele, rapport = mapper_liasse(
            liasse_minimale, unite=unite, convertir_en_usd=False
        )
        assert rapport["facteur_echelle"] == pytest.approx(facteur_attendu)
        attendu = liasse_minimale["total_actif"] * facteur_attendu
        np.testing.assert_allclose(modele["at"], attendu, rtol=1e-9)

    def test_conversion_en_dollars(self, liasse_minimale):
        taux = 9.85
        modele, rapport = mapper_liasse(
            liasse_minimale, unite="dirhams", mad_par_usd=taux, convertir_en_usd=True
        )
        attendu = liasse_minimale["total_actif"] / 1e6 / taux
        np.testing.assert_allclose(modele["at"], attendu, rtol=1e-9)
        assert rapport["devise_sortie"] == "millions USD"

    @staticmethod
    def _ratios(liasse, unite, convertir):
        from src.feature_engineering import add_fiscal_features

        modele, _ = mapper_liasse(liasse, unite=unite, convertir_en_usd=convertir)
        enrichi, noms = add_fiscal_features(modele)
        return enrichi[noms].sort_index()

    def test_ratios_fiscaux_invariants_dechelle(self, liasse_minimale):
        """Argument central du mémoire : les 6 ratios métier résistent au
        changement d'unité, contrairement aux 48 variables de niveau.

        L'invariance est vraie *à la garde epsilon près*. `add_fiscal_features`
        protège ses divisions par un terme constant (`eps = 1e-6`) ajouté au
        dénominateur — par exemple `txp / (|txt| + eps)`. Ce terme étant absolu
        et non relatif, il ne se simplifie pas lors d'un changement d'échelle :
        l'invariance est donc excellente mais pas exacte.

        Cet `eps` ne doit surtout pas être rendu relatif : le modèle sauvegardé
        a été entraîné avec cette formule exacte, et la modifier décalerait
        silencieusement toutes les variables.
        """
        en_usd = self._ratios(liasse_minimale, "dirhams", True)
        en_mad = self._ratios(liasse_minimale, "millions_dirhams", False)

        pd.testing.assert_frame_equal(en_usd, en_mad, rtol=1e-4)

    def test_effet_de_la_garde_epsilon_reste_negligeable(self, liasse_minimale):
        """Quantifie l'écart résiduel, au lieu de se contenter d'une tolérance.

        L'écart relatif vaut approximativement `eps / dénominateur`. Il croît
        donc quand les montants rétrécissent — et convertir des dirhams en
        millions de dollars les divise par ~10⁷. C'est sans conséquence pour des
        montants ordinaires, mais la marge n'est pas infinie pour de très
        petites entreprises : un point à connaître avant d'appliquer le
        dispositif à un portefeuille de TPE.
        """
        en_usd = self._ratios(liasse_minimale, "dirhams", True)
        en_mad = self._ratios(liasse_minimale, "millions_dirhams", False)

        ecart_relatif = ((en_usd - en_mad).abs() / en_mad.abs().clip(lower=1e-12)).max().max()
        assert ecart_relatif < 1e-5, (
            f"écart relatif de {ecart_relatif:.2e} — la garde epsilon pèse plus "
            "que prévu, vérifier les ordres de grandeur en entrée"
        )

    def test_cours_de_bourse_nest_pas_divise_par_un_million(self):
        """`prcc_f` est un cours unitaire, pas un montant agrégé : le ramener en
        « millions » n'aurait aucun sens."""
        df = pd.DataFrame([{
            "identifiant_fiscal": 1, "exercice": 2023, "cours_bourse_cloture": 500.0,
        }])
        modele, _ = mapper_liasse(df, unite="dirhams", mad_par_usd=10.0)
        assert modele["prcc_f"].iloc[0] == pytest.approx(50.0)

    def test_nombre_dactions_exprime_en_millions(self):
        """Compustat exprime `csho` en millions de titres, sans conversion de
        devise — un nombre d'actions n'a pas de monnaie."""
        df = pd.DataFrame([{
            "identifiant_fiscal": 1, "exercice": 2023, "nombre_actions": 2_500_000.0,
        }])
        modele, _ = mapper_liasse(df, unite="dirhams", mad_par_usd=10.0)
        assert modele["csho"].iloc[0] == pytest.approx(2.5)

    def test_unite_inconnue_rejetee(self, liasse_minimale):
        with pytest.raises(ValueError, match="Unité inconnue"):
            mapper_liasse(liasse_minimale, unite="euros")

    def test_taux_de_change_invalide_rejete(self, liasse_minimale):
        with pytest.raises(ValueError, match="strictement positif"):
            mapper_liasse(liasse_minimale, mad_par_usd=0)

    def test_toutes_les_unites_documentees_fonctionnent(self, liasse_minimale):
        for unite in UNITES_SUPPORTEES:
            modele, _ = mapper_liasse(liasse_minimale, unite=unite)
            assert len(modele) == len(liasse_minimale)


# ---------------------------------------------------------------------------
# Identifiants et colonnes descriptives
# ---------------------------------------------------------------------------
class TestIdentifiants:

    def test_colonnes_identifiantes_obligatoires(self, liasse_minimale):
        with pytest.raises(ValueError, match="identifiantes"):
            mapper_liasse(liasse_minimale.drop(columns=["exercice"]))

    def test_identifiant_alphanumerique_encode_sans_echec(self):
        """Un ICE alphanumérique ne doit pas faire tomber le mapping : on encode
        en catégories stables plutôt que d'échouer."""
        df = pd.DataFrame([
            {"identifiant_fiscal": "ICE-001", "exercice": 2022, "total_actif": 1.0},
            {"identifiant_fiscal": "ICE-002", "exercice": 2022, "total_actif": 2.0},
            {"identifiant_fiscal": "ICE-001", "exercice": 2023, "total_actif": 3.0},
        ])
        modele, rapport = mapper_liasse(df)
        assert rapport["n_contribuables"] == 2
        assert modele["gvkey"].iloc[0] == modele["gvkey"].iloc[2], (
            "un même contribuable doit garder le même identifiant entre exercices"
        )

    def test_colonnes_descriptives_recopiees(self, liasse_minimale):
        enrichi = liasse_minimale.assign(
            raison_sociale="SOCIETE TEST SARL", secteur="Négoce", commune="Tétouan",
        )
        modele, rapport = mapper_liasse(enrichi)
        assert (modele["raison_sociale"] == "SOCIETE TEST SARL").all()
        assert rapport["colonnes_ignorees"] == []

    def test_colonnes_inconnues_signalees_et_non_bloquantes(self, liasse_minimale):
        modele, rapport = mapper_liasse(liasse_minimale.assign(colonne_exotique=1.0))
        assert "colonne_exotique" in rapport["colonnes_ignorees"]
        assert len(modele) == len(liasse_minimale)


# ---------------------------------------------------------------------------
# Contrôles de cohérence comptable
# ---------------------------------------------------------------------------
class TestVerifierLiasse:

    def test_fichier_correct_est_scorable(self, liasse_minimale):
        diagnostic = verifier_liasse(liasse_minimale)
        assert diagnostic["scorable"]
        assert diagnostic["anomalies"] == []

    def test_bilan_equilibre_ne_declenche_pas_davertissement(self, liasse_minimale):
        avertissements = verifier_liasse(liasse_minimale)["avertissements"]
        assert not any("actif ≠ passif" in a for a in avertissements)

    def test_bilan_desequilibre_signale(self, liasse_minimale):
        fausse = liasse_minimale.copy()
        fausse["total_actif"] = fausse["total_actif"] * 1.5
        avertissements = verifier_liasse(fausse)["avertissements"]
        assert any("actif ≠ passif" in a for a in avertissements)

    def test_doublon_contribuable_exercice_est_bloquant(self, liasse_minimale):
        """Un doublon fausserait le calcul des variations d'un exercice à
        l'autre — c'est une erreur de jointure, pas un avertissement."""
        avec_doublon = pd.concat([liasse_minimale, liasse_minimale.head(1)])
        diagnostic = verifier_liasse(avec_doublon)
        assert not diagnostic["scorable"]
        assert any("doublon" in a.lower() for a in diagnostic["anomalies"])

    def test_identifiant_manquant_est_bloquant(self, liasse_minimale):
        sans_id = liasse_minimale.drop(columns=["identifiant_fiscal"])
        assert not verifier_liasse(sans_id)["scorable"]

    def test_exercice_vide_est_bloquant(self, liasse_minimale):
        trouee = liasse_minimale.copy()
        trouee.loc[0, "exercice"] = np.nan
        assert not verifier_liasse(trouee)["scorable"]

    def test_absence_dhistorique_signalee(self, liasse_minimale):
        """Sans historique, les 20 variables de trajectoire sont perdues :
        l'utilisateur doit le savoir avant de commenter des résultats."""
        un_seul_exercice = liasse_minimale[liasse_minimale["exercice"] == 2023]
        avertissements = verifier_liasse(un_seul_exercice)["avertissements"]
        assert any("un seul exercice" in a for a in avertissements)

    def test_postes_essentiels_absents_signales(self, liasse_minimale):
        ampute = liasse_minimale.drop(columns=["chiffre_affaires"])
        avertissements = verifier_liasse(ampute)["avertissements"]
        assert any("chiffre_affaires" in a for a in avertissements)


# ---------------------------------------------------------------------------
# Rapport de couverture
# ---------------------------------------------------------------------------
class TestRapportDeCouverture:

    def test_couverture_attendue_sur_liasse_complete(self, liasse_minimale):
        """24 variables sur 28 : les 4 hors liasse manquent structurellement."""
        _, rapport = mapper_liasse(liasse_minimale)
        assert rapport["n_variables_renseignees"] == 24
        assert rapport["taux_couverture_global_pct"] == pytest.approx(85.7, abs=0.1)

    def test_rapport_expose_les_parametres_de_conversion(self, liasse_minimale):
        _, rapport = mapper_liasse(liasse_minimale, unite="dirhams", mad_par_usd=9.5)
        assert rapport["unite_saisie"] == "dirhams"
        assert rapport["mad_par_usd"] == 9.5
        assert rapport["n_contribuables"] == 2
        assert rapport["exercices"] == [2021, 2022, 2023]

    def test_detail_couvre_les_28_variables(self, liasse_minimale):
        _, rapport = mapper_liasse(liasse_minimale)
        assert len(rapport["detail"]) == 28
        assert set(rapport["detail"]["variable"]) == set(RAW_ACCOUNTING_VARS)


# ---------------------------------------------------------------------------
# Le fichier d'exemple livré avec le projet
# ---------------------------------------------------------------------------
class TestExempleLivre:

    def test_exemple_est_scorable_et_equilibre(self, racine):
        chemin = racine / "dashboard" / "exemple_liasse_marocaine.csv"
        if not chemin.exists():
            pytest.skip("exemple_liasse_marocaine.csv absent")
        df = pd.read_csv(chemin)
        diagnostic = verifier_liasse(df)
        assert diagnostic["scorable"], diagnostic["anomalies"]
        assert diagnostic["avertissements"] == [], (
            "le fichier d'exemple doit être irréprochable : il sert de gabarit"
        )

    def test_exemple_produit_la_couverture_annoncee(self, racine):
        chemin = racine / "dashboard" / "exemple_liasse_marocaine.csv"
        if not chemin.exists():
            pytest.skip("exemple_liasse_marocaine.csv absent")
        _, rapport = mapper_liasse(pd.read_csv(chemin), unite="dirhams")
        assert rapport["taux_couverture_global_pct"] == pytest.approx(85.7, abs=0.1)
        assert rapport["colonnes_ignorees"] == []
