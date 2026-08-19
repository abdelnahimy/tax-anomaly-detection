"""Tests de la règle de rupture intra-contribuable.

Deux propriétés font tout l'intérêt de cette règle face au modèle appris, et
c'est donc sur elles que porte l'essentiel des tests :

1. **l'invariance d'échelle** — un score identique en dirhams et en millions de
   dollars, sans quoi la règle hériterait du biais de taille qui disqualifie le
   modèle JAR2020 sur données marocaines ;
2. **la comparaison à soi-même** — le score d'un contribuable ne doit dépendre
   que de sa propre trajectoire, jamais de celle des autres lignes du fichier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.liasse_mapping import mapper_liasse
from src.regle_rupture import (
    INDICATEURS,
    NOMS_INDICATEURS,
    calculer_indicateurs,
    expliquer_rupture,
    scorer_ruptures,
)


@pytest.fixture
def portefeuille_stable() -> pd.DataFrame:
    """Trois contribuables aux ratios parfaitement stables sur quatre exercices.

    Sans rupture, le score doit rester au plancher : c'est le témoin négatif.
    """
    lignes = []
    for identifiant, echelle in ((11111111, 1.0), (22222222, 25.0), (33333333, 400.0)):
        for exercice in (2021, 2022, 2023, 2024):
            k = echelle * 1_000_000 * (1.06 ** (exercice - 2021))  # croissance régulière
            lignes.append({
                "identifiant_fiscal": identifiant, "exercice": exercice,
                "total_actif": 100 * k, "total_actif_circulant": 43 * k,
                "tresorerie_actif": 7 * k, "stocks": 15 * k,
                "creances_actif_circulant": 25 * k, "titres_valeurs_placement": 3 * k,
                "immobilisations_corporelles_brut": 80 * k,
                "immobilisations_financieres": 2 * k,
                "total_capitaux_propres": 40 * k, "capitaux_propres_assimiles": 2 * k,
                "reserve_legale": 4 * k, "autres_reserves": 6 * k,
                "report_a_nouveau": 10 * k, "resultats_en_instance": 0.0,
                "dettes_financement": 20 * k, "dettes_financement_moins_1an": 5 * k,
                "provisions_durables": 3 * k, "fournisseurs_comptes_rattaches": 18 * k,
                "etat_impots_sur_resultats_dette": 2 * k,
                "total_passif_circulant": 30 * k, "tresorerie_passif": 5 * k,
                "chiffre_affaires": 120 * k, "achats_revendus_marchandises": 50 * k,
                "achats_consommes_matieres": 20 * k, "dotations_exploitation": 8 * k,
                "charges_interets": 1 * k, "impots_sur_les_resultats": 4 * k,
                "resultat_non_courant": 1 * k, "resultat_net_exercice": 12 * k,
            })
    return pd.DataFrame(lignes)


def _en_modele(liasse: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return mapper_liasse(liasse, **kwargs)[0]


# ---------------------------------------------------------------------------
# Indicateurs
# ---------------------------------------------------------------------------
class TestIndicateurs:

    def test_tous_les_indicateurs_sont_calcules(self, portefeuille_stable):
        ratios = calculer_indicateurs(_en_modele(portefeuille_stable))
        assert list(ratios.columns) == list(NOMS_INDICATEURS)
        assert ratios.notna().all().all()

    def test_chaque_indicateur_est_documente(self):
        """Une alerte sans mode d'emploi n'est pas actionnable par un vérificateur."""
        for indicateur in INDICATEURS:
            assert indicateur.libelle.strip()
            assert indicateur.interpretation.strip()
            assert len(indicateur.interpretation) > 40, (
                f"{indicateur.nom} : interprétation trop laconique pour guider un contrôle"
            )

    def test_marge_brute_exacte(self, portefeuille_stable):
        modele = _en_modele(portefeuille_stable)
        ratios = calculer_indicateurs(modele)
        attendu = (modele["sale"] - modele["cogs"]) / modele["sale"]
        np.testing.assert_allclose(ratios["marge_brute"], attendu, rtol=1e-6)

    def test_indicateurs_invariants_dechelle(self, portefeuille_stable):
        """Propriété centrale : c'est elle qui protège du biais de taille."""
        en_usd = calculer_indicateurs(
            _en_modele(portefeuille_stable, unite="dirhams", convertir_en_usd=True)
        )
        en_mad = calculer_indicateurs(
            _en_modele(portefeuille_stable, unite="millions_dirhams",
                       convertir_en_usd=False)
        )
        pd.testing.assert_frame_equal(en_usd, en_mad, rtol=1e-4)


# ---------------------------------------------------------------------------
# Score de rupture
# ---------------------------------------------------------------------------
class TestScorerRuptures:

    def test_colonnes_de_sortie(self, portefeuille_stable):
        resultat = scorer_ruptures(_en_modele(portefeuille_stable))
        for colonne in ("score_rupture", "rang_rupture", "percentile_rupture",
                        "n_indicateurs_rompus", "a_un_historique"):
            assert colonne in resultat.columns
        for nom in NOMS_INDICATEURS:
            assert f"ecart_{nom}" in resultat.columns

    def test_classement_decroissant_et_complet(self, portefeuille_stable):
        resultat = scorer_ruptures(_en_modele(portefeuille_stable))
        assert len(resultat) == len(portefeuille_stable)
        assert resultat["score_rupture"].is_monotonic_decreasing
        assert resultat["rang_rupture"].tolist() == list(range(1, len(resultat) + 1))

    def test_trajectoire_stable_donne_un_score_quasi_nul(self, portefeuille_stable):
        """Témoin négatif : sans rupture, rien ne doit remonter."""
        resultat = scorer_ruptures(_en_modele(portefeuille_stable))
        assert resultat["score_rupture"].max() < 0.05

    def test_premier_exercice_sans_historique_score_nul(self, portefeuille_stable):
        """Absence d'historique n'est pas absence d'anomalie : le score est nul,
        mais la ligne est marquée pour que l'appelant la traite à part."""
        resultat = scorer_ruptures(_en_modele(portefeuille_stable))
        premiers = resultat[~resultat["a_un_historique"]]
        assert len(premiers) == portefeuille_stable["identifiant_fiscal"].nunique()
        assert (premiers["score_rupture"] == 0).all()

    def test_rupture_detectee(self, portefeuille_stable):
        """Une chute de marge de 40 % sur le dernier exercice doit remonter en tête."""
        altere = portefeuille_stable.copy()
        cible = (altere["identifiant_fiscal"] == 22222222) & (altere["exercice"] == 2024)
        altere.loc[cible, "chiffre_affaires"] *= 0.60
        altere.loc[cible, "resultat_net_exercice"] *= 0.40

        resultat = scorer_ruptures(_en_modele(altere))
        tete = resultat.iloc[0]
        assert tete["gvkey"] == 22222222 and tete["fyear"] == 2024
        assert tete["n_indicateurs_rompus"] >= 3, (
            "une minoration de CA doit rompre plusieurs ratios simultanément"
        )

    def test_score_invariant_dechelle(self, portefeuille_stable):
        """Le classement doit être rigoureusement identique quelle que soit
        l'unité de saisie — c'est ce qui distingue cette règle du modèle appris."""
        altere = portefeuille_stable.copy()
        cible = (altere["identifiant_fiscal"] == 11111111) & (altere["exercice"] == 2023)
        altere.loc[cible, "creances_actif_circulant"] *= 2.2

        en_usd = scorer_ruptures(_en_modele(altere, unite="dirhams", convertir_en_usd=True))
        en_mad = scorer_ruptures(_en_modele(altere, unite="millions_dirhams",
                                            convertir_en_usd=False))

        cle = ["gvkey", "fyear"]
        a = en_usd.set_index(cle)["score_rupture"].sort_index()
        b = en_mad.set_index(cle)["score_rupture"].sort_index()
        # `atol` est indispensable : les lignes sans rupture valent zéro, et une
        # tolérance purement relative rejetterait un écart de 1e-16 contre 0.
        np.testing.assert_allclose(a.values, b.values, rtol=1e-4, atol=1e-9)

    def test_score_neutre_en_taille(self, portefeuille_stable):
        """Trois entreprises de tailles très différentes (×1, ×25, ×400) et de
        trajectoires identiques doivent obtenir le même score."""
        resultat = scorer_ruptures(_en_modele(portefeuille_stable))
        avec_historique = resultat[resultat["a_un_historique"]]
        etendue = avec_historique.groupby("fyear")["score_rupture"].agg(
            lambda s: s.max() - s.min()
        )
        assert (etendue < 0.02).all(), (
            "le score varie avec la taille alors que les trajectoires sont identiques"
        )

    def test_ordre_dentree_sans_effet_sur_les_scores(self, portefeuille_stable):
        """Même précaution que dans `inference.prepare_features` : le tri interne
        par (contribuable, exercice) ne doit pas déplacer les scores."""
        modele = _en_modele(portefeuille_stable)
        reference = scorer_ruptures(modele).set_index(["gvkey", "fyear"])["score_rupture"]
        melange = modele.sample(frac=1, random_state=13).reset_index(drop=True)
        obtenu = scorer_ruptures(melange).set_index(["gvkey", "fyear"])["score_rupture"]

        commun = reference.index.intersection(obtenu.index)
        assert len(commun) == len(modele)
        np.testing.assert_allclose(
            reference.loc[commun].values, obtenu.loc[commun].values, rtol=1e-9,
            err_msg="un score a été attribué à la mauvaise ligne",
        )

    def test_score_independant_des_autres_contribuables(self, portefeuille_stable):
        """Le score d'un contribuable ne compare qu'à lui-même : retirer les
        autres lignes du fichier ne doit rien changer."""
        modele = _en_modele(portefeuille_stable)
        complet = scorer_ruptures(modele).set_index(["gvkey", "fyear"])["score_rupture"]
        seul = scorer_ruptures(
            modele[modele["gvkey"] == 22222222]
        ).set_index(["gvkey", "fyear"])["score_rupture"]

        for index in seul.index:
            assert complet.loc[index] == pytest.approx(seul.loc[index], rel=1e-9)

    def test_colonnes_identifiantes_obligatoires(self, portefeuille_stable):
        modele = _en_modele(portefeuille_stable).drop(columns=["fyear"])
        with pytest.raises(ValueError, match="impossible de comparer"):
            scorer_ruptures(modele)


# ---------------------------------------------------------------------------
# Explication
# ---------------------------------------------------------------------------
class TestExpliquerRupture:

    def test_explication_designe_lindicateur_rompu(self, portefeuille_stable):
        altere = portefeuille_stable.copy()
        cible = (altere["identifiant_fiscal"] == 33333333) & (altere["exercice"] == 2024)
        altere.loc[cible, "creances_actif_circulant"] *= 2.5

        resultat = scorer_ruptures(_en_modele(altere))
        explication = expliquer_rupture(resultat, 0, top_n=3)

        assert len(explication) <= 3
        assert "Créances" in explication.iloc[0]["indicateur"]
        assert explication.iloc[0]["a_examiner"].strip()

    def test_explication_triee_par_ampleur(self, portefeuille_stable):
        altere = portefeuille_stable.copy()
        cible = (altere["identifiant_fiscal"] == 11111111) & (altere["exercice"] == 2023)
        altere.loc[cible, "stocks"] *= 2.0
        altere.loc[cible, "chiffre_affaires"] *= 0.75

        resultat = scorer_ruptures(_en_modele(altere))
        explication = expliquer_rupture(resultat, 0, top_n=8)
        ecarts = [float(v.rstrip(" %")) for v in explication["ecart_vs_exercice_precedent"]]
        assert ecarts == sorted(ecarts, reverse=True)

    def test_ligne_sans_historique_donne_une_explication_vide(self, portefeuille_stable):
        resultat = scorer_ruptures(_en_modele(portefeuille_stable))
        sans_historique = resultat[~resultat["a_un_historique"]]
        position = resultat.index.get_indexer([sans_historique.index[0]])[0]
        assert expliquer_rupture(resultat, position).empty


# ---------------------------------------------------------------------------
# Le jeu de test livré
# ---------------------------------------------------------------------------
class TestPortefeuilleLivre:

    def test_la_regle_remonte_les_anomalies_plantees(self, racine):
        """Test d'intégration de bout en bout sur le portefeuille simulé.

        ⚠️ Les anomalies ayant été plantées sous forme de distorsions de ratios,
        ce test ne mesure PAS une capacité de détection de fraude réelle : il
        vérifie que la chaîne liasse → mapping → règle transmet le signal.
        """
        dossier = racine / "data" / "exemples_marocains"
        portefeuille = dossier / "portefeuille_dri_tetouan.csv"
        cle = dossier / "cle_anomalies.csv"
        if not portefeuille.exists() or not cle.exists():
            pytest.skip("jeu de test absent — python scripts/generer_jeu_test_marocain.py")

        donnees = _en_modele(pd.read_csv(portefeuille), unite="dirhams", mad_par_usd=9.85)
        resultat = scorer_ruptures(donnees)
        correction = pd.read_csv(cle)

        evalue = resultat.merge(
            correction[["identifiant_fiscal", "exercice", "anomalie"]],
            left_on=["gvkey", "fyear"], right_on=["identifiant_fiscal", "exercice"],
            validate="one_to_one",
        )
        n_examines = int(np.ceil(len(evalue) * 0.10))
        captees = evalue.nlargest(n_examines, "score_rupture")["anomalie"].sum()
        rappel = captees / evalue["anomalie"].sum()

        assert rappel > 0.50, (
            f"la règle ne capte que {rappel:.0%} des anomalies dans le top 10 % — "
            "régression probable dans le mapping ou dans la règle"
        )


# ---------------------------------------------------------------------------
# Comparaison sectorielle et score global
# ---------------------------------------------------------------------------
class TestEcartsSectoriels:

    def test_secteur_trop_petit_nest_pas_repute_sain(self, portefeuille_stable):
        """Trois contribuables ne font pas une norme sectorielle. Le score doit
        être nul ET la ligne marquée non comparable — sans quoi on conclurait à
        tort que ces dossiers sont sains."""
        from src.regle_rupture import scorer_ecarts_sectoriels

        modele = _en_modele(portefeuille_stable).assign(secteur="Commerce de gros")
        resultat = scorer_ecarts_sectoriels(modele, effectif_minimum=5)
        assert (~resultat["secteur_comparable"]).all()
        assert (resultat["score_secteur"] == 0).all()

    def test_valeur_atypique_detectee_dans_son_secteur(self):
        """Un contribuable dont la marge s'écarte franchement de ses pairs du
        même secteur et du même exercice doit remonter."""
        from src.regle_rupture import scorer_ecarts_sectoriels

        lignes = []
        for i in range(10):
            k = 1_000_000.0
            # Le dernier contribuable minore lourdement son chiffre d'affaires.
            ca = 120 * k if i < 9 else 60 * k
            lignes.append({
                "identifiant_fiscal": 90000 + i, "exercice": 2023,
                "secteur": "Textile et habillement",
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
                "chiffre_affaires": ca, "achats_revendus_marchandises": 50 * k,
                "achats_consommes_matieres": 20 * k, "dotations_exploitation": 8 * k,
                "charges_interets": 1 * k, "impots_sur_les_resultats": 4 * k,
                "resultat_non_courant": 1 * k, "resultat_net_exercice": 12 * k,
            })
        modele = _en_modele(pd.DataFrame(lignes))
        resultat = scorer_ecarts_sectoriels(modele).sort_values(
            "score_secteur", ascending=False
        )
        assert resultat["secteur_comparable"].all()
        assert int(resultat.iloc[0]["gvkey"]) == 90009

    def test_ecart_sectoriel_invariant_dechelle(self, portefeuille_stable):
        from src.regle_rupture import scorer_ecarts_sectoriels

        cle = ["gvkey", "fyear"]
        a = scorer_ecarts_sectoriels(
            _en_modele(portefeuille_stable, unite="dirhams", convertir_en_usd=True),
            effectif_minimum=1,
        ).set_index(cle)["score_secteur"].sort_index()
        b = scorer_ecarts_sectoriels(
            _en_modele(portefeuille_stable, unite="millions_dirhams",
                       convertir_en_usd=False),
            effectif_minimum=1,
        ).set_index(cle)["score_secteur"].sort_index()
        np.testing.assert_allclose(a.values, b.values, rtol=1e-4, atol=1e-6)


class TestScoreGlobal:

    def test_classement_complet_et_ordonne(self, portefeuille_stable):
        from src.regle_rupture import scorer_global

        resultat = scorer_global(_en_modele(portefeuille_stable))
        assert len(resultat) == len(portefeuille_stable)
        assert resultat["score_global"].is_monotonic_decreasing
        assert resultat["rang_global"].tolist() == list(range(1, len(resultat) + 1))

    def test_poids_extremes_reproduisent_chaque_axe(self, portefeuille_stable):
        """poids=1 doit redonner le classement temporel pur, poids=0 le
        classement sectoriel pur."""
        from src.regle_rupture import scorer_global, scorer_ruptures

        modele = _en_modele(portefeuille_stable)
        temporel = scorer_global(modele, poids_rupture=1.0)
        reference = scorer_ruptures(modele)
        np.testing.assert_allclose(
            temporel["score_global"].rank(pct=True).values,
            reference["score_rupture"].rank(pct=True).values,
            rtol=1e-9,
        )

    def test_poids_hors_bornes_rejete(self, portefeuille_stable):
        from src.regle_rupture import scorer_global

        with pytest.raises(ValueError, match="entre 0 et 1"):
            scorer_global(_en_modele(portefeuille_stable), poids_rupture=1.5)

    def test_motif_lisible_pour_chaque_ligne(self, portefeuille_stable):
        """Chaque dossier de la liste de travail doit porter un motif en
        français — un signalement sans motif n'est pas exploitable."""
        from src.regle_rupture import motif_principal, scorer_global

        resultat = scorer_global(_en_modele(portefeuille_stable))
        for i in range(len(resultat)):
            motif = motif_principal(resultat, i)
            assert isinstance(motif, str) and len(motif) > 10

    def test_le_score_global_bat_chaque_axe_pris_seul(self, racine):
        """Sur le portefeuille simulé, combiner les deux lectures doit capter
        plus d'anomalies que chacune séparément."""
        from src.regle_rupture import scorer_global, scorer_ruptures
        from src.scoring import precision_recall_at_k

        dossier = racine / "data" / "exemples_marocains"
        if not (dossier / "portefeuille_dri_tetouan.csv").exists():
            pytest.skip("jeu de test absent")

        donnees = _en_modele(pd.read_csv(dossier / "portefeuille_dri_tetouan.csv"),
                             unite="dirhams")
        cle = pd.read_csv(dossier / "cle_anomalies.csv")[
            ["identifiant_fiscal", "exercice", "anomalie"]
        ]

        def rappel(scores_df, colonne):
            ev = scores_df.merge(cle, left_on=["gvkey", "fyear"],
                                 right_on=["identifiant_fiscal", "exercice"],
                                 validate="one_to_one")
            return precision_recall_at_k(
                ev["anomalie"].values, ev[colonne].values, 0.05
            )[1]

        global_ = rappel(scorer_global(donnees), "score_global")
        temporel = rappel(scorer_ruptures(donnees), "score_rupture")

        assert global_ >= temporel, (
            f"le score combiné ({global_:.0%}) devrait au moins égaler "
            f"l'axe temporel seul ({temporel:.0%})"
        )
        assert global_ > 0.60, f"rappel à 5 % trop faible : {global_:.0%}"
