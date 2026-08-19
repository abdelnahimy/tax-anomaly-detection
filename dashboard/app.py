"""Détection d'anomalies fiscales — Direction Régionale des Impôts de Tétouan.

Tableau de bord en **trois écrans**, et pas un de plus :

    Ciblage                → produire la liste des dossiers à vérifier
    Analyse d'un dossier   → pourquoi celui-ci remonte, et de combien
    Méthode et résultats   → ce qui a été testé, ce qui marche, ce qui échoue

Méthode retenue pour le ciblage
--------------------------------
Chaque contribuable est analysé sur trois axes, tous calculés à partir du seul
fichier déposé — aucune donnée externe n'est requise :

1. **à lui-même**, d'un exercice à l'autre (rupture de trajectoire) ;
2. **à ses pairs** du même secteur et du même exercice (écart sectoriel) ;
3. **à ce que sa propre liasse laisse attendre** — un modèle de reconstitution
   prédit le chiffre d'affaires, la charge d'impôt et le résultat, puis mesure
   l'écart avec le déclaré.

Le modèle entraîné sur JAR2020 n'est **délibérément pas** utilisé pour le
ciblage : appliqué à un portefeuille marocain, il classe par taille
d'entreprise et non par anomalie (docs/06_transferabilite.md). Le volet
apprentissage du ciblage est réentraîné sur les données marocaines déposées,
sans aucune étiquette de fraude (docs/07_modele_reconstitution.md). Le modèle
JAR2020 reste consultable dans le troisième écran, à titre de validation
méthodologique et de contre-exemple documenté.

Lancement :
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
import streamlit as st

from dashboard.theme import (
    ACCENT, ENCRE_MOYENNE, PALETTE, SERIE_2,
    entete, etape, injecter_css, note, style_axes,
)
from src.ciblage import METHODES, analyser_portefeuille
from src.config import MODELS_DIR, REPORTS_DIR
from src.inference import InferenceError, load_model
from src.liasse_mapping import (
    COLONNES_LIASSE, MAD_PAR_USD_DEFAUT, UNITES_SUPPORTEES, mapper_liasse,
    modele_csv_liasse, verifier_liasse,
)
from src.modele_marocain import expliquer_reconstitution
from src.regle_rupture import expliquer_rupture

st.set_page_config(
    page_title="Ciblage du contrôle fiscal — DRI Tétouan",
    page_icon=None,
    layout="wide",
)
injecter_css()

DOSSIER_EXEMPLES = _ROOT / "data" / "exemples_marocains"
LIBELLE_UNITE = {
    "dirhams": "MAD",
    "milliers_dirhams": "milliers MAD",
    "millions_dirhams": "millions MAD",
}

# État partagé entre les écrans. Streamlit exécute le corps de tous les onglets
# à chaque interaction, dans l'ordre du script : l'écran Ciblage remplit ce
# dictionnaire, les deux suivants le lisent.
ETAT: dict[str, object] = {}


@st.cache_data(show_spinner=False)
def lire_rapport(nom: str) -> pd.DataFrame | None:
    chemin = REPORTS_DIR / nom
    if not chemin.exists():
        return None
    return pd.read_csv(chemin, index_col=0)


# Le modèle JAR2020 n'est requis par aucun des trois écrans : son absence est
# signalée là où elle se voit, elle n'empêche jamais l'application de démarrer.
try:
    _, META_JAR = load_model()
    JAR_DISPONIBLE = True
except InferenceError:
    META_JAR, JAR_DISPONIBLE = {}, False


# ==========================================================================
# En-tête et barre latérale
# ==========================================================================
entete(
    "Ciblage du contrôle fiscal",
    "Aide à la programmation des vérifications de comptabilité. À partir des "
    "liasses fiscales déposées, l'outil produit un classement de risque destiné "
    "à orienter une capacité de vérification limitée.",
    mention="Direction Générale des Impôts · Direction Régionale de Tétouan",
)

with st.sidebar:
    st.markdown("### Méthode")
    st.markdown(
        "Chaque contribuable est comparé sur **trois axes**, tous établis à "
        "partir de votre seul fichier.\n\n"
        "**1. À lui-même** — ses ratios ont-ils brusquement changé par rapport "
        "à l'exercice précédent ?\n\n"
        "**2. À son secteur** — s'écarte-t-il de ses pairs, la même année ?\n\n"
        "**3. À ce que sa liasse laisse attendre** — un modèle reconstitue son "
        "chiffre d'affaires, sa charge d'impôt et son résultat à partir du "
        "reste de sa liasse. Déclare-t-il nettement moins ?\n\n"
        "Plus le portefeuille déposé est large, plus les normes sont fiables."
    )
    st.divider()

    st.markdown("### Ratios surveillés")
    st.caption(
        "Marge brute · taux d'imposition effectif · impôt sur chiffre "
        "d'affaires · créances sur chiffre d'affaires · stocks sur chiffre "
        "d'affaires · écritures d'ajustement du résultat · rotation de l'actif "
        "· fournisseurs sur achats"
    )
    st.divider()

    st.markdown("### Portée juridique")
    note(
        "Un score élevé signale un dossier <b>à examiner</b>, jamais une fraude "
        "établie. Il n'a aucune valeur probante et ne peut fonder aucun "
        "redressement (art. 220 et s. du CGI). La décision de vérifier "
        "appartient au chef de brigade.",
        "attention",
    )

ecran_ciblage, ecran_dossier, ecran_methode = st.tabs(
    ["Ciblage", "Analyse d'un dossier", "Méthode et résultats"]
)


# ==========================================================================
# Écran 1 — Ciblage
# ==========================================================================
with ecran_ciblage:
    etape(1, "Déposer les liasses fiscales")

    colonne_depot, colonne_format = st.columns([3, 2])

    with colonne_depot:
        fichier_cib = st.file_uploader(
            "Fichier CSV — un enregistrement par (contribuable, exercice)",
            type=["csv"], key="cib_upload",
        )
        exemple_cib = st.checkbox(
            "Utiliser le portefeuille d'exemple (220 contribuables fictifs)",
            value=False, key="cib_exemple",
        )

    with colonne_format:
        with st.expander("Quel format de fichier ?"):
            st.markdown(
                "Les postes du **bilan** (tableaux n° 1 et 2) et du **CPC** "
                "(tableau n° 3), au format CGNC.\n\n"
                "Deux colonnes obligatoires : `identifiant_fiscal` et "
                "`exercice`. Accents et majuscules sont tolérés dans les "
                "en-têtes.\n\n"
                "**Fournir au moins 3 exercices par contribuable** : c'est la "
                "comparaison d'une année à l'autre qui porte l'essentiel du "
                "signal."
            )
            st.download_button(
                "Télécharger le modèle de fichier vide",
                modele_csv_liasse().to_csv(index=False).encode("utf-8-sig"),
                file_name="modele_liasses.csv", mime="text/csv",
                width="stretch", key="cib_dl_modele",
            )
            st.code(", ".join(COLONNES_LIASSE), language="text")

    liasses_cib = None
    if fichier_cib is not None:
        try:
            liasses_cib = pd.read_csv(fichier_cib)
        except Exception as erreur_lecture:  # noqa: BLE001
            st.error(f"Lecture du fichier impossible : {erreur_lecture}")
    elif exemple_cib:
        chemin_cib = DOSSIER_EXEMPLES / "portefeuille_dri_tetouan.csv"
        if chemin_cib.exists():
            liasses_cib = pd.read_csv(chemin_cib)
            note(
                "<b>Portefeuille d'exemple chargé</b> — 220 contribuables "
                "fictifs de la région, sur 4 exercices. Aucune entreprise "
                "réelle. Le fichier se régénère avec "
                "<code>python scripts/generer_jeu_test_marocain.py</code>.",
                "reference",
            )
        else:
            st.error(
                "Portefeuille d'exemple introuvable. Lancez d'abord :\n\n"
                "`python scripts/generer_jeu_test_marocain.py`"
            )

    if liasses_cib is None:
        note(
            "Déposez un fichier, ou cochez la case ci-dessus pour travailler "
            "sur le portefeuille d'exemple.",
        )
    else:
        diagnostic_cib = verifier_liasse(liasses_cib)
        for anomalie_cib in diagnostic_cib["anomalies"]:
            st.error(f"**Fichier inexploitable** — {anomalie_cib}")
        for avertissement_cib in diagnostic_cib["avertissements"]:
            st.warning(avertissement_cib)
        if not diagnostic_cib["scorable"]:
            st.stop()

        with st.expander("Unité de saisie (sans effet sur le classement)"):
            unite_cib = st.selectbox(
                "Les montants du fichier sont exprimés en :",
                list(UNITES_SUPPORTEES), index=0, key="cib_unite",
            )
            st.caption(
                "Les huit ratios utilisés sont des rapports entre deux postes : "
                "ils ne dépendent ni de l'unité, ni de la devise. Le classement "
                "est identique quelle que soit la réponse."
            )

        donnees_cib, rapport_cib = mapper_liasse(
            liasses_cib, unite=unite_cib, mad_par_usd=MAD_PAR_USD_DEFAUT,
            convertir_en_usd=False,
        )

        if rapport_cib["colonnes_ignorees"]:
            st.caption(
                "Colonnes non reconnues, ignorées : "
                + ", ".join(f"`{c}`" for c in rapport_cib["colonnes_ignorees"][:12])
                + (" …" if len(rapport_cib["colonnes_ignorees"]) > 12 else "")
            )

        st.divider()
        etape(2, "Déclarer la capacité de contrôle")

        n_lignes_cib = len(donnees_cib)
        n_contribuables_cib = int(donnees_cib["gvkey"].nunique())

        colonne_curseur, colonne_chiffres = st.columns([3, 2])
        with colonne_curseur:
            part_cib = st.slider(
                "Part du portefeuille que la brigade peut vérifier",
                min_value=1, max_value=30, value=5, step=1, format="%d %%",
                key="cib_part",
            )
            n_dossiers_cib = max(1, int(round(n_lignes_cib * part_cib / 100)))
        with colonne_chiffres:
            mesure_a, mesure_b, mesure_c = st.columns(3)
            mesure_a.metric("Contribuables", f"{n_contribuables_cib:,}".replace(",", " "))
            mesure_b.metric("Liasses", f"{n_lignes_cib:,}".replace(",", " "))
            mesure_c.metric("À vérifier", f"{n_dossiers_cib:,}".replace(",", " "))

        cles_methodes_cib = list(METHODES)
        methode_cib = st.radio(
            "Méthode d'analyse", cles_methodes_cib,
            index=cles_methodes_cib.index("combine"),
            format_func=lambda cle: METHODES[cle], horizontal=True,
            key="cib_methode",
        )

        st.divider()
        etape(3, f"Liste de travail — {n_dossiers_cib} dossiers prioritaires")

        with st.spinner("Analyse en cours — entraînement du modèle sur le portefeuille…"):
            classement_cib, diag_cib = analyser_portefeuille(
                donnees_cib, methode=methode_cib
            )

        if methode_cib != "regles" and not diag_cib["ia_disponible"]:
            note(
                "<b>Le volet apprentissage n'a pas pu être entraîné</b> — "
                + diag_cib.get("ia_indisponible_motif", "").split("\n")[0]
                + " L'analyse se poursuit avec les règles seules. Déposez un "
                "portefeuille plus large pour activer le modèle.",
                "attention",
            )
        elif diag_cib["ia_disponible"]:
            qualites_cib = " · ".join(
                f"{d['libelle']} : R² = {d['r2']:.2f}"
                for d in diag_cib["diagnostics_ia"].values() if "r2" in d
            )
            st.caption(
                f"Modèle entraîné sur vos {diag_cib['n_contribuables']} "
                f"contribuables, par validation croisée — {qualites_cib}. "
                "Chaque dossier est prédit par un modèle qui ne l'a jamais vu."
            )

        liste_cib = classement_cib.head(n_dossiers_cib).copy()
        sans_historique_cib = int((~classement_cib["a_un_historique"]).sum())

        colonnes_liste_cib = [c for c in [
            "rang", "raison_sociale", "gvkey", "secteur", "commune", "fyear",
            "motif", "n_indicateurs_rompus", "n_ratios_atypiques",
            "n_cibles_sous_declarees",
        ] if c in liste_cib.columns]

        st.dataframe(
            liste_cib[colonnes_liste_cib].rename(columns={
                "rang": "N°", "raison_sociale": "Contribuable",
                "gvkey": "Identifiant fiscal", "secteur": "Secteur",
                "commune": "Commune", "fyear": "Exercice",
                "motif": "Motif du signalement",
                "n_indicateurs_rompus": "Ruptures",
                "n_ratios_atypiques": "Atypies",
                "n_cibles_sous_declarees": "Sous-décl.",
            }),
            hide_index=True, width="stretch",
            height=min(560, 60 + 35 * len(liste_cib)),
            column_config={
                "N°": st.column_config.NumberColumn("N°", format="%d", width="small"),
                "Identifiant fiscal": st.column_config.NumberColumn(
                    "Identifiant fiscal", format="%d"),
                "Exercice": st.column_config.NumberColumn(
                    "Exercice", format="%d", width="small"),
                "Ruptures": st.column_config.NumberColumn(
                    "Ruptures", format="%d", width="small",
                    help="Ratios ayant varié de plus de 30 % par rapport à "
                         "l'exercice précédent"),
                "Atypies": st.column_config.NumberColumn(
                    "Atypies", format="%d", width="small",
                    help="Ratios très éloignés de la médiane du secteur"),
                "Sous-décl.": st.column_config.NumberColumn(
                    "Sous-décl.", format="%d", width="small",
                    help="Grandeurs (CA, impôt, résultat) déclarées nettement "
                         "en deçà de ce que le modèle attendait"),
            },
        )

        colonne_export, colonne_reserve = st.columns([1, 3])
        with colonne_export:
            st.download_button(
                "Télécharger la liste",
                liste_cib.to_csv(index=False).encode("utf-8-sig"),
                file_name="liste_de_travail.csv", mime="text/csv",
                width="stretch", key="cib_dl_liste",
            )
        with colonne_reserve:
            if sans_historique_cib:
                st.caption(
                    f"**{sans_historique_cib} liasses sans exercice de "
                    "comparaison** — la première année d'observation de chaque "
                    "contribuable. Leur trajectoire ne peut pas être analysée : "
                    "ne les réputez pas saines, traitez-les séparément."
                )

        note(
            "Le détail de chaque dossier — ce qui a changé, et de combien — se "
            "trouve dans l'écran <b>Analyse d'un dossier</b>.",
            "reference",
        )

        ETAT.update({
            "liste": liste_cib,
            "diag": diag_cib,
            "rapport": rapport_cib,
            "unite": unite_cib,
        })


# ==========================================================================
# Écran 2 — Analyse d'un dossier
# ==========================================================================
with ecran_dossier:
    if "liste" not in ETAT:
        note(
            "Aucun portefeuille n'est chargé. Déposez un fichier dans l'écran "
            "<b>Ciblage</b> : les dossiers signalés apparaîtront ici.",
        )
    else:
        liste_dos = ETAT["liste"]
        diag_dos = ETAT["diag"]
        rapport_dos = ETAT["rapport"]
        unite_dos = ETAT["unite"]

        libelles_dos = []
        for position in range(len(liste_dos)):
            ligne_dos = liste_dos.iloc[position]
            nom_dos = ligne_dos.get("raison_sociale")
            if not isinstance(nom_dos, str) or not nom_dos.strip():
                nom_dos = f"contribuable {int(ligne_dos['gvkey'])}"
            libelles_dos.append(
                f"N° {int(ligne_dos['rang'])} — {nom_dos} — exercice "
                f"{int(ligne_dos['fyear'])}"
            )

        choix_dos = st.selectbox(
            "Dossier à analyser", range(len(libelles_dos)),
            format_func=lambda i: libelles_dos[i], key="dos_choix",
        )
        dossier_dos = liste_dos.iloc[choix_dos]

        st.markdown(f"#### {dossier_dos.get('motif', 'Dossier signalé')}")

        facteur_dos = rapport_dos["facteur_echelle"] or 1.0
        unite_lisible_dos = LIBELLE_UNITE[unite_dos]

        def _montant_dos(cle: str) -> str:
            valeur = dossier_dos.get(cle)
            if pd.isna(valeur):
                return "non renseigné"
            return f"{valeur / facteur_dos:,.0f}".replace(",", " ")

        montant_a, montant_b, montant_c = st.columns(3)
        montant_a.metric(f"Chiffre d'affaires ({unite_lisible_dos})", _montant_dos("sale"))
        montant_b.metric(f"Résultat net ({unite_lisible_dos})", _montant_dos("ni"))
        montant_c.metric(f"Impôt sur les résultats ({unite_lisible_dos})", _montant_dos("txt"))

        # --- Ce que la liasse laissait attendre ---------------------------
        if diag_dos["ia_disponible"]:
            reconstitution_dos = expliquer_reconstitution(liste_dos, choix_dos)
            if not reconstitution_dos.empty:
                st.markdown("##### Ce que la liasse laissait attendre")
                affichage_dos = reconstitution_dos.copy()
                for colonne_dos in ("declare", "attendu"):
                    affichage_dos[colonne_dos] = (
                        affichage_dos[colonne_dos] / facteur_dos
                    ).map(lambda v: f"{v:,.0f}".replace(",", " "))
                affichage_dos["ecart_pct"] = affichage_dos["ecart_pct"].map(
                    lambda v: "—" if pd.isna(v) else f"{v:+.0f} %"
                )
                st.dataframe(
                    affichage_dos.rename(columns={
                        "grandeur": "Grandeur",
                        "declare": f"Déclaré ({unite_lisible_dos})",
                        "attendu": f"Attendu ({unite_lisible_dos})",
                        "ecart_pct": "Écart", "lecture": "Lecture",
                    }),
                    hide_index=True, width="stretch",
                    column_config={
                        "Lecture": st.column_config.TextColumn(
                            "Lecture", width="large"),
                    },
                )
                st.caption(
                    "« Attendu » est le montant prédit à partir du reste de la "
                    "liasse — appareil productif, stocks, achats, créances, "
                    "secteur — par un modèle qui n'a jamais vu ce dossier à "
                    "l'entraînement. C'est la transposition, en apprentissage "
                    "automatique, de la reconstitution du chiffre d'affaires "
                    "(art. 213 du CGI). Un écart n'établit pas une fraude : "
                    "sinistre, démarrage, sous-traitance massive ou changement "
                    "d'activité peuvent l'expliquer."
                )

        # --- Ce qui a changé ------------------------------------------------
        detail_dos = expliquer_rupture(liste_dos, choix_dos, top_n=5)
        st.markdown("##### Ce qui a changé par rapport à l'exercice précédent")
        if detail_dos.empty:
            note(
                "Ce contribuable n'a pas d'exercice antérieur dans le fichier : "
                "aucune rupture de trajectoire ne peut être mesurée. S'il "
                "figure dans la liste, c'est au titre de son écart au secteur.",
            )
        else:
            st.dataframe(
                detail_dos.rename(columns={
                    "indicateur": "Ratio",
                    "valeur_exercice": "Valeur de l'exercice",
                    "ecart_vs_exercice_precedent": "Variation",
                    "a_examiner": "Ce qu'il faut vérifier",
                }),
                hide_index=True, width="stretch",
                column_config={
                    "Ce qu'il faut vérifier": st.column_config.TextColumn(
                        "Ce qu'il faut vérifier", width="large"),
                },
            )

        if dossier_dos.get("secteur_comparable", False):
            st.caption(
                f"Comparaison sectorielle établie sur "
                f"{int(dossier_dos['effectif_secteur'])} dossiers du secteur "
                f"« {dossier_dos.get('secteur', 'non renseigné')} », exercice "
                f"{int(dossier_dos['fyear'])}."
            )
        else:
            st.caption(
                "Secteur trop peu représenté dans le fichier pour établir une "
                "norme sectorielle : seule la comparaison à soi-même a été "
                "utilisée."
            )

        note(
            "<b>Le calcul est l'explication.</b> Contrairement à un modèle "
            "d'apprentissage, il n'y a ici aucune approximation : les écarts "
            "affichés <i>sont</i> ce qui a produit le classement. Un "
            "vérificateur qui connaît une explication légitime — litige "
            "commercial, report déficitaire, changement de méthode comptable, "
            "sinistre — peut écarter le dossier en connaissance de cause.",
            "reference",
        )


# ==========================================================================
# Écran 3 — Méthode et résultats
# ==========================================================================
with ecran_methode:
    st.markdown("#### Le modèle américain ne se transfère pas")
    st.markdown(
        "Un modèle XGBoost entraîné sur **JAR2020** — sociétés américaines "
        "cotées, fraudes sanctionnées par la SEC — atteint une AUC-PR de "
        "0,0168 sur son propre terrain, soit environ 2,5 fois le hasard. "
        "Appliqué à un portefeuille marocain, il s'effondre : il trie par "
        "**taille d'entreprise**, pas par anomalie."
    )

    resultat_met = pd.DataFrame(
        {
            "Corrélation avec l'anomalie": [-0.026, 0.300],
            "Corrélation avec la taille": [0.592, 0.016],
            "Anomalies captées dans le top 5 %": ["11,8 %", "52,9 %"],
            "AUC-ROC": [0.460, 0.946],
        },
        index=["Modèle JAR2020", "Règle de rupture"],
    )
    st.dataframe(resultat_met, width="stretch")
    st.caption(
        "Portefeuille simulé de 880 liasses, 34 lignes porteuses d'une anomalie "
        "plantée. Régénérable par `python scripts/evaluer_jeu_test_marocain.py`."
    )

    note(
        "<b>La cause est structurelle, pas conjoncturelle.</b> Le total de "
        "bilan médian de JAR2020 avoisine 105 millions de dollars, contre "
        "environ 1 million pour un portefeuille de direction régionale "
        "marocaine. Les contribuables marocains se massent dans la queue basse "
        "de la distribution d'entraînement — et comme 48 des 54 variables du "
        "modèle sont des variables de <b>niveau</b>, celui-ci trie par ordre de "
        "grandeur. Analyse complète : <code>docs/06_transferabilite.md</code>.",
        "attention",
    )

    st.divider()

    # --- Règles contre modèle appris ---------------------------------------
    st.markdown("#### Ce que le modèle appris apporte, et ce qu'il n'apporte pas")
    st.markdown(
        "Faute d'étiquette de fraude au Maroc, le modèle de reconstitution "
        "apprend à prédire **ce qu'un contribuable aurait normalement déclaré**, "
        "puis mesure l'écart. Comparé à la règle sur le même portefeuille, il "
        "perd globalement — mais l'écart se referme sur le sous-type qu'il est "
        "précisément fait pour détecter."
    )

    comparaison_met = lire_rapport("comparaison_regles_ia_marocain.csv")
    if comparaison_met is None:
        note(
            "Tableau comparatif absent de <code>reports/</code>. Exécutez le "
            "notebook <code>07_modele_marocain_reconstitution.ipynb</code>.",
        )
    else:
        colonnes_met = [
            c for c in comparaison_met.columns
            if "AUC-ROC" in c or "Rappel" in c
        ]
        st.dataframe(
            comparaison_met[colonnes_met].round(3), width="stretch"
        )

        # Deux séries, barres fines, valeurs précises laissées au tableau
        # ci-dessus : le graphique porte la comparaison, pas les chiffres.
        # Légende sortie du tracé — posée dessus, elle recouvrait les barres.
        figure_met, axes_met = plt.subplots(figsize=(8, 3.3))
        style_axes(axes_met)
        etiquettes_met = list(comparaison_met.index)
        positions_met = list(range(len(etiquettes_met)))
        hauteur_met = 0.28
        minorations_met = comparaison_met["AUC-ROC — minoration"]
        autres_met = comparaison_met["AUC-ROC — autre"]

        axes_met.barh(
            [p + hauteur_met / 2 + 0.03 for p in positions_met], minorations_met,
            height=hauteur_met, color=ACCENT, label="Minoration déclarative",
        )
        axes_met.barh(
            [p - hauteur_met / 2 - 0.03 for p in positions_met], autres_met,
            height=hauteur_met, color=SERIE_2, label="Autre irrégularité",
        )
        axes_met.set_yticks(positions_met)
        axes_met.set_yticklabels(etiquettes_met)
        axes_met.set_xlim(0, 1.0)
        axes_met.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axes_met.set_xticklabels(["0", "0,25", "0,50 — hasard", "0,75", "1,00"])
        axes_met.set_xlabel("AUC-ROC par type d'anomalie plantée")
        axes_met.legend(
            frameon=False, fontsize=9, labelcolor=ENCRE_MOYENNE, ncol=2,
            loc="upper center", bbox_to_anchor=(0.5, -0.24),
        )
        figure_met.tight_layout()
        st.pyplot(figure_met, width="stretch")
        plt.close(figure_met)

        note(
            "Le modèle de reconstitution reste <b>peu sensible</b> à ce qui "
            "n'affecte pas la base imposable déclarée — stocks surévalués, "
            "créances gonflées. Ce n'est pas une limite cachée : c'est la "
            "conséquence directe du choix des cibles. Sur les minorations "
            "déclaratives en revanche, il approche la règle, et il apporte ce "
            "qu'aucune règle ne donne : <b>un montant chiffré</b>, exploitable "
            "dans une procédure de reconstitution (art. 213 du CGI).",
            "reference",
        )

    st.divider()

    # --- Modèles entraînés --------------------------------------------------
    st.markdown("#### Modèles entraînés sur JAR2020")
    st.caption(
        "Résultats agrégés sur les douze années de test 2003-2014, moyenne "
        "pondérée par le nombre d'observations. Conservés parce qu'ils "
        "constituent la validation méthodologique du dispositif — et parce que "
        "leur échec sur données marocaines est un résultat en soi."
    )

    rapport_v2_met = lire_rapport("comparison_v2_all_models.csv")
    if rapport_v2_met is None:
        note(
            "Tableau absent de <code>reports/</code>. Exécutez les notebooks "
            "03 à 05.",
        )
    else:
        st.dataframe(rapport_v2_met, width="stretch")

        if "auc_pr" in rapport_v2_met.columns:
            figure_jar, axes_jar = plt.subplots(
                figsize=(8, max(2.2, 0.44 * len(rapport_v2_met)))
            )
            style_axes(axes_jar)
            donnees_jar = rapport_v2_met.sort_values("auc_pr")
            meilleur_jar = donnees_jar["auc_pr"].idxmax()
            couleurs_jar = [
                ACCENT if index == meilleur_jar else PALETTE["filet_fort"]
                for index in donnees_jar.index
            ]
            axes_jar.barh(
                donnees_jar.index.astype(str), donnees_jar["auc_pr"],
                color=couleurs_jar, height=0.6,
            )
            axes_jar.axvline(0.0066, color=SERIE_2, linewidth=1.4)
            axes_jar.annotate(
                "hasard (0,0066)", xy=(0.0066, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(5, 2), textcoords="offset points",
                color=SERIE_2, fontsize=9, annotation_clip=False,
            )
            for rang_jar, valeur_jar in enumerate(donnees_jar["auc_pr"]):
                axes_jar.annotate(
                    f"{valeur_jar:.4f}".replace(".", ","),
                    xy=(valeur_jar, rang_jar), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    color=ENCRE_MOYENNE, fontsize=9,
                )
            axes_jar.xaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:.4f}".replace(".", ","))
            )
            axes_jar.set_xlabel("AUC-PR — plus haut est meilleur")
            axes_jar.set_xlim(0, float(donnees_jar["auc_pr"].max()) * 1.24)
            figure_jar.tight_layout()
            st.pyplot(figure_jar, width="stretch")
            plt.close(figure_jar)

    if JAR_DISPONIBLE:
        st.caption(
            f"Modèle retenu : `models/xgboost_v2_best.pkl` — "
            f"{META_JAR['n_features']} variables "
            f"({len(META_JAR.get('temporal_features', []))} de trajectoire, "
            f"{len(META_JAR.get('fiscal_features', []))} ratios fiscaux métier), "
            f"validation temporelle avec écart de "
            f"{META_JAR.get('split_gap_years', 2)} ans."
        )
    else:
        st.caption(
            f"Modèle JAR2020 absent de `{MODELS_DIR.name}/` — les tableaux "
            "ci-dessus proviennent de `reports/` et restent lisibles. Exécutez "
            "le notebook 05 pour le régénérer."
        )

    # --- Interprétabilité ----------------------------------------------------
    figure_shap_met = REPORTS_DIR / "figures" / "04_shap_summary_full_model.png"
    if figure_shap_met.exists():
        st.divider()
        st.markdown("#### Ce qui fait le score — décomposition SHAP")
        st.markdown(
            "Un dossier signalé sans motif n'est pas actionnable. La "
            "décomposition SHAP attribue à chaque variable sa part dans le "
            "score, ce qui permet à un vérificateur d'écarter un dossier en "
            "connaissance de cause. **Un modèle contestable est un modèle "
            "utilisable ; une boîte noire ne l'est pas.**"
        )
        st.image(str(figure_shap_met), width="stretch")

    st.divider()
    st.caption(
        "Documentation méthodologique complète dans `docs/` — cadrage (01), "
        "correspondance de la liasse (02), transférabilité (06), modèle de "
        "reconstitution (07), liasses réelles de sociétés cotées (08). "
        "124 tests automatisés : `docs/05_guide_de_test.md`."
    )
