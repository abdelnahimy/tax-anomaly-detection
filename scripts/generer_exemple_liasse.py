"""Génère `dashboard/exemple_liasse_marocaine.csv` — jeu de liasses FICTIVES.

⚠️ Les entreprises, les montants et les scores qui en découlent sont
entièrement inventés. Ce fichier a deux usages, et deux seulement :

1. servir de **gabarit de saisie** rempli, plus parlant qu'un en-tête vide ;
2. démontrer que la **chaîne de traitement fonctionne** de bout en bout.

Il n'a **aucune valeur d'analyse**. Aucun contribuable réel n'est concerné, et
un score élevé sur ce fichier ne signifie strictement rien.

Le générateur est versionné dans le projet (plutôt que le CSV seul) pour que le
jeu soit reproductible et que la façon dont il a été construit soit vérifiable :
un jury peut voir qu'aucune donnée réelle n'a été utilisée.

Contrainte de construction : **le bilan de chaque ligne est équilibré**
(actif = passif), pour que le fichier passe les contrôles de cohérence de
`src.liasse_mapping.verifier_liasse()` sans avertissement. Un gabarit qui
déclencherait lui-même des alertes serait un mauvais gabarit.

Usage :
    python scripts/generer_exemple_liasse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.liasse_mapping import mapper_liasse, verifier_liasse

GRAINE = 42
EXERCICES = (2021, 2022, 2023)

# Entreprises fictives, choisies pour refléter la structure économique de la
# région Tanger-Tétouan-Al Hoceïma : agroalimentaire et pêche sur la côte,
# textile, BTP et tourisme balnéaire, négoce frontalier.
# (identifiant fiscal, raison sociale, secteur, commune, chiffre d'affaires initial)
ENTREPRISES = [
    (15204871, "CONSERVERIE DU DETROIT SA", "Industrie agroalimentaire", "Tetouan", 42_000_000),
    (15207334, "TEXTILE RIF SARL", "Textile et habillement", "Tetouan", 18_500_000),
    (15211902, "BTP MARTIL CONSTRUCTION SA", "Batiment et travaux publics", "Martil", 115_000_000),
    (15214650, "DISTRIBUTION NORD SARL", "Commerce de gros", "Tetouan", 67_000_000),
    (15218113, "HOTELIERE CABO NEGRO SA", "Hebergement et restauration", "M'diq", 88_000_000),
    (15220487, "TRANSPORT LOGISTIQUE TANGEROISE SARL", "Transport et logistique", "Fnideq", 31_000_000),
    (15223905, "PECHERIES OUED LAOU SARL", "Peche et transformation", "Oued Laou", 12_800_000),
    (15226240, "MATERIAUX CHEFCHAOUEN SA", "Negoce de materiaux", "Chefchaouen", 24_300_000),
]

SORTIE = Path(__file__).resolve().parent.parent / "dashboard" / "exemple_liasse_marocaine.csv"


def generer() -> pd.DataFrame:
    rng = np.random.default_rng(GRAINE)
    lignes = []

    for identifiant, raison_sociale, secteur, commune, ca_initial in ENTREPRISES:
        # Chaque entreprise a sa propre trajectoire, pour que les variables de
        # trajectoire (_d1 / _d2) portent un signal plutôt que du bruit uniforme.
        croissance = rng.uniform(0.02, 0.14)
        marge = rng.uniform(0.03, 0.11)
        rotation = rng.uniform(0.9, 1.6)  # chiffre d'affaires / total de bilan

        for decalage, exercice in enumerate(EXERCICES):
            ca = ca_initial * (1 + croissance) ** decalage * rng.uniform(0.97, 1.03)
            actif_total = ca / rotation

            # --- ACTIF ----------------------------------------------------
            immob_net = actif_total * rng.uniform(0.34, 0.52)
            immob_brut = immob_net * rng.uniform(1.45, 1.95)  # avant amortissements
            immob_financieres = actif_total * rng.uniform(0.005, 0.03)

            stocks = actif_total * rng.uniform(0.10, 0.24)
            creances = actif_total * rng.uniform(0.16, 0.30)
            titres_placement = actif_total * rng.uniform(0.0, 0.025)
            actif_circulant = stocks + creances + titres_placement
            tresorerie_actif = actif_total - immob_net - actif_circulant

            if tresorerie_actif < 0:
                # Rééquilibrage : on réduit les stocks plutôt que d'accepter une
                # trésorerie négative, qui n'existe pas à l'actif.
                stocks += tresorerie_actif - actif_total * 0.01
                actif_circulant = stocks + creances + titres_placement
                tresorerie_actif = actif_total * 0.01

            # --- PASSIF (bouclé sur actif_total) --------------------------
            cp_assimiles = actif_total * rng.uniform(0.0, 0.02)
            dettes_financement = actif_total * rng.uniform(0.08, 0.28)
            dettes_financement_ct = dettes_financement * rng.uniform(0.10, 0.30)
            provisions_durables = actif_total * rng.uniform(0.0, 0.02)
            tresorerie_passif = actif_total * rng.uniform(0.01, 0.07)
            fournisseurs = actif_total * rng.uniform(0.12, 0.26)

            resultat_net = ca * marge * rng.uniform(0.85, 1.15)
            impots = max(resultat_net, 0.0) * rng.uniform(0.24, 0.34)
            etat_is_dette = impots * rng.uniform(0.15, 0.55)
            autres_dettes = actif_total * rng.uniform(0.03, 0.09)
            passif_circulant = fournisseurs + etat_is_dette + autres_dettes

            capitaux_propres = (actif_total - cp_assimiles - dettes_financement
                                - provisions_durables - passif_circulant
                                - tresorerie_passif)

            # Une sous-capitalisation extrême serait invraisemblable : on la
            # corrige en réduisant l'endettement à long terme.
            plancher = actif_total * 0.12
            if capitaux_propres < plancher:
                manque = plancher - capitaux_propres
                dettes_financement = max(dettes_financement - manque, actif_total * 0.03)
                capitaux_propres = (actif_total - cp_assimiles - dettes_financement
                                    - provisions_durables - passif_circulant
                                    - tresorerie_passif)

            # Ventilation des capitaux propres (le report à nouveau absorbe le solde)
            capital_social = capitaux_propres * rng.uniform(0.35, 0.60)
            reserve_legale = min(capital_social * 0.10, capitaux_propres * 0.10)
            autres_reserves = capitaux_propres * rng.uniform(0.03, 0.12)
            report_a_nouveau = (capitaux_propres - capital_social - reserve_legale
                                - autres_reserves - resultat_net)

            # --- CPC ------------------------------------------------------
            achats_marchandises = ca * rng.uniform(0.30, 0.52)
            achats_matieres = ca * rng.uniform(0.10, 0.22)
            dotations = immob_brut * rng.uniform(0.06, 0.11)
            charges_interets = dettes_financement * rng.uniform(0.035, 0.065)
            resultat_non_courant = ca * rng.uniform(-0.006, 0.010)

            lignes.append({
                "identifiant_fiscal": identifiant,
                "exercice": exercice,
                "raison_sociale": raison_sociale,
                "secteur": secteur,
                "commune": commune,
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
                "chiffre_affaires": ca,
                "achats_revendus_marchandises": achats_marchandises,
                "achats_consommes_matieres": achats_matieres,
                "dotations_exploitation": dotations,
                "charges_interets": charges_interets,
                "impots_sur_les_resultats": impots,
                "resultat_non_courant": resultat_non_courant,
                "resultat_net_exercice": resultat_net,
            })

    df = pd.DataFrame(lignes)
    montants = df.select_dtypes("number").columns.drop(["identifiant_fiscal", "exercice"])
    df[montants] = df[montants].round(2)
    return df.sort_values(["identifiant_fiscal", "exercice"]).reset_index(drop=True)


def main() -> None:
    df = generer()

    # Le gabarit doit passer les contrôles sans la moindre alerte : un exemple
    # qui déclenche lui-même des avertissements est un mauvais exemple.
    diagnostic = verifier_liasse(df)
    if not diagnostic["scorable"]:
        raise SystemExit("Anomalies bloquantes :\n- " + "\n- ".join(diagnostic["anomalies"]))
    if diagnostic["avertissements"]:
        raise SystemExit("Avertissements :\n- " + "\n- ".join(diagnostic["avertissements"]))

    _, rapport = mapper_liasse(df, unite="dirhams")

    df.to_csv(SORTIE, index=False)
    print(f"Écrit : {SORTIE}")
    print(f"  {len(ENTREPRISES)} entreprises fictives × {len(EXERCICES)} exercices "
          f"= {len(df)} lignes")
    print("  Bilan équilibré, aucun avertissement")
    print(f"  Couverture après mapping : {rapport['taux_couverture_global_pct']} % "
          f"({rapport['n_variables_renseignees']}/28 variables)")
    print(f"  Variables hors liasse : {', '.join(rapport['variables_vides'])}")


if __name__ == "__main__":
    main()
