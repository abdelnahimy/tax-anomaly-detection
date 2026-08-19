# Guide de test

Comment vérifier que l'application fonctionne — et savoir quoi faire quand ce
n'est pas le cas.

Trois niveaux, du plus rapide au plus complet :

| Niveau | Durée | Ce qu'il vérifie |
|---|---|---|
| [1. Tests automatisés](#1-tests-automatisés) | ~10 s | Le code : correspondances comptables, invariants, conversions |
| [2. Parcours manuel du dashboard](#2-parcours-manuel-du-dashboard) | ~10 min | L'interface, écran par écran |
| [3. Évaluation sur portefeuille](#3-évaluation-sur-le-portefeuille-simulé) | ~2 min | Le comportement du ciblage à l'échelle |

---

## 0. Préparation

```bash
python -m pip install -r requirements.txt
python scripts/generer_jeu_test_marocain.py
```

Sortie attendue :

```
Portefeuille : data/exemples_marocains/portefeuille_dri_tetouan.csv
  220 contribuables × 4 exercices = 880 lignes
  14 secteurs · 12 communes
  Bilan équilibré, aucun avertissement
  Couverture après mapping : 85.7 % (24/28 variables)

Clé de correction : data/exemples_marocains/cle_anomalies.csv
  18 contribuables porteurs d'une anomalie (8.2%)
  34 lignes concernées (3.9%)
```

Trois fichiers sont produits dans `data/exemples_marocains/` :

| Fichier | Lignes | Usage |
|---|---|---|
| `portefeuille_dri_tetouan.csv` | 880 | Test à l'échelle. **Ne contient aucune étiquette** — c'est le fichier à déposer dans l'application |
| `cle_anomalies.csv` | 880 | Clé de correction : qui porte une anomalie, et laquelle |
| `echantillon_demo.csv` | 48 | 12 contribuables — prise en main rapide |

> Ces trois fichiers sont **entièrement fictifs**. Aucune entreprise, aucun
> identifiant fiscal, aucun montant ne correspond à une réalité.

Un **quatrième fichier, lui bien réel**, complète le dossier :

```bash
python scripts/collecte_ammc.py
```

| Fichier | Lignes | Usage |
|---|---|---|
| `societes_cotees_reelles.csv` | 26 | Comptes sociaux CGNC publiés de 12 sociétés cotées (AMMC). Vérifie que le mapping et les règles tiennent sur de **vraies** liasses, avec leurs postes manquants et leurs exercices atypiques |

Sortie attendue (le script rejoue au passage les quatre contrôles comptables
décrits dans [`08_donnees_reelles_ammc.md`](08_donnees_reelles_ammc.md)) :

```
Contribuables      : 12
Lignes             : 26
Equilibre verifie  : 25 / 26
Controle impossible: 1 (postes non lisibles dans les etats publies)
Lignes en ecart    : 0
Cellules vides     : 46 / 728 (6.3 %) — champs non lisibles, jamais estimes
```

`Lignes en ecart : 0` est la seule valeur à surveiller. Si elle est non nulle,
une valeur a été altérée : le script nomme la société et l'exercice concernés.

Ces sociétés ne portent **aucune étiquette de fraude** : ce fichier teste le
comportement du dispositif, il ne mesure aucune performance.

---

## 1. Tests automatisés

```bash
python -m pytest
```

**Attendu : `124 passed`** en une cinquantaine de secondes.

Ce qu'ils couvrent, par ordre d'importance :

**Les défauts silencieux** — ceux qui ne lèvent aucune erreur et produisent des
résultats plausibles mais faux. C'est là que se concentre l'effort :

- **l'ordre des colonnes** — les modèles ont été entraînés sur des tableaux
  NumPy et ne portent pas les noms de variables en interne ; une permutation
  donnerait des scores silencieusement erronés ;
- **l'ordre des lignes** — le calcul des variations trie par (contribuable,
  exercice) ; sans restauration de l'ordre d'entrée, les scores sont attribués
  aux mauvaises lignes. Ce bug s'est réellement produit : 16 418 lignes sur
  16 423 étaient concernées.

**L'exactitude comptable** — chacune des 28 correspondances est vérifiée
arithmétiquement sur une liasse aux montants ronds : `act` = actif circulant +
trésorerie-actif, `lt` = total actif − capitaux propres − assimilés, etc.

**La sémantique de l'absence** — un poste non renseigné doit rester `NaN` et
jamais devenir `0`.

**Les conversions d'unités** et l'invariance d'échelle des ratios.

### Cibler un sous-ensemble

```bash
python -m pytest tests/test_liasse_mapping.py       # la correspondance CGNC
python -m pytest tests/test_inference.py            # le pipeline de scoring
python -m pytest tests/test_regle_rupture.py        # la règle de rupture
python -m pytest -k "unites"                        # par mot-clé
python -m pytest -v                                 # détail test par test
```

### Si un test échoue

| Symptôme | Cause probable |
|---|---|
| `models/xgboost_v2_best.pkl absent` (tests sautés) | Normal si vous n'avez pas exécuté le notebook 05. Les tests concernés sont ignorés, pas en échec |
| `ordre des colonnes` | `model_meta_v2.json` désynchronisé du modèle — régénérez les deux ensemble |
| `un score a été attribué à la mauvaise ligne` | Régression sur la restauration de l'ordre dans `prepare_features` — **critique** |
| `LIASSE_MAPPING ne couvre pas exactement` | Une variable a été ajoutée ou retirée d'un côté sans l'autre |

---

## 2. Parcours manuel du dashboard

```bash
python -m streamlit run dashboard/app.py
```

L'application s'ouvre sur <http://localhost:8501>. Trois écrans :
**Ciblage** (l'outil de travail), **Analyse d'un dossier** (pourquoi celui-ci
remonte) et **Méthode et résultats** (le volet soutenance).

### Étape 1 — L'application démarre

**Attendu :** un en-tête sobre — surtitre en petites capitales, titre, chapô —
trois onglets, et une barre latérale décrivant la méthode, les ratios surveillés
et la portée juridique. Aucun dégradé, aucun emoji, aucun aplat de couleur vive :
si vous en voyez, la feuille de style de `dashboard/theme.py` n'a pas été
appliquée.

**Rien ne doit dépendre du modèle JAR2020.** Même si `models/xgboost_v2_best.pkl`
est absent, les trois écrans s'affichent : seule une ligne descriptive du modèle
est remplacée par une note expliquant comment le régénérer.

### Étape 2 — Écran Ciblage

Cochez « Utiliser le portefeuille d'exemple (220 contribuables fictifs) ».

**Attendu :**

| Indicateur | Valeur |
|---|---|
| Contribuables | 220 |
| Liasses | 880 |
| À vérifier (curseur à 5 %) | 44 |

Puis une liste de travail triée par score décroissant, avec un motif de
signalement lisible pour chaque ligne — « Charge d'impôt inférieure de 90 % à
l'attendu », et non un score nu.

**À vérifier aussi :**

- Basculez « Méthode d'analyse » entre les trois options (Règles + modèle IA /
  Règles seules / Modèle IA seul) → la liste et les motifs changent, sans erreur.
- Le sous-titre indique les R² du modèle réentraîné sur votre portefeuille :
  attendus **0,99 / 0,93 / 0,93** pour chiffre d'affaires, charge d'impôt et
  résultat net.
- Bouton « Télécharger la liste » → un CSV s'ouvre correctement dans Excel
  (encodage UTF-8 avec BOM, accents préservés).

❌ *« Fichier inexploitable »* → vérifiez que `identifiant_fiscal` et
`exercice` sont présents et sans valeur vide.
❌ *Avertissement « actif ≠ passif »* → l'extraction est incomplète, ou les
colonnes Brut et Net ont été confondues.
❌ *Note « le volet apprentissage n'a pas pu être entraîné »* → normal sous
40 contribuables (`EFFECTIF_MINIMUM`) ; l'analyse continue avec les règles
seules, et le motif du repli est affiché.

### Étape 3 — Écran Analyse d'un dossier

Sans portefeuille chargé, l'écran doit afficher une note renvoyant à l'écran
Ciblage — pas une trace Python. Après l'étape 2, sélectionnez le dossier n° 1.

**Attendu :**

- Le motif du signalement en titre, puis trois indicateurs : chiffre d'affaires,
  résultat net, impôt sur les résultats, dans l'unité de saisie.
- Un bloc « Ce que la liasse laissait attendre » avec des montants **Déclaré** et
  **Attendu** du **même ordre de grandeur**. Si l'un est mille fois plus grand
  que l'autre, c'est une régression du bug d'inversion d'échelle documenté dans
  [`07_modele_reconstitution.md`](07_modele_reconstitution.md) §2c — couvert par
  `tests/test_modele_marocain.py`.
- Un bloc « Ce qui a changé par rapport à l'exercice précédent » : les cinq
  ratios les plus rompus, avec ce qu'il faut vérifier pour chacun.
- Le sélecteur doit afficher la **raison sociale**, pas le matricule.

### Étape 4 — Écran Méthode et résultats

Ne dépend d'aucun fichier déposé : il lit `reports/`.

**Attendu :**

- Le tableau de transférabilité — modèle JAR2020 contre règle de rupture :
  corrélation avec l'anomalie **−0,026 / 0,300**, avec la taille
  **0,592 / 0,016**, AUC-ROC **0,46 / 0,946**.
- Le comparatif règles / IA par type d'anomalie, en tableau puis en graphique.
  Le point à voir d'un coup d'œil : la barre « Autre irrégularité » de
  « IA seule » s'effondre à **0,639** quand les deux autres restent au-dessus de
  0,89. C'est le comportement attendu, pas un défaut.
- Le classement des modèles JAR2020 par AUC-PR, avec la ligne de référence
  « hasard (0,0066) » et le meilleur modèle seul mis en accent.
- La figure de synthèse SHAP, si `reports/figures/04_shap_summary_full_model.png`
  existe ; sinon la section est simplement absente.

**Lisibilité des graphiques :** la légende doit être **sous** le tracé, jamais
posée par-dessus les barres ; les valeurs décimales doivent utiliser la virgule.

### Étape 5 — Robustesse de l'interface

| Test | Attendu |
|---|---|
| Déposer un fichier vide | Message d'erreur clair, pas de trace Python |
| Déposer un CSV sans `exercice` | « Colonnes identifiantes absentes » |
| Déposer un fichier avec un doublon (contribuable, exercice) | Anomalie bloquante signalée |
| Déposer un fichier à un seul exercice | Avertissement sur la couverture trajectoire |
| Ajouter une colonne inventée | Signalée comme « non reconnue et ignorée », sans blocage |
| Renommer/supprimer `models/xgboost_v2_best.pkl` | Les trois écrans fonctionnent ; seule la ligne descriptive du modèle est remplacée par une note |
| Ouvrir « Analyse d'un dossier » avant d'avoir chargé un portefeuille | Note explicative, pas d'erreur |

### Étape 6 — Le dispositif sur de vraies liasses

C'est l'étape qui distingue « l'application marche sur mes données » de
« l'application marche ». Dans l'écran **Ciblage**, déposez
`data/exemples_marocains/societes_cotees_reelles.csv`.

| Point à vérifier | Attendu |
|---|---|
| Contrôles de cohérence | **Aucune anomalie bloquante** — 26 lignes, 12 contribuables |
| Couverture après mapping | **80,4 %** (24 variables sur 28) — plus basse que sur le jeu simulé, ce qui est normal : de vraies liasses ont des postes non ventilés |
| Méthode appliquée | **Règles seules**, avec le motif affiché : 12 contribuables < 40, le volet IA est écarté automatiquement |
| Tête de classement | Lesieur Cristal 2024, puis SNEP 2023 et Risma 2025 |
| Motifs | Formulés en langage métier (« Rupture sur marge brute », « Rupture sur taux d'imposition effectif »), avec le pourcentage de variation |

**Ce que ce test prouve — et ce qu'il ne prouve pas.** Il prouve que la chaîne
complète encaisse des données réelles, hétérogènes, avec des trous. Il ne prouve
**rien** sur la détection : ces sociétés ne portent aucune étiquette de fraude,
et les dossiers en tête de classement sont des ruptures comptables réelles et
parfaitement légitimes — Risma remonte à cause d'une restructuration de
participations en 2024, pas d'une irrégularité. Voir
[`08_donnees_reelles_ammc.md`](08_donnees_reelles_ammc.md) §7.

---

## 3. Évaluation sur le portefeuille simulé

```bash
python scripts/evaluer_jeu_test_marocain.py
```

C'est le test le plus riche : il score les 880 lignes avec **deux dispositifs**
et les confronte à la clé de correction.

**Attendu (valeurs de référence) :**

```
=== Diagnostic — sur quoi chaque dispositif classe-t-il ? ===
      dispositif  corr_avec_anomalie  corr_avec_taille
  Modèle JAR2020              -0.026             0.592
Règle de rupture               0.300             0.016

=== Métriques globales ===
                     AUC-PR    AUC-ROC
Modèle JAR2020      0.0475    0.4604
Règle de rupture    0.5488    0.9459
Référence hasard    0.0386    0.5000
```

### Comment lire ce résultat

**Le modèle JAR2020 échoue, et c'est le résultat principal du travail.** Son
AUC-ROC de 0,46 est inférieure au hasard, et son score est corrélé à +0,59 avec
la **taille** du contribuable contre −0,03 avec l'anomalie. Il trie par ordre de
grandeur, pas par risque.

Ce n'est pas un bug de l'application : c'est un constat sur la transférabilité,
analysé dans [`06_transferabilite.md`](06_transferabilite.md).

**La règle de rupture réussit**, mais son succès est **partiellement
circulaire** : les anomalies ont été plantées sous forme de distorsions de
ratios, et la règle détecte des distorsions de ratios. Ce qu'elle établit
vraiment, c'est que la chaîne de traitement transmet correctement le signal.

**⚠️ Ces chiffres ne mesurent pas une capacité à détecter la fraude réelle.** Les
anomalies sont plantées par nous. Une validation supposerait des dossiers
authentiques et une vérité terrain issue de vérifications définitives —
inaccessibles (article 246 du CGI).

### Si les valeurs diffèrent des références ci-dessus

| Écart constaté | Interprétation |
|---|---|
| Légèrement différentes | Le jeu a été régénéré avec un `--contribuables` différent. La graine est fixée : à paramètres identiques, les valeurs sont reproductibles à l'identique |
| Règle de rupture qui s'effondre (AUC-ROC < 0,7) | Régression dans `src/regle_rupture.py` ou dans le mapping — lancez d'abord `pytest` |
| Modèle JAR2020 qui deviendrait excellent | Suspect. Vérifiez qu'aucune colonne de la clé de correction n'a fuité dans le portefeuille |

---

## 4. Les notebooks

```bash
python -m jupyter notebook notebooks/06_application_contexte_marocain.ipynb
python -m jupyter notebook notebooks/07_modele_marocain_reconstitution.ipynb
```

Ou en une commande, pour vérifier qu'ils s'exécutent de bout en bout :

```bash
python -m nbconvert --to notebook --execute --inplace \
    notebooks/06_application_contexte_marocain.ipynb
python -m nbconvert --to notebook --execute --inplace \
    notebooks/07_modele_marocain_reconstitution.ipynb
```

**Attendu :** aucune cellule en erreur.

- Le notebook 06 déroule la chaîne liasse → mapping → modèle JAR2020 et sert
  de démonstration reproductible de son échec sur données marocaines.
- Le notebook 07 entraîne le modèle de reconstitution sur le portefeuille
  simulé, démontre son invariance d'échelle, et le compare chiffre à chiffre
  à la règle de rupture. Il ne dépend pas de JAR2020 ni du modèle entraîné —
  seul le portefeuille simulé (§1) lui suffit.

---

## 5. Test rapide en ligne de commande

Pour vérifier la chaîne sans lancer l'interface :

```python
import pandas as pd
from src.liasse_mapping import lire_liasse, verifier_liasse, mapper_liasse
from src.inference import score_with_diagnostics
from src.regle_rupture import scorer_ruptures, expliquer_rupture

liasses = lire_liasse("data/exemples_marocains/echantillon_demo.csv")

print(verifier_liasse(liasses))                    # {'scorable': True, ...}

donnees, rapport = mapper_liasse(liasses, unite="dirhams", mad_par_usd=9.85)
print(rapport["taux_couverture_global_pct"])       # 85.7

classement = score_with_diagnostics(donnees)["scored"]
print(classement[["raison_sociale", "fyear", "risk_score"]].head())

ruptures = scorer_ruptures(donnees)
print(ruptures[["raison_sociale", "fyear", "score_rupture",
                "n_indicateurs_rompus"]].head())
print(expliquer_rupture(ruptures, 0))              # pourquoi ce dossier remonte
```

---

## 6. Récapitulatif — la liste à cocher

```
[ ] python -m pip install -r requirements.txt
[ ] python scripts/generer_jeu_test_marocain.py   → 880 lignes, aucun avertissement
[ ] python scripts/collecte_ammc.py               → 26 lignes réelles, 0 ligne en écart
[ ] python -m pytest                              → 124 passed
[ ] python scripts/evaluer_jeu_test_marocain.py   → AUC-ROC règle ≈ 0,95 / modèle JAR2020 ≈ 0,46
[ ] streamlit run dashboard/app.py                → 3 écrans, aucune erreur
[ ]   écran Ciblage                 → 220 / 880 / 44, liste de travail motivée
[ ]   écran Analyse d'un dossier    → déclaré et attendu du même ordre de grandeur
[ ]   écran Méthode et résultats    → AUC-ROC 0,46 vs 0,946 ; légendes sous les tracés
[ ]   liasses réelles déposées      → 80,4 % de couverture, repli motivé sur les règles
[ ]   fichiers malformés            → messages clairs, aucune trace Python
[ ] notebook 06                                   → s'exécute sans erreur
[ ] notebook 07                                   → s'exécute sans erreur, R² > 0,9 sur les 3 cibles
```

Si tout est coché, l'application est en état de démonstration.
