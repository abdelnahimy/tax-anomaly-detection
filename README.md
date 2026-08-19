# Détection d'anomalies fiscales par des techniques de l'Intelligence Artificielle

**Cas de la Direction Régionale des Impôts de Tétouan**

Dispositif d'aide à la programmation du contrôle fiscal. À partir des **liasses
fiscales** déposées par les contribuables, il produit un **classement de risque**
destiné à orienter une capacité de vérification limitée vers les dossiers les
plus susceptibles de donner lieu à redressement.

La question opérationnelle d'une direction régionale n'est jamais « qui
fraude ? » mais « **par lesquels commencer ?** ».

Ce document est le **point d'entrée unique** du projet : installation, usage,
architecture, données, tests, structure. La documentation méthodologique
détaillée — celle destinée à être reprise dans le mémoire — vit dans `docs/`,
dont le sommaire figure [plus bas](#documentation-méthodologique).

---

## Sommaire

- [Le résultat principal](#le-résultat-principal)
- [Statut du travail — à lire avant toute interprétation](#statut-du-travail--à-lire-avant-toute-interprétation)
- [Installation](#installation)
- [Le tableau de bord](#le-tableau-de-bord)
- [De la liasse fiscale au classement de risque](#de-la-liasse-fiscale-au-classement-de-risque)
- [Les deux jeux de données marocains](#les-deux-jeux-de-données-marocains)
- [Le jeu JAR2020](#le-jeu-jar2020)
- [Le modèle JAR2020 — pour mémoire](#le-modèle-jar2020--pour-mémoire)
- [Tests](#tests)
- [Structure du projet](#structure-du-projet)
- [Documentation méthodologique](#documentation-méthodologique)
- [Reproduire l'ensemble](#reproduire-lensemble)
- [Avertissement d'usage](#avertissement-dusage)

---

## Le résultat principal

> **Un modèle entraîné sur des données américaines ne se transfère pas au
> contexte fiscal marocain.** Appliqué à un portefeuille marocain, il classe par
> **taille d'entreprise** (corrélation +0,59) et non par anomalie (−0,03). Son
> AUC-ROC de 0,46 est inférieure au hasard.

Ce constat, obtenu expérimentalement et non supposé, est le cœur du travail. Il
délimite précisément ce qui est transposable — la méthode, la correspondance des
variables, le protocole de validation — et ce qui ne l'est pas : les
coefficients appris.

**La cause est structurelle.** JAR2020 ne contient que des sociétés américaines
cotées, dont le total de bilan médian avoisine 105 millions de dollars, quand
celui d'un portefeuille de direction régionale marocaine se situe autour de
1 million. Les contribuables marocains se massent dans la queue basse de la
distribution d'entraînement, et comme 48 des 54 variables du modèle sont des
variables de **niveau**, celui-ci trie par ordre de grandeur.

**La réponse apportée** combine deux lectures complémentaires, aucune ne
nécessitant d'étiquette de fraude :

- une **règle de rupture intra-contribuable** — ratios invariants d'échelle,
  chaque contribuable comparé **à lui-même** d'un exercice à l'autre ;
- un **modèle de reconstitution appris sur données marocaines** — XGBoost
  entraîné à prédire le chiffre d'affaires, la charge d'impôt et le résultat net
  *attendus* à partir du reste de la liasse, pour signaler les dossiers qui
  déclarent moins que ce que leur liasse laisse attendre (article 213 du CGI).

| Sur portefeuille simulé (880 liasses) | Modèle JAR2020 | Règle de rupture |
|---|---|---|
| Corrélation avec l'anomalie | −0,03 | **+0,30** |
| Corrélation avec la taille | +0,59 | **+0,02** |
| Anomalies captées dans le top 5 % | 11,8 % | **52,9 %** |
| AUC-ROC | 0,46 | **0,95** |
| Écart de signalement entre classes de taille | 73,0 pts | **10,4 pts** |

Analyse complète, avec l'honnêteté requise sur ce que chaque chiffre vaut :
[`docs/06_transferabilite.md`](docs/06_transferabilite.md).
Le modèle appris de remplacement et ses garde-fous :
[`docs/07_modele_reconstitution.md`](docs/07_modele_reconstitution.md).

---

## Statut du travail — à lire avant toute interprétation

Le modèle a été entraîné et validé sur **JAR2020** (entreprises américaines
cotées, fraudes sanctionnées par la SEC), **et non sur des dossiers de la DRI**.
Les données de l'administration fiscale marocaine sont couvertes par le **secret
professionnel (article 246 du CGI)** et aucun jeu marocain étiqueté n'est public.

Le projet manipule **deux jeux marocains, de nature différente**, qu'il ne faut
jamais confondre :

| Jeu | Nature | Ce qu'il permet | Ce qu'il ne permet pas |
|---|---|---|---|
| `portefeuille_dri_tetouan.csv` | **Entièrement fictif** — 220 contribuables, 4 exercices, anomalies plantées | Tester la chaîne de traitement à échelle réaliste, entraîner le modèle de reconstitution | Mesurer une performance de détection : les anomalies ayant été plantées, l'exercice est partiellement circulaire |
| `societes_cotees_reelles.csv` | **Réel** — 12 sociétés cotées, comptes sociaux CGNC publiés et audités, collectés auprès de l'AMMC | Vérifier que le mapping et les règles se comportent correctement sur de vraies liasses | Mesurer un taux de détection : ces sociétés ne portent **aucune étiquette de fraude** |

Aucun dossier de la DRI n'a été utilisé, ni ne pouvait l'être.

Cadrage complet : [`docs/01_cadrage_methodologique.md`](docs/01_cadrage_methodologique.md)
· Collecte des données réelles : [`docs/08_donnees_reelles_ammc.md`](docs/08_donnees_reelles_ammc.md)

---

## Installation

### Windows — le plus simple

Double-cliquez sur **`setup_venv.bat`**. Il crée un environnement virtuel isolé
`.venv\`, met pip à jour et installe les dépendances **paquet par paquet** —
ainsi un seul échec réseau n'annule pas toute la session, et le cache pip
conserve ce qui a déjà abouti. Relancer le script reprend là où il s'était
arrêté.

Puis double-cliquez sur **`run_dashboard.bat`**.

### Toutes plateformes

```bash
python -m pip install -r requirements.txt --timeout 120 --retries 10
python scripts/generer_jeu_test_marocain.py
python -m streamlit run dashboard/app.py
```

L'application s'ouvre sur <http://localhost:8501>. Pour la prendre en main,
cochez « Utiliser le portefeuille d'exemple (220 contribuables fictifs) ».

### Un seul fichier de dépendances

`requirements.txt` couvre tout : socle scientifique, modélisation, notebooks,
tableau de bord et tests.

**`torch` en est délibérément absent** — il est déclaré en commentaire, en bas
du fichier, avec sa commande d'installation. Il ne sert qu'à la section
autoencodeur du notebook 03 (qui se saute proprement en son absence), pèse
plusieurs centaines de mégaoctets, et son téléchargement était la cause la plus
fréquente des `ReadTimeoutError` sur connexion lente — une erreur qui faisait
échouer l'installation de **tous** les autres paquets.

Python **3.11** est recommandé : chaque paquet y dispose d'une roue précompilée
pour Windows 64 bits, donc aucune compilation locale.

### Dépannage

**`streamlit : le terme 'streamlit' n'est pas reconnu`** (PowerShell / cmd) — le
paquet est installé mais le dossier `Scripts\` de Python n'est pas dans le
`PATH`, cas classique quand pip installe en mode utilisateur. Contournez-le :

```powershell
python -m streamlit run dashboard/app.py
```

**`ModuleNotFoundError: No module named 'xgboost'` au démarrage** — `xgboost`
n'apparaît dans aucun `import` du tableau de bord, mais le modèle sauvegardé est
un `XGBClassifier` : joblib doit pouvoir reconstruire cette classe. Il figure
dans `requirements.txt`.

**L'installation échoue sur `torch`** — vous l'avez décommenté ou installé à la
main. Ce n'est pas bloquant : rien dans le projet n'en dépend hors du
notebook 03.

---

## Le tableau de bord

`dashboard/app.py` est l'unique point d'entrée. **Trois écrans, et pas un de
plus.**

| Écran | Ce qu'il produit |
|---|---|
| **Ciblage** | La liste de travail. Dépôt de la liasse, capacité de contrôle, choix de la méthode, classement motivé, export CSV |
| **Analyse d'un dossier** | Pourquoi celui-ci remonte, et de combien. Montants déclarés, table de reconstitution « déclaré contre attendu », ratios rompus, périmètre sectoriel |
| **Méthode et résultats** | Ce qui a été testé, ce qui marche, ce qui échoue. L'échec du transfert de JAR2020, la comparaison règles / modèle appris par type d'anomalie, les performances des modèles, la décomposition SHAP |

### 1. Ciblage — l'outil de travail

Trois étapes : dépôt de la liasse (ou chargement du portefeuille d'exemple),
déclaration de la capacité de contrôle, puis liste de travail classée avec le
motif de chaque signalement.

Trois méthodes au choix : « Règles + modèle IA » (défaut), « Règles seules »,
« Modèle IA seul ». En dessous de 40 contribuables (`EFFECTIF_MINIMUM`), le
volet apprentissage est écarté automatiquement et le motif du repli est affiché.

Chaque contribuable est analysé sur **trois axes**, tous établis à partir du
seul fichier déposé — aucune donnée externe n'est requise :

1. **à lui-même**, d'un exercice à l'autre (rupture de trajectoire) ;
2. **à ses pairs** du même secteur et du même exercice (écart sectoriel) ;
3. **à ce que sa propre liasse laisse attendre** (modèle de reconstitution).

Huit ratios sont surveillés : marge brute, taux d'imposition effectif, impôt sur
chiffre d'affaires, créances sur chiffre d'affaires, stocks sur chiffre
d'affaires, écritures d'ajustement du résultat, rotation de l'actif,
fournisseurs sur achats.

### 2. Analyse d'un dossier — pourquoi celui-ci remonte

Montants déclarés, table de reconstitution (« déclaré » contre « attendu », avec
l'écart en pourcentage et sa lecture métier), ratios ayant rompu par rapport à
l'exercice précédent, et périmètre de la comparaison sectorielle.

C'est l'écran qui rend la liste actionnable. Une liste sans motif ne l'est pas.

### 3. Méthode et résultats — pour la soutenance

Lit `reports/` : reste consultable même sans le modèle JAR2020 sérialisé ni le
jeu de 46 Mo.

### Rien n'est bloquant

Les deux premiers écrans ne dépendent d'aucun modèle pré-entraîné. Un fichier
absent est signalé là où il manque, et n'empêche jamais le démarrage.

| Fichier | Nécessaire pour | Si absent |
|---|---|---|
| `data/exemples_marocains/portefeuille_dri_tetouan.csv` | la case « portefeuille d'exemple » | message renvoyant à `scripts/generer_jeu_test_marocain.py` ; le dépôt de fichier reste possible |
| `reports/comparaison_regles_ia_marocain.csv` | le comparatif règles / IA de l'écran 3 | note renvoyant au notebook 07, le reste de l'écran s'affiche |
| `reports/comparison_v2_all_models.csv` | le tableau des modèles JAR2020 | note renvoyant aux notebooks 03-05 |
| `reports/figures/04_shap_summary_full_model.png` | la section SHAP de l'écran 3 | la section est simplement omise |
| `models/xgboost_v2_best.pkl` | la ligne descriptive du modèle retenu | une note indique comment le régénérer |

### Format d'entrée

CSV avec un enregistrement par couple **(contribuable, exercice)**. Deux
colonnes obligatoires : `identifiant_fiscal` et `exercice`. Les en-têtes
tolèrent majuscules, accents et espaces.

Les postes attendus sont ceux du **bilan** (tableaux n° 1 et 2) et du **CPC**
(tableau n° 3), au format CGNC. Un gabarit vide est téléchargeable depuis
l'écran Ciblage ; la correspondance poste à poste figure dans
[`docs/02_mapping_liasse_fiscale.md`](docs/02_mapping_liasse_fiscale.md).

**Fournissez au moins 3 exercices par contribuable** : c'est la comparaison
d'une année à l'autre qui porte l'essentiel du signal.

> **Deux pièges de saisie récurrents.**
> 1. `immobilisations_corporelles_brut` doit venir de la colonne **BRUT** du
>    bilan actif, pas de la colonne Net.
> 2. `chiffre_affaires` est la ligne « Chiffre d'affaires » du CPC, **pas** le
>    total des produits d'exploitation.

**Unités.** Les huit ratios du ciblage sont des rapports entre deux postes : ils
ne dépendent ni de l'unité de saisie, ni de la devise. Le classement est donc
identique quelle que soit la réponse au sélecteur d'unité — celui-ci ne sert
qu'à l'affichage des montants.

Trois fichiers permettent d'éprouver l'interface :
`dashboard/exemple_liasse_marocaine.csv` (8 entreprises fictives × 3 exercices),
`data/exemples_marocains/echantillon_demo.csv` (12 contribuables fictifs) et
`data/exemples_marocains/societes_cotees_reelles.csv` (**26 liasses réelles**,
avec leurs postes non ventilés et leurs exercices atypiques).

### Parti pris graphique

Registre **administratif neutre** : échelle de gris ardoise, une seule couleur
d'accent (`#2C5F9E`), aucun dégradé, aucun aplat vif, aucun emoji. La hiérarchie
vient de la typographie, du filet et du blanc.

Le couple de teintes des graphiques (`#2C5F9E` bleu acier, `#A05B1E` ocre
brûlée) n'a pas été choisi à l'œil : bande de clarté, plancher de chroma,
séparation en vision déficiente (ΔE 19,7 protan · 22,1 tritan) et contraste sur
fond clair ont été **validés**. Les couleurs de statut sont réservées et
s'accompagnent toujours d'un libellé texte — jamais de la couleur seule, pour
rester lisible en impression monochrome.

Les couleurs des widgets natifs de Streamlit sont fixées dans
`.streamlit/config.toml` ; tout le reste du style est injecté par
`dashboard/theme.py::injecter_css()`, qui fait foi.

### Architecture

Le tableau de bord ne contient **aucune logique de modélisation** : il appelle
`src/ciblage.py`, `src/regle_rupture.py`, `src/modele_marocain.py`,
`src/liasse_mapping.py` et `src/inference.py`. Les mêmes fonctions peuvent donc
servir une API ou un script batch, sans duplication.

> **Note technique.** Streamlit exécute le corps de **tous** les onglets à
> chaque interaction, y compris ceux qui ne sont pas visibles, et un bloc
> `with onglet:` ne crée pas de portée Python. Les variables de l'écran Ciblage
> sont donc suffixées `_cib`, celles de l'écran Analyse d'un dossier `_dos`,
> celles de l'écran Méthode et résultats `_met`, et chaque widget porte une
> `key=` explicite. Le passage de données entre écrans se fait par le
> dictionnaire `ETAT`, rempli par l'écran Ciblage et lu par les deux suivants —
> l'ordre du script garantit qu'il est rempli avant d'être lu.

> **Note sur les modèles sauvegardés.** Ils ont été entraînés sur des tableaux
> NumPy et ne portent pas les noms de colonnes en interne : **l'ordre des
> variables est la seule chose qui relie une valeur à la bonne feature**. Tout
> passe donc par `models/model_meta_v2.json`, qui fait foi. De plus, le calcul
> des variations trie les lignes par (contribuable, exercice) :
> `src/inference.py` restaure explicitement l'ordre d'entrée avant de renvoyer
> les scores — sans quoi les scores seraient attribués aux mauvaises lignes.

---

## De la liasse fiscale au classement de risque

```python
from src.liasse_mapping import lire_liasse, verifier_liasse, mapper_liasse
from src.ciblage import analyser_portefeuille
from src.regle_rupture import expliquer_rupture

# 1. Lecture — les en-têtes tolèrent majuscules, accents et espaces
liasses = lire_liasse("data/exemples_marocains/portefeuille_dri_tetouan.csv")

# 2. Contrôles de cohérence comptable AVANT scoring
diagnostic = verifier_liasse(liasses)          # équilibre du bilan, doublons, historique
if not diagnostic["scorable"]:
    raise SystemExit("\n".join(diagnostic["anomalies"]))

# 3. Correspondance CGNC → variables du modèle
donnees, rapport = mapper_liasse(liasses, unite="dirhams", convertir_en_usd=False)
print(rapport["taux_couverture_global_pct"])   # 85.7

# 4. Ciblage
classement, diag = analyser_portefeuille(donnees, methode="combine")
print(expliquer_rupture(classement, 0))        # pourquoi le 1er dossier remonte
```

**22 des 28 variables** du modèle sont reconstituables à partir de la seule
liasse fiscale. Les 4 restantes (`prcc_f`, `csho`, `dltis`, `sstk`) sont
laissées à `NaN` — jamais comblées par des zéros — et leur effet a été mesuré :
le classement résiste (Spearman 0,865), le niveau du score non (moyenne 0,264 →
0,471). D'où une règle d'usage claire : **raisonner en rang, jamais par seuil
absolu**.

Correspondance poste à poste :
[`docs/02_mapping_liasse_fiscale.md`](docs/02_mapping_liasse_fiscale.md)
· Démonstration exécutée :
[`notebooks/06_application_contexte_marocain.ipynb`](notebooks/06_application_contexte_marocain.ipynb)

---

## Les deux jeux de données marocains

### 1. Le portefeuille simulé — support de test et d'entraînement

```bash
python scripts/generer_jeu_test_marocain.py
```

Produit dans `data/exemples_marocains/` :

| Fichier | Contenu |
|---|---|
| `portefeuille_dri_tetouan.csv` | 220 contribuables × 4 exercices = **880 liasses**, 14 secteurs, 12 communes, 4 classes de taille. **Sans étiquette** — c'est le fichier à déposer dans l'application |
| `cle_anomalies.csv` | Clé de correction : qui porte une anomalie, laquelle, à partir de quel exercice |
| `echantillon_demo.csv` | 12 contribuables — prise en main rapide |

18 contribuables (3,9 % des lignes) portent une **anomalie plantée**, construite
à partir des schémas que recherche un vérificateur — et non à partir de la liste
des variables du modèle, ce qui limite la circularité de l'exercice :

`minoration_ca` · `majoration_charges` · `creances_fictives` ·
`stocks_surevalues` · `is_anormalement_bas` · `accruals_eleves` ·
`rupture_trajectoire`

Le bilan de chaque ligne est **équilibré par construction** : le fichier passe
les contrôles sans aucun avertissement, y compris pour les lignes anormales —
une anomalie fiscale déplace le contenu d'un bilan, elle ne le déséquilibre pas.

Le jeu est reproductible à l'identique (graine fixée).

```bash
python scripts/evaluer_jeu_test_marocain.py    # évaluation comparée des deux méthodes
```

### 2. Les sociétés cotées — le contrepoids réel

```bash
python scripts/collecte_ammc.py            # régénère le CSV + rapport de contrôle
python scripts/collecte_ammc.py --verifier # contrôles seuls, sans réécriture
```

`data/exemples_marocains/societes_cotees_reelles.csv` réunit les **comptes
sociaux (CGNC, entité seule)** de **12 sociétés cotées à la Bourse de
Casablanca**, lus dans les états de synthèse publiés au titre de l'information
périodique de l'AMMC : Med Paper, SNEP, Lesieur Cristal, Cosumar, Unimer,
Ciments du Maroc, Label Vie, CTM, Risma, Auto Nejma, Delta Holding, Jet
Contractors — **26 liasses réelles**, exercices 2020 à 2025.

Ce sont de **vraies liasses**, avec ce que la simulation ne produit pas : des
exercices atypiques (Risma 2024, −400,8 MDH de résultat non courant après
restructuration), des holdings pures dont le compte social ne ressemble en rien
à leur secteur affiché (Delta Holding), des postes que l'émetteur ne ventile
pas. Elles ne portent en revanche **aucune étiquette de fraude** — elles servent
à valider le comportement du dispositif, jamais à mesurer un taux de détection.

Chaque ligne est passée au contrôle : équation du bilan, concordance du résultat
net entre CPC et bilan passif, recomposition des capitaux propres, cascade du
CPC. **25 lignes sur 26 bouclent à 0,000 %** ; la 26ᵉ laisse un résidu de
0,003 % entièrement expliqué (un poste « écarts de conversion » absent du schéma
de la liasse retenu). Aucune valeur n'a été estimée : un poste illisible dans
les états publiés reste **vide**, jamais comblé.

Méthode de collecte, sources document par document, limites :
[`docs/08_donnees_reelles_ammc.md`](docs/08_donnees_reelles_ammc.md)

---

## Le jeu JAR2020

`data/raw/data_FraudDetection_JAR2020.csv` (~46 Mo) **n'est pas versionné**, en
raison de sa taille.

```bash
# depuis la racine du projet
curl -L -o data/raw/data_FraudDetection_JAR2020.csv \
  https://raw.githubusercontent.com/JarFraud/FraudDetection/master/data_FraudDetection_JAR2020.csv
```

Ou manuellement depuis <https://github.com/JarFraud/FraudDetection>, puis
déposer le fichier dans `data/raw/`.

| | |
|---|---|
| Observations | 146 045 couples (entreprise, exercice) |
| Période | 1991-2014 |
| Variables | 28 postes comptables bruts (nomenclature Compustat) + identifiants |
| Étiquette | `misstate` — 964 cas positifs, soit **0,66 %** |
| Source des étiquettes | *Accounting and Auditing Enforcement Releases* (AAER) de la SEC |

Colonnes identifiantes : `gvkey` (entreprise), `fyear` (exercice fiscal),
`p_aaer` (numéro d'affaire de fraude — sert à l'anti-fuite entre entraînement et
test), `misstate` (cible).

### Ce qui fonctionne sans ce fichier

| Élément | Sans `data_FraudDetection_JAR2020.csv` |
|---|---|
| Tableau de bord — les trois écrans | fonctionne |
| `scripts/generer_jeu_test_marocain.py` | fonctionne |
| `scripts/evaluer_jeu_test_marocain.py` | fonctionne |
| `scripts/collecte_ammc.py` | fonctionne |
| `scripts/robustesse_variables_absentes.py` | s'arrête avec un message |
| Notebooks 01 à 05 | nécessitent le fichier |
| Notebooks 06 et 07 | fonctionnent |
| Suite de tests | fonctionne |

Le modèle entraîné (`models/xgboost_v2_best.pkl`) étant fourni, **l'essentiel de
l'outil est utilisable immédiatement**.

`data/processed/data_transformed.csv` (~114 Mo) est produit par le notebook 02 à
partir du fichier ci-dessus. Il est régénérable à tout moment, n'est pas
versionné, et n'a pas à être conservé ni transmis — mais les notebooks 03, 04 et
05 le lisent.

---

## Le modèle JAR2020 — pour mémoire

Conservé parce qu'il constitue la validation méthodologique du dispositif, et
parce que son échec sur données marocaines est un résultat en soi.

- **XGBoost**, hyperparamètres optimisés par Optuna, `scale_pos_weight ≈ 121`
- **54 variables** : 28 postes comptables en log signé + 20 variations sur 1 et
  2 exercices + 6 ratios fiscaux métier
- **Validation temporelle stricte** : entraînement sur les exercices ≤ *t*−2,
  test sur l'exercice *t*, avec retrait des affaires de fraude pluriannuelles
- **AUC-PR 0,0168** contre 0,0066 pour le hasard sur JAR2020 — soit environ 2,5×
- Sur données marocaines simulées : **AUC-ROC 0,46**, inférieure au hasard

> **Le score n'est pas une probabilité.** Le `scale_pos_weight` gonfle
> mécaniquement les probabilités produites : un score de 0,70 ne signifie pas
> « 70 % de chances de fraude ». Seul le classement fait sens — et seulement
> dans le contexte pour lequel le modèle a été entraîné.

---

## Tests

```bash
python -m pip install -r requirements.txt
python -m pytest
```

**124 tests.** Ils visent en priorité les défauts qui ne lèvent aucune erreur et
produisent des résultats plausibles mais faux :

- **l'ordre des colonnes** — les modèles ont été entraînés sur des tableaux
  NumPy et ne portent pas les noms de variables en interne : une permutation
  donnerait des scores silencieusement erronés ;
- **l'ordre des lignes** — le calcul des variations trie par (contribuable,
  exercice) ; sans restauration de l'ordre d'entrée, les scores sont attribués
  aux mauvaises lignes (bug réel, corrigé, désormais sous test) ;
- **l'exactitude arithmétique** de chacune des 28 correspondances comptables ;
- **la sémantique de l'absence** — un poste non renseigné doit rester `NaN` et
  jamais devenir `0` ;
- **l'invariance d'échelle** de la règle de rupture ET du modèle de
  reconstitution — la propriété qui les protège du biais de taille ;
- **la réciprocité exacte** de la transformation log signée normalisée
  (`_log_signe` / `_inverse_log_signe`) — sans elle, le montant « attendu »
  affiché au vérificateur perd tout rapport avec le montant déclaré (bug réel,
  corrigé, désormais sous test) ;
- **le repli automatique** de l'orchestrateur (`src/ciblage.py`) sur les règles
  seules quand le portefeuille est trop petit pour entraîner un modèle.

Les tests marqués `slow` chargent le modèle entraîné et sont automatiquement
sautés s'il est absent.

Parcours de recette complet, valeurs attendues et que faire quand elles ne
correspondent pas : [`docs/05_guide_de_test.md`](docs/05_guide_de_test.md).

---

## Structure du projet

```
README.md                          Ce document — point d'entrée unique
requirements.txt                   Toutes les dépendances (torch en option, commenté)
setup_venv.bat                     Installation Windows
run_dashboard.bat                  Lancement Windows du tableau de bord
nettoyer.bat                       Purge des caches et des fichiers obsolètes
pytest.ini                         Configuration de la suite de tests

docs/                              Documentation méthodologique — à reprendre dans le mémoire
  01_cadrage_methodologique.md       Statut du travail, choix du dataset, limites
  02_mapping_liasse_fiscale.md       Correspondance CGNC → variables du modèle
  03_deploiement_DRI_Tetouan.md      Insertion opérationnelle, cadre juridique
  04_bibliographie.md                Sources académiques, juridiques et techniques
  05_guide_de_test.md                Comment tester l'application
  06_transferabilite.md              Le résultat principal
  07_modele_reconstitution.md        Le modèle appris sur données marocaines
  08_donnees_reelles_ammc.md         Collecte des liasses réelles (sociétés cotées)

src/                               Code réutilisable — aucune dépendance à Streamlit
  liasse_mapping.py                  Liasse fiscale marocaine → variables du modèle
  regle_rupture.py                   Ciblage transparent, sans apprentissage
  modele_marocain.py                 Modèle de reconstitution, appris sur données marocaines
  ciblage.py                         Orchestration règles + modèle, avec repli automatique
  inference.py                       Données brutes → score de risque + SHAP
  config.py                          Chemins, listes de variables, protocole
  preprocessing.py                   Log signé, indicateurs de valeurs manquantes
  feature_engineering.py             Split temporel, trajectoire, ratios fiscaux
  models/                            Entraînement supervisé et non supervisé
  scoring.py                         Métriques adaptées au déséquilibre
  viz.py                             Fonctions de tracé partagées

scripts/                           Points d'entrée exécutables — un rôle chacun
  generer_jeu_test_marocain.py       Portefeuille simulé + clé de correction
  evaluer_jeu_test_marocain.py       Évaluation comparée des deux méthodes
  collecte_ammc.py                   Liasses réelles des sociétés cotées + contrôle qualité
  generer_exemple_liasse.py          Petit jeu d'exemple (8 entreprises)
  generer_tableau_mapping.py         Régénère le tableau de correspondance depuis le code
  robustesse_variables_absentes.py   Effet des 4 variables hors liasse

notebooks/                         Démonstrations exécutables, dans l'ordre de lecture
  01_EDA.ipynb  →  05_modeling_supervised_v2.ipynb    Construction du modèle sur JAR2020
  06_application_contexte_marocain.ipynb              Application au contexte marocain (et son échec)
  07_modele_marocain_reconstitution.ipynb             Modèle appris sur données marocaines

dashboard/                         Interface Streamlit — un seul point d'entrée
  app.py                             Ciblage · Analyse d'un dossier · Méthode et résultats
  theme.py                           Charte graphique (palette validée, en-tête, notes)
  exemple_*.csv                      Gabarits de saisie

tests/                             Suite pytest — 124 tests
.streamlit/config.toml             Couleurs des widgets natifs de Streamlit

data/exemples_marocains/           Les deux jeux marocains
  portefeuille_dri_tetouan.csv       Simulé — 220 contribuables × 4 exercices
  cle_anomalies.csv                  Clé de correction du jeu simulé
  echantillon_demo.csv               Simulé — 12 contribuables, prise en main
  societes_cotees_reelles.csv        Réel — 12 sociétés cotées, 26 liasses CGNC
data/raw/                          data_FraudDetection_JAR2020.csv (non versionné)
data/processed/                    Sorties intermédiaires des notebooks (non versionnées)
models/                            Modèles entraînés + métadonnées (ordre des variables)
reports/                           Tableaux comparatifs, mapping exporté, figures
```

**Règle de rangement.** `src/` ne contient que du code appelable sans effet de
bord ; tout ce qui s'exécute et écrit un fichier vit dans `scripts/` ; tout ce
qui raconte vit dans `docs/` ou dans `notebooks/`. Aucun script n'est dupliqué
entre les deux, et aucun fichier n'est conservé « au cas où » : ce dépôt ne
contient que ce dont le mémoire ou l'application ont besoin.

Pour purger ce qui se régénère seul — caches Python, caches pytest, dossiers
`.ipynb_checkpoints` — double-cliquez sur **`nettoyer.bat`**. Il ne touche ni au
code, ni aux notebooks, ni aux modèles, ni aux données, et récapitule à la fin
les gros fichiers qu'il a délibérément laissés en place.

---

## Documentation méthodologique

Les documents de `docs/` sont rédigés pour être **repris directement dans le
mémoire**.

| Document | Question à laquelle il répond |
|---|---|
| [`01_cadrage_methodologique.md`](docs/01_cadrage_methodologique.md) | *Sur quelles données ce travail a-t-il été fait, et en quoi concerne-t-il la DRI de Tétouan ?* |
| [`02_mapping_liasse_fiscale.md`](docs/02_mapping_liasse_fiscale.md) | *Les données dont dispose réellement l'administration peuvent-elles alimenter ce modèle ?* |
| [`03_deploiement_DRI_Tetouan.md`](docs/03_deploiement_DRI_Tetouan.md) | *Où le dispositif se brancherait-il dans le travail de la DRI, et sous quelles garanties ?* |
| [`04_bibliographie.md`](docs/04_bibliographie.md) | Sources académiques, juridiques et techniques |
| [`05_guide_de_test.md`](docs/05_guide_de_test.md) | *Comment vérifier que l'application fonctionne ?* |
| [`06_transferabilite.md`](docs/06_transferabilite.md) | *Le modèle se transfère-t-il au contexte marocain ?* — **le résultat principal** |
| [`07_modele_reconstitution.md`](docs/07_modele_reconstitution.md) | *Peut-on entraîner un modèle sur données marocaines sans étiquette de fraude ?* |
| [`08_donnees_reelles_ammc.md`](docs/08_donnees_reelles_ammc.md) | *Le dispositif tient-il sur de vraies liasses marocaines, et non sur une simulation ?* |

**Si vous rédigez la partie méthodologique** → document 01. Il contient le
cadrage à reprendre tel quel, la formulation de titre recommandée, et les huit
limites à assumer explicitement.

**Si vous préparez la soutenance** → document 03, section 8 (« Réponses aux
objections prévisibles »), puis document 01, section 5 (limites).

**Si vous travaillez sur les données** → document 02, puis document 08.

### Les quatre arguments qui rattachent le projet au sujet

1. **Le cadrage.** Les données de la DRI sont inaccessibles (art. 246 CGI) et
   aucun jeu marocain étiqueté n'est public. JAR2020 est donc un jeu de
   **validation méthodologique**, non un substitut prétendant aux mêmes
   résultats. Le dire soi-même vaut mieux que de se le voir opposer.

2. **La transposition.** 22 des 28 variables du modèle sont reconstituables à
   partir de la seule liasse fiscale marocaine. La correspondance est
   implémentée (`src/liasse_mapping.py`), testée (`tests/`) et démontrée
   (notebook 06, écran Ciblage du tableau de bord).

3. **La robustesse mesurée.** Les 4 variables absentes de la liasse ont été
   testées empiriquement : le classement résiste (Spearman 0,865), le niveau du
   score non (moyenne 0,264 → 0,471). D'où une règle d'usage claire — raisonner
   en rang, jamais par seuil absolu.

4. **L'épreuve du réel.** La correspondance a été appliquée à **26 liasses
   véritables** — les comptes sociaux CGNC de 12 sociétés cotées, collectés
   auprès de l'AMMC. Elles passent les contrôles sans anomalie, avec 80,4 % de
   couverture des variables. Ce n'est pas une mesure de performance — ces
   sociétés ne portent aucune étiquette de fraude — mais c'est la preuve que le
   dispositif ne tient pas seulement sur une simulation.

### Régénération de la documentation

Le tableau de correspondance du document 02 est **produit depuis le code**, il
n'est jamais recopié à la main : code et documentation ne peuvent donc pas
diverger.

```bash
python scripts/generer_tableau_mapping.py        # met à jour le document
python scripts/generer_tableau_mapping.py --csv  # exporte aussi reports/mapping_liasse_fiscale.csv
```

---

## Reproduire l'ensemble

1. Télécharger `data_FraudDetection_JAR2020.csv` — voir
   [Le jeu JAR2020](#le-jeu-jar2020).
2. `setup_venv.bat` ou `pip install -r requirements.txt`.
3. Notebooks 01 à 05 dans l'ordre → produisent `models/xgboost_v2_best.pkl`.
4. `python scripts/robustesse_variables_absentes.py`
5. `python scripts/generer_jeu_test_marocain.py`
6. `python scripts/evaluer_jeu_test_marocain.py`
7. `python scripts/collecte_ammc.py` — liasses réelles des sociétés cotées et
   rapport de contrôle qualité.
8. Notebook 06 — application au contexte marocain (et son échec).
9. Notebook 07 — entraînement du modèle de reconstitution sur données
   marocaines, sans étiquette de fraude.

Le modèle entraîné étant fourni, **le volet marocain fonctionne sans télécharger
JAR2020** : seuls les notebooks 01-05 et le script de robustesse en dépendent.
Le notebook 07 ne dépend que du portefeuille simulé (étape 5) — il n'a besoin ni
de JAR2020, ni du modèle JAR2020 entraîné.

---

## Avertissement d'usage

Cet outil est un **support de programmation du contrôle fiscal**. Un score élevé
signale un dossier à examiner — jamais une fraude établie. Le score n'a **aucune
valeur probante** : il n'est ni un commencement de preuve, ni un motif opposable
au contribuable, et ne peut fonder aucun redressement (articles 220 et suivants
du CGI). La décision de vérifier relève du chef de brigade, et d'elle seule.

**Ne jamais déposer d'export de liasses réelles dans ce dépôt.** Toute donnée
issue du système d'information de l'administration relève du secret
professionnel (article 246 du CGI). Le `.gitignore` comporte des garde-fous en
ce sens, mais ils ne remplacent pas la vigilance.

---

## Référence

Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020). *Detecting Accounting
Fraud in Publicly Traded U.S. Firms Using a Machine Learning Approach.*
Journal of Accounting Research, 58(1), 199-235.

Bibliographie complète : [`docs/04_bibliographie.md`](docs/04_bibliographie.md)
