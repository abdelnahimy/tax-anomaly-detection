# Un modèle appris sur données marocaines, sans étiquette de fraude

**Question :** puisque le modèle JAR2020 ne transfère pas
([`06_transferabilite.md`](06_transferabilite.md)), faut-il renoncer à
l'apprentissage automatique au Maroc, ou existe-t-il une cible exploitable
malgré l'absence de dossiers étiquetés « fraude » ?

**Réponse : oui.** Le montant déclaré lui-même est une cible disponible sur
chaque liasse. On n'apprend pas à prédire la fraude — personne, hors
administration, n'a accès à cette étiquette (article 246 du CGI) — on apprend à
prédire **ce qu'un contribuable aurait normalement déclaré**, compte tenu du
reste de sa liasse, puis on mesure l'écart avec ce qu'il a effectivement
déclaré. C'est la transposition, en apprentissage supervisé, de la
**reconstitution du chiffre d'affaires** que pratique un vérificateur lorsqu'il
écarte une comptabilité (article 213 du CGI).

Implémentation : [`src/modele_marocain.py`](../src/modele_marocain.py).

---

## 1. Le principe

Trois grandeurs sont reconstituées à partir du reste de la liasse — appareil
productif, stocks, achats, créances, secteur — jamais à partir d'une autre
grandeur déclarative qui rendrait la prédiction triviale :

| Cible | Prédite à partir de… | Volontairement exclu | Ce qu'un écart négatif suggère |
|---|---|---|---|
| Chiffre d'affaires | Immobilisations, stocks, achats, créances, dette | Résultat, impôt | Ventes non déclarées |
| Charge d'impôt | Appareil productif **et** chiffre d'affaires | Résultat net, résultat courant | Charges non justifiées, provisions non déductibles, reports déficitaires contestables |
| Résultat net | Appareil productif, chiffre d'affaires, dette | Résultat courant, impôt | Minoration de la base imposable, quel qu'en soit le canal |

L'exclusion délibérée du résultat et de l'impôt l'un de l'autre n'est pas un
détail : prédire la charge d'impôt à partir du résultat courant donnerait un
modèle excellent et inutile — l'impôt est mécaniquement proportionnel au
résultat, et un contribuable qui minore son résultat verrait sa charge d'impôt
jugée « normale ». Le modèle doit fonder sa prédiction sur les **moyens mis en
œuvre**, pas sur une autre ligne déclarative.

Seuls les écarts **négatifs** comptent dans le score : déclarer plus que la
norme n'est pas un risque fiscal.

---

## 2. Deux garde-fous techniques, sans lesquels le résultat serait faux

### a) Prédiction strictement hors échantillon

Si le modèle prédisait le chiffre d'affaires d'une entreprise qu'il a vue à
l'entraînement, il reproduirait sa déclaration — y compris minorée — et
l'écart s'annulerait. Chaque prédiction est donc produite par un modèle qui
n'a **jamais vu ce dossier**, via validation croisée.

### b) Regroupement par contribuable (`GroupKFold`)

Les exercices d'un même contribuable se ressemblent fortement. Séparer 2023 et
2024 d'une même entreprise entre entraînement et test laisserait fuir
l'information — le modèle « prédirait » 2024 en ayant appris 2023 du même
contribuable. Le découpage se fait donc par identifiant fiscal
(`GroupKFold(groups=gvkey)`), jamais au hasard des lignes.

### c) Invariance d'échelle — le même défaut que JAR2020, corrigé avant qu'il ne se reproduise

Le log signé naïf, `sign(x)·log(1+|x|)`, n'est **pas invariant d'échelle** :
autour de 1 il se comporte comme l'identité, autour de 10⁶ comme un
logarithme. Selon que les montants sont saisis en dirhams ou en millions, la
compression n'est donc pas la même — et le modèle réapprend la taille de
l'entreprise au lieu de la neutraliser.

C'est très exactement le mécanisme qui disqualifie le modèle JAR2020 sur
données marocaines (48 de ses 54 variables sont des variables de niveau — voir
[`06_transferabilite.md`](06_transferabilite.md) §3). Un premier essai du
modèle de reconstitution a reproduit ce défaut : corrélation de +0,43 entre le
score et le total de bilan, et des résultats qui changeaient selon que le
fichier était saisi en dirhams ou en millions de dirhams — la preuve la plus
directe qu'une échelle non neutralisée fuyait dans le score.

**Correction :** chaque colonne est d'abord divisée par son propre ordre de
grandeur médian (`_echelle_robuste`) avant application du log signé
(`_log_signe`). Le résidu du modèle devient alors `log(déclaré / attendu)`,
un **écart relatif**, indépendant de l'unité de saisie comme de la taille du
contribuable.

| | Avant correction | Après correction |
|---|---|---|
| Corrélation du score avec le total de bilan | +0,43 | **+0,05** |
| Résultat identique dirhams / millions de dirhams | non | **oui** |

Restituer un montant « attendu » lisible par le vérificateur impose ensuite
d'inverser cette normalisation (`_inverse_log_signe`) : le résidu (donc le
score et l'écart en %) est indifférent à l'échelle, mais l'**affichage** du
montant reconstitué ne l'est pas. Un défaut d'inversion produirait un montant
attendu sans rapport avec le montant déclaré — c'est un bug qui s'est
effectivement produit pendant le développement (l'« attendu » affichait des
montants plusieurs ordres de grandeur au-dessus du déclaré) avant d'être
corrigé et couvert par un test de non-régression
(`tests/test_modele_marocain.py::TestReconstitutionSurPortefeuilleReel::test_attendu_du_meme_ordre_de_grandeur_que_declare`).

---

## 3. Résultat sur le portefeuille simulé

Entraînement `XGBRegressor` (300 arbres, profondeur 4), validation croisée à 5
plis groupés par contribuable, sur les 220 contribuables × 4 exercices.

| Cible | R² hors échantillon | Lignes prédites |
|---|---|---|
| Chiffre d'affaires | 0,99 | 880 |
| Charge d'impôt | 0,93 | 880 |
| Résultat net | 0,93 | 880 |

Ces R² élevés portent sur la prédiction du **niveau déclaratif normal**, pas
sur la détection d'anomalie elle-même — un contribuable parfaitement sincère
doit être prédit avec précision, sinon le moindre écart deviendrait un faux
signal.

### Le modèle apporte-t-il quelque chose que la règle n'a pas ?

La comparaison directe favorise la règle
([`06_transferabilite.md`](06_transferabilite.md) documente déjà pourquoi une
comparaison globale sur ce portefeuille lui est structurellement favorable :
les anomalies y ont été plantées comme des distorsions de ratios). Le tableau
ci-dessous le confirme :

| | AUC-ROC globale | top 5 % | top 10 % | top 20 % |
|---|---|---|---|---|
| Règles seules | **0,982** | **79,4 %** | **88,2 %** | **100,0 %** |
| IA seule (reconstitution) | 0,828 | 38,2 % | 61,8 % | 73,5 % |
| Combiné (60 % règles / 40 % IA) | 0,957 | 73,5 % | 85,3 % | 85,3 % |

Mais en isolant les anomalies **selon qu'elles réduisent ou non la base
imposable déclarée**, l'écart se referme, et s'inverse même sur le sous-type
que le modèle est censé détecter :

| Type d'anomalie planté | AUC-ROC — Règles | AUC-ROC — IA (reconstitution) |
|---|---|---|
| **Minoration déclarative** (CA minoré, charges majorées, impôt anormalement bas) | 0,993 | **0,946** |
| **Autre irrégularité** (créances fictives, stocks surévalués, accruals, rupture) | 0,965 | 0,639 |

Le modèle de reconstitution reste, à dessein, **peu sensible** à ce qui
n'affecte pas directement la base imposable déclarée (stocks surévalués,
créances gonflées) : ce n'est pas ce qu'il a été construit pour reconstituer.
Sur les minorations déclaratives en revanche, il approche la règle — et il
apporte quelque chose qu'aucune règle ne donne : **un montant chiffré**.
« Vous avez déclaré 3 122 907 MAD, la liasse en laisse attendre 5 520 713 » est
directement exploitable dans une procédure de reconstitution (article 213 du
CGI) ; « votre marge a varié de 43 % » ne l'est pas de la même façon.

C'est pourquoi le dashboard combine les deux lectures par défaut plutôt que de
choisir l'une contre l'autre (`src/ciblage.py`, `methode="combine"`,
`poids_regles=0,6`) — la pondération privilégie légèrement les règles, qui
couvrent un éventail plus large d'irrégularités, tout en conservant l'apport
du modèle sur les minorations.

---

## 4. Ce que ce résultat établit, et ce qu'il n'établit pas

| Conclusion | Statut | Pourquoi |
|---|---|---|
| Un modèle appris sur données marocaines, sans étiquette de fraude, est possible et cohérent (R² > 0,9, invariant d'échelle) | 🟢 **Solide** | Repose sur la structure de la liasse (équilibre bilan/CPC), vérifiable indépendamment de toute étiquette |
| Sur les minorations déclaratives, il approche la règle (AUC-ROC 0,946 contre 0,993) | 🟠 **Partiellement circulaire** | Les anomalies évaluées ont été **plantées**, y compris pour ce sous-type. La performance sur des minorations réelles, plus subtiles, n'est pas mesurée ici |
| Il reste peu sensible aux irrégularités qui ne réduisent pas la base imposable déclarée | 🟢 **Attendu et documenté** | Conséquence directe du choix des cibles, pas une limite cachée |

Ce que ni ce document ni le code n'établissent : qu'un écart de reconstitution
signale une fraude. Un écart important signale une **incohérence
déclarative** — sinistre, changement d'activité, sous-traitance massive,
démarrage, restructuration peuvent légitimement l'expliquer. Le dispositif
produit des dossiers **à examiner**, jamais des conclusions
(article 220 et suivants du CGI : un score n'a pas de valeur probante).

---

## 5. Où le trouver dans l'application

- **`dashboard/app.py`** — étape 2, sélecteur de méthode (« Règles + modèle
  IA », « Règles seules », « Modèle IA seul ») ; étape 4, bloc
  « 🤖 Ce que la liasse laissait attendre » pour chaque dossier.
- **`src/modele_marocain.py`** — `entrainer_reconstitution`,
  `scorer_reconstitution`, `expliquer_reconstitution`.
- **`src/ciblage.py`** — orchestration règles + IA, avec repli automatique et
  motivé sur les règles seules si le portefeuille est trop petit
  (`EFFECTIF_MINIMUM = 40` contribuables — en dessous, un modèle appris
  reproduirait du bruit plutôt qu'une norme).

## 6. Reproduire

```bash
python -m pytest tests/test_modele_marocain.py tests/test_ciblage.py -v
jupyter notebook notebooks/07_modele_marocain_reconstitution.ipynb
```

Le notebook déroule l'ensemble de ce document de façon exécutable : la
démonstration du biais d'échelle (§2 ci-dessus), l'entraînement des trois
cibles avec leurs R², l'explication d'un dossier signalé, et la comparaison
chiffrée avec la règle de rupture (§3). Il régénère
`reports/figures/07_modele_marocain_importances.png`,
`reports/figures/07_modele_marocain_comparaison.png` et
`reports/comparaison_regles_ia_marocain.csv`.

Le jeu de données est celui de [`06_transferabilite.md`](06_transferabilite.md)
§1 — mêmes 220 contribuables simulés, même clé de correction, régénérable à
l'identique.

> **Rappel.** Toutes les données de ce portefeuille sont **fictives**. Les
> données réelles de l'administration sont couvertes par le secret
> professionnel (article 246 du CGI).
