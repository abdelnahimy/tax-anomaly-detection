# Des liasses marocaines réelles, pour sortir de la simulation

**Problème.** Tout ce que le dispositif a montré jusqu'ici l'a été sur un
portefeuille **simulé** ([`01_cadrage_methodologique.md`](01_cadrage_methodologique.md) §2).
Ses anomalies ont été plantées par nous ; ses bilans sont équilibrés par
construction ; ses secteurs sont propres. C'est un excellent banc de recette, et
un mauvais témoin de la réalité comptable marocaine — on y mesure surtout la
qualité de sa propre simulation.

**Ce qu'il fallait.** De vraies liasses marocaines, publiques, opposables,
vérifiables par un tiers. Elles existent : les sociétés cotées à la Bourse de
Casablanca publient chaque année leurs **états de synthèse au format CGNC** au
titre de l'information périodique imposée par l'**AMMC** (Autorité Marocaine du
Marché des Capitaux — loi 44-12, circulaire AMMC 03/19).

Implémentation et données figées : [`scripts/collecte_ammc.py`](../scripts/collecte_ammc.py)
→ [`data/exemples_marocains/societes_cotees_reelles.csv`](../data/exemples_marocains/societes_cotees_reelles.csv).

---

## 1. Le jeu constitué

**12 sociétés, 26 liasses, exercices 2020 à 2025.**

| Identifiant | Société | Secteur retenu | Exercices |
|---|---|---|---|
| 1001 | Med Paper | Industrie papetière | 2020, 2021, 2023, 2024 |
| 1002 | SNEP | Industrie plastique | 2022, 2023 |
| 1003 | Lesieur Cristal | Agroalimentaire | 2023, 2024 |
| 1004 | Cosumar | Agroalimentaire | 2024, 2025 |
| 1005 | Unimer | Agroalimentaire | 2024, 2025 |
| 1006 | Ciments du Maroc | Matériaux de construction | 2023, 2024 |
| 1007 | Label Vie | Commerce de détail | 2023, 2024 |
| 1008 | CTM | Transport et logistique | 2024, 2025 |
| 1009 | Risma | Hôtellerie et tourisme | 2024, 2025 |
| 1010 | Auto Nejma | Distribution automobile | 2024, 2025 |
| 1011 | Delta Holding | Holding et services | 2024, 2025 |
| 1012 | Jet Contractors | BTP et construction | 2024, 2025 |

L'`identifiant_fiscal` est un **numéro d'ordre conventionnel** (1001, 1002, …),
pas l'identifiant fiscal réel des sociétés — celui-ci n'est pas public et
n'aurait aucune utilité ici. Il joue le rôle de clé de regroupement, comme le
`gvkey` de JAR2020.

L'amplitude est ce qui fait l'intérêt du jeu : de Delta Holding (bilan social
1,96 Md DH, chiffre d'affaires 84 MDH) à Label Vie (bilan 11,53 Md DH, chiffre
d'affaires 8,14 Md DH), soit un facteur **97 sur le chiffre d'affaires**. Un
dispositif qui trierait par taille se trahirait immédiatement sur ce jeu — c'est
exactement ce à quoi il sert.

---

## 2. Comptes sociaux, jamais consolidés — la décision qui structure tout

Un émetteur coté publie **deux jeux de comptes** dans le même document : ses
comptes **sociaux** (entité seule, référentiel CGNC) et ses comptes
**consolidés** (groupe, référentiel IFRS). Les seconds sont plus visibles — ce
sont eux que reprend la presse financière — et ce sont les mauvais.

La liasse fiscale marocaine est un document **d'entité**. La correspondance de
[`02_mapping_liasse_fiscale.md`](02_mapping_liasse_fiscale.md) n'a de sens qu'au
niveau social : c'est l'entité qui dépose, c'est elle qui est vérifiée, c'est
son résultat qui supporte l'IS. Prendre les comptes consolidés reviendrait à
comparer un contribuable à un périmètre qui n'existe pas fiscalement.

L'écart n'est pas cosmétique :

| Société | Exercice | CA / résultat **social** | CA / résultat **consolidé** |
|---|---|---|---|
| Delta Holding | 2025 | CA 84,1 MDH | CA 3 050 MDH (×36) |
| Label Vie | 2024 | CA 8 140 MDH | CA 16 400 MDH (×2) |
| Ciments du Maroc | 2024 | CA 3 903 MDH | CA ≈ 4 360 MDH |
| Jet Contractors | 2025 | RN 140,0 MDH | RN 222,3 MDH |
| Auto Nejma | 2025 | RN 300,0 MDH | RN 324 MDH |

Chaque société a donc fait l'objet d'un **contrôle explicite social/consolidé**,
consigné dans la table `SOURCES` du script de collecte. C'est le piège numéro un
de cette collecte, et il ne se voit pas : les deux jeux de chiffres sont
plausibles, seul le rapprochement les distingue.

---

## 3. Comment ces états ont été atteints

### Ce qui ne marche pas

Le **Rapport Financier Annuel** (RFA) est le document de référence déposé à
l'AMMC (`ammc.ma/sites/default/files/{Societe}_RFA_{annee}.pdf`). Il contient
tout — et il est inexploitable automatiquement : 10 à 33 Mo, 150 à 300 pages,
avec les comptes sociaux relégués en annexe. Tout extracteur de texte tronque
avant de les atteindre. Sur douze sociétés, la voie RFA n'a abouti directement
que pour une seule (Delta Holding).

### Ce qui marche

Les mêmes états de synthèse figurent, **intégralement**, dans des documents
courts et autonomes. Par ordre d'efficacité constatée :

| Rang | Document | Pourquoi |
|---|---|---|
| 1 | **Avis de convocation à l'AGO** | Reproduit bilan actif BRUT/AMORT/NET, bilan passif et CPC complets. Publié mars-juin, sur le site de l'émetteur et souvent relayé par l'AMMC |
| 2 | **Publication annuelle légale** | Même contenu, format d'annonce légale |
| 3 | **Communiqué financier annuel** | Souvent complet, parfois limité aux agrégats |
| 4 | **Rapport financier semestriel** | Sa colonne comparative donne une **lecture indépendante** du bilan de clôture N−1 |

Le quatrième mérite d'être souligné : c'est l'outil qui a permis de **trancher
les valeurs douteuses**. Un chiffre lu une fois est une hypothèse ; le même
chiffre retrouvé dans la colonne comparative d'un document publié six mois plus
tard, par un autre canal, est une donnée.

---

## 4. Contrôle qualité — quatre identités, rejouées à chaque exécution

`python scripts/collecte_ammc.py` recalcule les contrôles et refuse de passer
silencieusement sur un écart.

1. **Équation du bilan** — total actif = capitaux propres + capitaux propres
   assimilés + provisions durables + dettes de financement + total passif
   circulant + trésorerie passif.
2. **Concordance du résultat net** entre le CPC et le détail des capitaux
   propres du bilan passif.
3. **Recomposition des capitaux propres** — capital + primes + réserve légale +
   autres réserves + report à nouveau + résultat net.
4. **Cascade du CPC** — exploitation + financier = courant ; courant + non
   courant = avant impôts ; avant impôts − IS = net.

**Résultat : 25 lignes sur 26 bouclent à 0,000 %.**

Deux cas méritent d'être explicités plutôt que masqués :

- **Jet Contractors 2025** laisse un résidu de 184 692,03 DH, soit **0,0033 %**
  du bilan. Ce n'est pas une erreur de saisie : c'est le poste « Écarts de
  conversion — Passif (durables) », élément du financement permanent au CGNC,
  que le schéma de liasse retenu ici ne comporte pas. Le résidu est nul en 2024,
  où ce poste est à néant.
- **Med Paper 2020** ne peut pas être contrôlée : son bilan actif n'était pas
  intégralement lisible dans le document source, et plusieurs postes du passif
  manquent. La ligne est conservée pour l'historique du contribuable, mais le
  script la classe explicitement « contrôle impossible » plutôt que « conforme ».

---

## 5. Ce qui a été volontairement laissé vide

**46 cellules sur 728 (6,3 %) sont vides.** Aucune n'a été comblée.

Le cas le plus fréquent est `etat_impots_sur_resultats_dette`, absent chez huit
sociétés sur douze. La raison est structurelle : leur bilan passif ne publie
qu'un poste **« État » agrégé**, mélangeant TVA, IS, IR et autres taxes, et la
ventilation ne figure que dans l'ETIC détaillé. Retenir le poste agrégé aurait
produit une colonne remplie et fausse — pour Auto Nejma 2025, 196,7 MDH d'« État »
contre 186,4 MDH d'IS effectivement dû au CPC : plausible, et faux.

C'est le principe qui a gouverné toute la collecte : **un champ illisible reste
vide**. Une valeur vide se voit et se traite (`NaN`, jamais `0` — voir
[`02_mapping_liasse_fiscale.md`](02_mapping_liasse_fiscale.md)) ; une valeur
inventée se propage silencieusement dans tous les ratios qui la mobilisent.

Il faut le distinguer du **vrai zéro**, qui est une information : Delta Holding
n'a pas de stocks parce que c'est une holding, Cosumar n'a pas d'achats revendus
de marchandises parce qu'elle transforme. Dans ces cas le zéro est retenu — et
seulement lorsqu'il est confirmé par la réconciliation exacte d'un sous-total.

---

## 6. Une erreur d'extraction instructive

L'extraction de Label Vie, faite d'abord sur le seul communiqué annuel, a produit
un jeu de valeurs **cohérent, équilibré à 0,000 % — et faux**. Un décalage
systématique d'une ligne dans les tableaux avait :

- affecté aux immobilisations financières le **net des immobilisations
  corporelles** ;
- affecté au report à nouveau le **résultat net de l'exercice** (d'où une ligne
  « résultat net » apparemment vide au bilan passif) ;
- inversé les colonnes 2024 / 2023 sur les lignes de charges du CPC ;
- produit un résultat net social de **−430,7 MDH** là où la société a réalisé
  **+153,4 MDH**.

L'équation du bilan bouclait pourtant parfaitement, parce qu'un décalage
uniforme préserve les totaux. Ce qui a permis de détecter l'erreur, c'est la
**recomposition des capitaux propres** (contrôle n° 3) : capital 289,4 MDH +
prime 1 125,0 MDH + réserve légale 28,9 MDH + report à nouveau + résultat net
ne retombait sur le total publié qu'avec la bonne affectation des deux dernières
lignes. La correction a ensuite été confirmée par la résolution d'affectation du
résultat votée en assemblée générale — un document juridique, indépendant des
états comptables.

**L'enseignement dépasse cette collecte.** Un contrôle d'équilibre est nécessaire
et **notoirement insuffisant** : il valide la structure, pas l'affectation. Il
faut au moins une seconde identité, portant sur une décomposition et non sur un
total. C'est la même logique que celle qui gouverne la suite de tests du projet
(voir [`05_guide_de_test.md`](05_guide_de_test.md)) : traquer les défauts qui ne
lèvent aucune erreur et produisent des résultats plausibles.

---

## 7. Ce que ce jeu établit, et ce qu'il n'établit pas

| Conclusion | Statut | Pourquoi |
|---|---|---|
| La correspondance CGNC → variables du modèle fonctionne sur de vraies liasses marocaines | 🟢 **Solide** | `verifier_liasse` ne remonte aucune anomalie ; `mapper_liasse` atteint 80,4 % de couverture (24 variables sur 28), les 4 manquantes étant les variables hors liasse déjà documentées |
| Le dispositif se comporte correctement sur des données réelles hétérogènes | 🟢 **Solide** | Chaîne complète exécutée : lecture → contrôles → mapping → règle de rupture → classement motivé, sur un jeu couvrant un facteur 97 en chiffre d'affaires |
| Ce jeu permet de mesurer un taux de détection | 🔴 **Non** | **Aucune étiquette de fraude.** Aucune de ces sociétés n'est réputée fraudeuse. Rien ici ne mesure une performance |
| Ce jeu peut entraîner le modèle de reconstitution | 🔴 **Non** | 12 contribuables, contre `EFFECTIF_MINIMUM = 40` (`src/ciblage.py`). En dessous, un modèle appris reproduirait du bruit |
| Ce jeu représente le tissu fiscal d'une direction régionale | 🔴 **Non** | Biais de sélection maximal : les sociétés cotées sont les plus grandes, les plus auditées et les plus contrôlées du pays |

Deux limites techniques s'ajoutent :

- **Profondeur d'historique.** Dix sociétés sur douze n'ont que 2 exercices, là
  où [`02_mapping_liasse_fiscale.md`](02_mapping_liasse_fiscale.md) demande au
  moins 3 exercices consécutifs. Les variations sur 2 exercices ne sont donc pas
  calculables pour elles, et la règle de rupture travaille sur une seule
  comparaison au lieu d'une trajectoire.
- **Ruptures légitimes.** Risma 2024 (−400,8 MDH de résultat non courant après
  une restructuration de participations) et Delta Holding (holding pure sans
  stocks, dont 94 % du résultat provient de titres de participation) produisent
  des signaux forts qui n'ont **rien de fiscal**. C'est précisément leur
  intérêt : ce sont des faux positifs réels, du type que la simulation ne génère
  pas, et qui rappellent qu'un écart signale une **incohérence déclarative à
  examiner**, jamais une fraude.

---

## 8. Reproduire

```bash
python scripts/collecte_ammc.py             # régénère le CSV + rapport de contrôle
python scripts/collecte_ammc.py --verifier  # contrôles seuls, sans réécriture
```

Le script ne consulte pas le site de l'AMMC en direct : il **fige** le résultat
d'une lecture manuelle document par document, avec sa provenance complète (table
`SOURCES`, une entrée par société : URLs des documents effectivement lus et
trace du contrôle social/consolidé). C'est ce qui rend le jeu de données
auditable — un lecteur du mémoire peut rouvrir chaque source et refaire la
lecture ligne à ligne.

Vérification de bout en bout dans l'application :

```python
from src.liasse_mapping import lire_liasse, verifier_liasse, mapper_liasse
from src.ciblage import analyser_portefeuille

liasses = lire_liasse("data/exemples_marocains/societes_cotees_reelles.csv")
print(verifier_liasse(liasses))          # {'scorable': True, 'anomalies': [], ...}

donnees, rapport = mapper_liasse(liasses, unite="dirhams", convertir_en_usd=False)
print(rapport["taux_couverture_global_pct"])   # 80.4

classement, diagnostic = analyser_portefeuille(donnees, methode="regles")
print(classement[["raison_sociale", "fyear", "score_ciblage", "motif"]].head())
```

La méthode `"regles"` est ici la bonne : avec 12 contribuables,
`analyser_portefeuille` écarte de toute façon le volet IA et le signale dans son
diagnostic (`ia_disponible: False`) — comportement attendu, et testé.

> **Rappel.** Ces données sont **publiques et publiées** par les émetteurs
> eux-mêmes. Elles ne relèvent en rien du secret professionnel de l'article 246
> du CGI, contrairement à tout export du système d'information de
> l'administration, qui ne doit **jamais** être déposé dans ce dépôt.
