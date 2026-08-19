# Scénario d'insertion à la Direction Régionale des Impôts de Tétouan

Ce document décrit **où** le dispositif se brancherait dans le travail réel
d'une direction régionale, **sous quelles garanties juridiques**, et **comment
mesurer s'il apporte quelque chose**. C'est la pièce qui rattache un travail de
*machine learning* au sujet posé.

Il s'agit d'un scénario documenté, non d'un déploiement réalisé : aucune donnée
de la DRI n'a été utilisée (voir [`01_cadrage_methodologique.md`](01_cadrage_methodologique.md) §2).

---

## 1. Le problème métier : programmer le contrôle sous contrainte

Une direction régionale gère un portefeuille de plusieurs milliers de
contribuables soumis à l'IS et à l'IR professionnel. Ses brigades de
vérification ne peuvent en contrôler qu'une fraction chaque année — quelques
pourcents au plus. **La question opérationnelle n'est donc jamais « qui
fraude ? » mais « par lesquels commencer ? »**

C'est exactement la forme du problème que le modèle traite. Il ne rend pas de
verdict : il ordonne un portefeuille par risque décroissant, pour qu'une capacité
de contrôle donnée soit dirigée là où le rendement attendu est le plus élevé.

Deux conséquences pratiques :

- La métrique pertinente n'est ni l'exactitude ni l'AUC-ROC, mais le
  **rendement à budget constant** : à nombre de vérifications inchangé, combien
  de redressements de plus qu'avec la programmation actuelle ?
- Un modèle dont la performance absolue est modeste peut être opérationnellement
  utile. Capter 13 % des cas en examinant 5 % du portefeuille, contre 5 % par
  tirage aléatoire, représente un gain d'efficacité de l'ordre de 2,5× sur la
  même dépense de moyens.

---

## 2. Les données existent déjà, sous forme électronique

Un point souvent négligé : le dispositif ne suppose aucune collecte nouvelle.

La **télédéclaration et le télépaiement sont obligatoires** pour les personnes
morales et physiques exerçant une activité professionnelle depuis le
1<sup>er</sup> janvier 2017. Les liasses fiscales arrivent donc à la DGI sous
forme structurée, via le portail **SIMPL**. L'article 210 du CGI prévoit par
ailleurs expressément que les contribuables tenant une comptabilité par
procédés électroniques présentent leurs documents comptables **sur support
électronique** et en remettent copie sous format dématérialisé.

Autrement dit, la matière première du modèle — les postes du bilan et du CPC,
par contribuable et par exercice — est déjà constituée, déjà numérique, et déjà
détenue par l'administration. Ce qui manque n'est pas la donnée d'entrée mais
**l'étiquette** : l'issue définitive des vérifications, appariée aux liasses.

---

## 3. Insertion dans le processus de programmation

```
Liasses télédéclarées (SIMPL)
        │
        ▼
  ┌──────────────────────┐
  │ Extraction annuelle  │   un enregistrement par (contribuable, exercice),
  │ 3 à 5 exercices      │   3 exercices minimum pour les variables de trajectoire
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ Contrôles de         │   équilibre du bilan, doublons, complétude
  │ cohérence            │   → src.liasse_mapping.verifier_liasse()
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ Correspondance CGNC  │   28 variables reconstituées, mise à l'échelle
  │ → variables modèle   │   → src.liasse_mapping.mapper_liasse()
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ Scoring + SHAP       │   score de risque + décomposition par variable
  │                      │   → src.inference.score_with_diagnostics()
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ Liste de travail     │   dossiers classés, avec les motifs de signalement
  │ soumise au chef de   │
  │ brigade              │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ ARBITRAGE HUMAIN     │   ◄── la décision de vérifier reste ici, et nulle part ailleurs
  └──────────┬───────────┘
             ▼
     Programme annuel de vérification
             │
             ▼
  ┌──────────────────────┐
  │ Retour d'expérience  │   issue des contrôles → réentraînement annuel
  └──────────────────────┘
```

Le modèle **précède** la décision de vérifier ; il ne la remplace pas. Il produit
une liste de travail argumentée, que le chef de brigade croise avec ce que
l'outil ignore : signalements, recoupements de tiers, antériorité de contrôle,
contexte sectoriel local, informations issues de la DGI centrale.

---

## 4. Cadre juridique du dispositif

| Question | Fondement | Conséquence pratique |
|---|---|---|
| L'administration peut-elle exploiter les liasses à des fins de ciblage ? | Art. 210 CGI — droit de contrôle des déclarations | Oui : le ciblage est un acte préparatoire au contrôle, pas une procédure autonome |
| Un score suffit-il à déclencher une vérification ? | Art. 212 CGI — avis de vérification, procédure formalisée | Non : la vérification obéit à une procédure propre. Le score n'est qu'un critère de programmation interne |
| Un score peut-il fonder un redressement ? | Art. 220 et s. CGI — procédure contradictoire | **Non.** Le redressement se fonde sur des constatations, jamais sur un score. Le modèle est en amont, sans valeur probante |
| Les données de ciblage sont-elles protégées ? | Art. 246 CGI — secret professionnel | Oui : scores, listes et modèles relèvent du même secret que les dossiers eux-mêmes |
| Le traitement est-il soumis à la loi 09-08 ? | Loi 09-08 — protection des données personnelles, CNDP | Oui dès lors que des personnes physiques sont concernées : formalité préalable, finalité déterminée, durée de conservation bornée |

**Le point à ne pas manquer en soutenance :** un score de risque n'a aucune
valeur probante. Il n'est ni un commencement de preuve, ni un motif opposable au
contribuable. Il oriente l'allocation de moyens de contrôle — rien de plus. Le
présenter autrement serait juridiquement faux et invaliderait la procédure.

---

## 5. Pourquoi l'interprétabilité n'est pas un supplément d'âme

Une liste de matricules classés sans justification est inexploitable. Un chef de
brigade doit pouvoir dire pourquoi il retient un dossier, et un vérificateur
doit savoir où regarder en priorité.

C'est la fonction de la décomposition **SHAP** intégrée à l'outil : pour chaque
dossier signalé, elle indique les variables qui ont poussé le score vers le haut
et celles qui l'ont retenu vers le bas. Une sortie typique se lit ainsi :

> Dossier signalé principalement par : une progression des créances clients sans
> progression du chiffre d'affaires (`rect_log_d1` élevé, `sale_log_d1` faible),
> un taux effectif d'imposition inférieur à la norme sectorielle
> (`effective_tax_rate`), et des *accruals* élevés relativement au total de
> bilan (`accruals`).

Cette formulation est actionnable : elle désigne les postes à examiner. Elle
autorise aussi le désaccord — un vérificateur qui connaît une explication
légitime (litige commercial, report déficitaire, changement de méthode) peut
écarter le dossier en connaissance de cause. **Un modèle contestable est un
modèle utilisable ; une boîte noire ne l'est pas.**

Trois des six ratios métier (`tax_burden`, `effective_tax_rate`, `tax_to_sales`)
ont été construits précisément pour parler ce langage : ils sont directement
lisibles par un fiscaliste sans médiation technique.

---

## 6. Gouvernance et garde-fous

1. **Décision humaine systématique.** Aucune programmation automatique. Le
   modèle propose, le chef de brigade dispose, et sa décision est tracée.

2. **Journalisation.** Conserver, pour chaque campagne : la version du modèle,
   les données d'entrée, les scores produits, les décisions prises et leurs
   motifs. C'est la condition d'un audit *a posteriori* et d'un réentraînement
   sur des étiquettes fiables.

3. **Surveillance des biais.** Contrôler à chaque campagne que le classement ne
   concentre pas mécaniquement les contrôles sur un secteur, une commune ou une
   classe de taille. Le portefeuille d'une direction régionale n'est pas
   homogène : dans la région de Tétouan, l'agroalimentaire, le tourisme, le BTP
   et le négoce frontalier ont des profils comptables structurellement
   différents. Un modèle risque d'apprendre le secteur plutôt que la fraude.
   La parade est un contrôle explicite des taux de signalement par strate, et le
   cas échéant une normalisation sectorielle des ratios.

4. **Boucle de rétroaction maîtrisée.** Réentraîner sur les seuls dossiers
   contrôlés enferme le modèle dans ses propres choix : il apprend à reproduire
   la sélection passée. Réserver une part aléatoire du programme annuel de
   vérification est le prix à payer pour conserver un échantillon non biaisé.
   Cette part est un investissement dans la qualité du ciblage futur, pas une
   perte de rendement.

5. **Péremption.** Les comportements évoluent ; les dispositifs légaux aussi.
   Réentraînement annuel, et suivi de la dérive des distributions d'entrée d'un
   exercice à l'autre.

6. **Confidentialité.** Scores, listes et modèles relèvent du secret
   professionnel de l'article 246 CGI, au même titre que les dossiers. Le
   dispositif doit rester dans le système d'information de l'administration.

---

## 7. Ce qui reste à faire avant tout déploiement

Le tableau ci-dessous distingue ce que le mémoire livre de ce qui relève de
l'administration. Le présenter tel quel en soutenance désamorce la question
« votre travail est-il utilisable ? ».

| Étape | Statut | Responsable |
|---|---|---|
| Chaîne de traitement, modèle entraîné, validation temporelle | ✅ livré | mémoire |
| Correspondance liasse fiscale marocaine → variables du modèle | ✅ livré | mémoire |
| Outil de restitution et d'interprétation (dashboard, SHAP) | ✅ livré | mémoire |
| Scénario d'insertion et cadre juridique | ✅ livré | mémoire |
| Constitution d'un jeu étiqueté interne (liasses × issue des contrôles) | ⬜ à faire | DGI / DRI |
| Traitement du biais de sélection des dossiers contrôlés | ⬜ à faire | DGI / DRI |
| Réentraînement sur données marocaines | ⬜ à faire | DGI / DRI |
| Évaluation comparative à budget de contrôle constant | ⬜ à faire | DRI |
| Formalités CNDP et validation juridique | ⬜ à faire | DGI |

Le protocole détaillé des quatre premières lignes « à faire » figure au §6 de
[`01_cadrage_methodologique.md`](01_cadrage_methodologique.md).

---

## 8. Réponses aux objections prévisibles

**« Votre modèle est entraîné sur des entreprises américaines, il ne vaut rien
ici. »**
Sur ce point précis, l'objection est **fondée, et le travail l'établit
lui-même** : testé sur un portefeuille marocain simulé, le modèle classe par
taille d'entreprise et non par anomalie (AUC-ROC 0,46, inférieure au hasard).
Ce n'est pas une faiblesse dissimulée du mémoire, c'est son résultat principal
— voir [`06_transferabilite.md`](06_transferabilite.md).

Ce qui reste transposable et démontré : la chaîne de traitement, la
correspondance des variables comptables, le protocole de validation temporelle,
et le constat, chiffré, de ce qui bloque le transfert. La réponse pratique est
double : réentraîner sur données marocaines en normalisant les variables de
niveau par strate, et utiliser d'ici là la **règle de rupture
intra-contribuable**, qui ne requiert aucun apprentissage et se révèle neutre en
taille (corrélation +0,02).

**« 13 % de fraudes captées, c'est faible. »**
C'est 2,5 fois le tirage aléatoire, à budget de contrôle identique. Sur un
portefeuille de plusieurs milliers de dossiers, la question n'est pas la valeur
absolue mais le gain marginal par rapport à la méthode de programmation en
vigueur. Et un modèle qui annoncerait 90 % de détection sur un phénomène dont la
prévalence est inférieure à 1 % devrait éveiller la méfiance, pas l'enthousiasme.

**« Pourquoi pas une approche non supervisée, puisque vous n'avez pas
d'étiquettes ? »**
Elle a été testée — Isolation Forest, LOF, autoencodeur — et elle échoue :
performances au niveau du hasard (notebook 03, `reports/comparison_unsupervised.csv`).
Le résultat est conservé dans le mémoire parce qu'il est instructif : en matière
fiscale, l'atypique n'est pas le frauduleux. Une entreprise aux ratios
inhabituels est le plus souvent une entreprise d'un secteur particulier.

**« Quatre des variables de votre modèle n'existent même pas dans une liasse
marocaine. »**
Exact, et cela a été mesuré plutôt que supposé. Le test de robustesse
([`02_mapping_liasse_fiscale.md`](02_mapping_liasse_fiscale.md) §8) montre que
le classement résiste — corrélation de rang de 0,865, AUC-ROC pratiquement
inchangée — mais que le niveau du score est fortement déplacé. D'où deux règles
d'usage : raisonner exclusivement en rang, jamais par seuil absolu ; et retirer
ces quatre variables du jeu lors du réentraînement sur données marocaines,
plutôt que de les laisser vides.

**« Un algorithme va-t-il décider des contrôles ? »**
Non, et le dispositif est construit pour que ce soit impossible : le modèle
produit une liste argumentée, la décision reste au chef de brigade, elle est
tracée, et le score n'a aucune valeur probante dans la procédure de rectification.
