# Cadrage méthodologique

**Sujet :** *Détection d'anomalies fiscales par des techniques de l'Intelligence
Artificielle : cas de la Direction Régionale des Impôts de Tétouan.*

Ce document répond à la question qu'un jury posera en premier : **sur quelles
données ce travail a-t-il été fait, et en quoi concerne-t-il la DRI de Tétouan ?**
Il est écrit pour être repris directement dans la partie méthodologique du
mémoire.

---

## 1. Ce que le sujet demande, et ce que ce travail livre

Le sujet comporte deux volets qu'il faut distinguer, parce qu'ils n'ont pas le
même statut de faisabilité :

| Volet du sujet | Traité ici | Comment |
|---|---|---|
| Techniques d'IA appliquées à la détection d'anomalies fiscales | **Oui, intégralement** | Chaîne complète : exploration, ingénierie de variables, modèles non supervisés (Isolation Forest, LOF, autoencodeur), modèles supervisés (XGBoost optimisé par Optuna), validation temporelle stricte, interprétabilité SHAP, outil de restitution |
| Cas de la Direction Régionale des Impôts de Tétouan | **Comme cadre d'application, pas comme jeu d'entraînement** | Passerelle liasse fiscale marocaine → modèle, scénario d'insertion dans le processus de programmation du contrôle, protocole de validation sur données réelles |

L'entraînement et l'évaluation reposent sur le jeu **JAR2020**. Ce choix n'est
pas un pis-aller dissimulé : c'est une contrainte structurelle, exposée
ci-dessous, et la façon dont elle est traitée fait partie de la contribution du
travail.

> **Formulation recommandée du titre**
>
> *Détection d'anomalies fiscales par des techniques de l'Intelligence
> Artificielle : conception et validation méthodologique d'un dispositif de
> ciblage, appliqué au contexte de la Direction Régionale des Impôts de Tétouan.*
>
> Le sujet initial peut être conservé si le corps du mémoire porte clairement le
> cadrage ci-dessous. Ce qui serait indéfendable, c'est de laisser croire que le
> modèle a été entraîné sur des dossiers de la DRI.

---

## 2. Pourquoi les données de la DRI ne peuvent pas constituer le jeu d'entraînement

Quatre obstacles, dont trois sont juridiques et un est méthodologique. Aucun
n'est contournable dans le cadre d'un mémoire.

### 2.1 Le secret professionnel fiscal — article 246 du CGI

L'article 246 du Code Général des Impôts astreint au secret professionnel
« toutes les personnes appelées à l'occasion de leurs fonctions ou attributions à
intervenir dans l'établissement, le contrôle, la perception ou le contentieux
des impôts ». La communication de renseignements ou de copies de documents à
des tiers — un étudiant, un chercheur — est interdite hors décision de justice.
Les seules levées prévues bénéficient à des administrations limitativement
énumérées (douanes, Trésorerie, Office des changes, sécurité sociale), sur
demande écrite adressée au ministre chargé des finances.

Conséquence directe : **aucune extraction nominative ou même anonymisée de
dossiers de la DRI ne peut être remise dans le cadre d'un travail académique.**

### 2.2 La protection des données à caractère personnel — loi 09-08

Le traitement automatisé de données relatives à des personnes physiques
(entreprises individuelles, professions libérales, dirigeants) relève de la loi
09-08 et du contrôle de la CNDP : déclaration ou autorisation préalable,
finalité déterminée, proportionnalité, durée de conservation limitée. Un
traitement de profilage à finalité de contrôle fiscal ne s'improvise pas hors du
cadre institutionnel de la DGI.

### 2.3 Il n'existe pas d'équivalent marocain public de JAR2020

Aux États-Unis, la SEC publie les *Accounting and Auditing Enforcement Releases*
(AAER) : une liste publique et datée d'entreprises sanctionnées pour états
financiers frauduleux. C'est cette publication qui rend possible un jeu de
données étiqueté. **Le Maroc n'a pas d'équivalent public.** Les redressements
notifiés par la DGI ne font l'objet d'aucune publication nominative, et les
décisions des commissions de taxation ne sont pas diffusées sous une forme
exploitable. Chercher un « JAR2020 marocain » revient à chercher quelque chose
qui n'existe pas.

### 2.4 L'étiquette elle-même est difficile à construire

Même avec un accès autorisé, définir la variable cible poserait un problème de
fond. Un redressement notifié n'est pas une fraude établie : il ouvre une
procédure contradictoire (articles 220 et suivants du CGI), susceptible de
recours devant les commissions locale puis nationale, puis devant le juge. Un
étiquetage fiable suppose donc d'attendre l'épuisement des voies de recours —
soit plusieurs années. C'est exactement le **biais de censure** que l'on observe
sur JAR2020 après 2010, et qui explique la chute des métriques sur 2012-2014
(voir §5).

### 2.5 Ce qui reste accessible : des liasses marocaines réelles, mais non étiquetées

L'inaccessibilité porte sur l'**étiquette**, pas sur la liasse. Les sociétés
cotées à la Bourse de Casablanca publient chaque année leurs **états de synthèse
au format CGNC** — bilan actif BRUT/AMORT/NET, bilan passif, CPC — au titre de
l'information périodique imposée par l'AMMC (loi 44-12). Ce sont de véritables
liasses marocaines, auditées, opposables et vérifiables par un tiers.

Elles ont été collectées : **12 sociétés, 26 exercices, 2020-2025**
(`scripts/collecte_ammc.py`). Elles ne résolvent pas le problème de l'étiquette
— aucune de ces sociétés n'est réputée fraudeuse, et rien ne permet d'y mesurer
un taux de détection. Elles règlent en revanche une question distincte, et
souvent confondue avec la première : **le dispositif fonctionne-t-il sur de
vraies données marocaines, ou seulement sur notre simulation ?**

La distinction structure tout le travail :

| Question | Réponse | Sur quoi |
|---|---|---|
| La correspondance CGNC → modèle tient-elle sur du réel ? | 🟢 Oui | 26 liasses réelles, 80,4 % de couverture, aucune anomalie de cohérence |
| Le dispositif détecte-t-il la fraude au Maroc ? | ⚪ **Non mesurable** | Aucune étiquette n'existe, ni publique ni accessible |

📖 [`08_donnees_reelles_ammc.md`](08_donnees_reelles_ammc.md)

---

## 3. Le jeu JAR2020 : ce qu'il est, et pourquoi il est un substitut défendable

**Référence.** Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020). *Detecting
Accounting Fraud in Publicly Traded U.S. Firms Using a Machine Learning
Approach.* Journal of Accounting Research, 58(1), 199-235.

**Contenu.** 146 045 observations (entreprise × exercice), 1991-2014, 28
variables comptables brutes issues de Compustat, étiquetées à partir des AAER de
la SEC. Taux de fraude : 0,66 % — soit un déséquilibre d'environ 150 pour 1.

**Ce qui rend le transfert méthodologique légitime :**

1. **Le mécanisme étudié est le même.** La fraude fiscale par minoration de
   base imposable et la fraude aux états financiers passent toutes deux par la
   manipulation des mêmes postes : produits, charges, stocks, créances,
   provisions, accruals. Ce ne sont pas deux phénomènes différents qui se
   trouveraient partager un vocabulaire — c'est la même mécanique comptable.

2. **Les variables d'entrée sont universelles.** Les 28 variables sont des
   postes de bilan et de compte de résultat, non des spécificités du droit
   américain. Le document [`02_mapping_liasse_fiscale.md`](02_mapping_liasse_fiscale.md)
   établit leur correspondance poste à poste avec la liasse fiscale marocaine :
   22 des 28 ont un équivalent direct ou reconstructible.

3. **Le protocole de validation est publié et réplicable.** Split temporel avec
   marge de 2 ans, anti-fuite sur les affaires de fraude pluriannuelles
   (`p_aaer`), test année par année sur 2003-2014. Ce protocole est reproduit à
   l'identique ici, ce qui rend les résultats comparables à la littérature.

4. **Le déséquilibre de classes est du même ordre.** Le taux de fraude détectée
   de 0,66 % est du même ordre de grandeur que les taux de redressement
   significatif observés dans les administrations fiscales. Les difficultés
   techniques que le modèle doit surmonter — métriques trompeuses, seuils
   inopérants, nécessité de raisonner en rang plutôt qu'en probabilité — sont
   donc les mêmes.

**Ce que ce transfert ne permet pas :** affirmer que les taux de détection
mesurés ici seraient obtenus tels quels sur le portefeuille de la DRI de
Tétouan. La structure du tissu économique (prédominance des TPE/PME, poids de
l'informel, secteurs dominants), le référentiel comptable et la nature des
fraudes visées diffèrent. **Les chiffres de performance de ce mémoire mesurent
la validité de la méthode, pas le rendement attendu d'un déploiement.** Cette
phrase doit figurer telle quelle dans la conclusion.

---

## 4. Statut exact du travail

Ce mémoire est une **preuve de faisabilité méthodologique**, articulée en trois
pièces :

| Pièce | Question à laquelle elle répond | Où elle se trouve |
|---|---|---|
| Modèle entraîné et validé sur JAR2020 | *La méthode fonctionne-t-elle ? Fait-elle mieux que le hasard, et de combien ?* | Notebooks 01-05, `reports/` |
| Passerelle liasse fiscale marocaine → modèle | *Les données dont dispose réellement la DRI peuvent-elles alimenter ce modèle ?* | `src/liasse_mapping.py`, [`02_mapping_liasse_fiscale.md`](02_mapping_liasse_fiscale.md), onglet « Liasse marocaine » du dashboard |
| Scénario d'insertion opérationnelle | *Où le dispositif se brancherait-il dans le travail de la DRI, et sous quelles garanties ?* | [`03_deploiement_DRI_Tetouan.md`](03_deploiement_DRI_Tetouan.md) |

La troisième pièce est ce qui rattache le travail au sujet. Sans elle, le
mémoire serait un exercice de *machine learning* sur données américaines ; avec
elle, c'est une proposition de dispositif dont le seul maillon manquant — les
données réelles — est explicitement identifié comme relevant de l'administration
et non de l'étudiant.

---

## 5. Limites à assumer explicitement

À reprendre dans la section « Limites » du mémoire. Les énoncer soi-même vaut
toujours mieux que de les voir soulevées en soutenance.

1. **Périmètre des données d'entraînement.** Entreprises américaines cotées,
   normes US GAAP, fraudes aux états financiers sanctionnées par la SEC. Ni le
   référentiel comptable, ni la taille des entreprises, ni la nature de
   l'infraction ne coïncident exactement avec le portefeuille d'une direction
   régionale marocaine.

2. **Performance modeste en valeur absolue.** L'AUC-PR agrégée du meilleur
   modèle est de **0,0168** (0,0153 hors 2011, exercice atypique), contre une
   référence de hasard à **0,0066**. Le gain est réel — de l'ordre de 2,5× — mais
   il ne s'agit pas d'un détecteur : c'est un **instrument de priorisation**. En
   investiguant les 5 % de dossiers les mieux classés, on capte environ 13 % des
   fraudes, contre 5 % par tirage aléatoire. C'est utile à budget de contrôle
   constant ; ce n'est pas de la détection au sens courant du terme.

3. **Le score n'est pas une probabilité.** Le modèle est entraîné avec un
   `scale_pos_weight` d'environ 121 pour compenser le déséquilibre : les
   probabilités qu'il produit sont mécaniquement gonflées. Un score de 0,70 ne
   signifie pas « 70 % de chances de fraude ». **Seul le classement fait sens**,
   et toute lecture du score comme une probabilité serait une erreur
   d'interprétation.

4. **Biais de censure des étiquettes.** Les sanctions AAER mettent plusieurs
   années à être publiées : les exercices récents (2012-2014) comportent des
   fraudes non encore étiquetées, comptabilisées comme non-fraudes. Les
   métriques y sont mécaniquement dégradées. Le même biais frapperait tout jeu
   construit sur des redressements marocains définitifs.

5. **Les variables de trajectoire exigent de l'historique.** 20 des 54 variables
   comparent un contribuable à lui-même 1 ou 2 exercices plus tôt. Un fichier ne
   contenant qu'un seul exercice prive le modèle de sa dimension dynamique, qui
   est précisément l'un de ses apports. Le taux de couverture est affiché à
   chaque scoring plutôt que passé sous silence.

6. **L'apport non supervisé est faible.** Isolation Forest, LOF et l'autoencodeur
   font à peine mieux que le hasard (AUC-ROC ≈ 0,50 pour l'autoencodeur). C'est
   un résultat négatif, conservé et documenté : il montre qu'en matière fiscale,
   l'atypique n'est pas le frauduleux. Une entreprise aux ratios inhabituels est
   souvent une entreprise d'un secteur particulier, pas un fraudeur. Ce constat
   a une portée pratique : il déconseille les approches purement non supervisées
   pour ce cas d'usage.

7. **Quatre variables du modèle n'existent pas dans la liasse marocaine.**
   `prcc_f`, `csho`, `dltis` et `sstk` sont laissées vides. Le test de
   robustesse conduit à ce sujet
   ([`scripts/robustesse_variables_absentes.py`](../scripts/robustesse_variables_absentes.py),
   résultats détaillés au §8 de [`02_mapping_liasse_fiscale.md`](02_mapping_liasse_fiscale.md))
   montre que **le classement résiste** (Spearman 0,865 ; AUC-ROC 0,714 → 0,712)
   mais que **le niveau du score est fortement déplacé** (moyenne 0,264 → 0,471).
   Deux conséquences à énoncer : le dispositif ne doit s'utiliser qu'en rang, et
   à budget de contrôle fixé les deux configurations ne programment que 51 à
   66 % des mêmes dossiers.

8. **Le modèle ne se transfère pas tel quel au contexte marocain.** Testé sur un
   portefeuille marocain simulé (880 liasses), il classe par **taille
   d'entreprise** (corrélation +0,59) et non par anomalie (−0,03) ; son AUC-ROC
   de 0,46 y est inférieure au hasard. La cause est structurelle : JAR2020 ne
   contient que des sociétés cotées américaines, dont le bilan médian est ~100
   fois supérieur à celui d'un contribuable de direction régionale, et 48 des 54
   variables du modèle sont des variables de niveau.
   **C'est la limite la plus importante — et c'est aussi un résultat en soi**,
   puisqu'elle délimite expérimentalement ce qui est transposable (la méthode,
   la correspondance des variables, le protocole) et ce qui ne l'est pas (les
   coefficients appris). Analyse complète :
   [`06_transferabilite.md`](06_transferabilite.md).

9. **Aucune validation sur données marocaines réelles.** Les jeux marocains de
   ce projet sont simulés et leurs anomalies plantées : ils testent la chaîne de
   traitement, ils ne valident pas la détection. Cette limite est irréductible
   dans le cadre du mémoire. Le protocole qui permettrait de la lever est décrit
   au §6.

---

## 6. Protocole de validation sur données réelles

Ce que la DRI devrait faire pour transformer cette preuve de faisabilité en
dispositif opérationnel. Cette section vaut proposition de recherche et
constitue une réponse naturelle à la question « et ensuite ? » en soutenance.

**Étape 1 — Constitution d'un jeu étiqueté interne.**
Extraire, sur les 6 à 10 derniers exercices, les liasses des contribuables
relevant de la direction, appariées avec l'issue des vérifications de
comptabilité. Étiquette proposée : `1` si le contrôle a donné lieu à un
redressement définitif (recours épuisés) supérieur à un seuil de
matérialité — en montant absolu et en pourcentage de la base déclarée —, `0`
sinon. Les dossiers dont le contentieux est en cours doivent être écartés, non
étiquetés `0`.

**Étape 2 — Correction du biais de sélection.**
Difficulté centrale, à traiter explicitement : les dossiers vérifiés ne sont pas
un échantillon aléatoire du portefeuille. Ils ont été choisis par les méthodes
de programmation actuelles, elles-mêmes fondées sur des critères de risque. Un
modèle entraîné sur ces seuls dossiers apprendrait à reproduire la sélection
existante plutôt qu'à détecter la fraude. Deux réponses possibles :
réserver une fraction aléatoire du programme annuel de contrôle pour constituer
un échantillon non biaisé, ou recourir à des méthodes de correction du biais de
sélection (pondération par la propension à être contrôlé).

**Étape 3 — Réentraînement et validation temporelle.**
Réappliquer le protocole des notebooks 04 et 05 — mêmes variables, mêmes
familles de modèles, même split temporel avec marge — sur les données internes.
Les hyperparamètres retenus ici (`models/model_meta_v2.json`) servent de point
de départ à l'optimisation.

**Étape 4 — Évaluation à budget de contrôle constant.**
La métrique décisionnelle n'est ni l'exactitude ni l'AUC-ROC, mais le
**rendement à budget fixé** : parmi les N dossiers que la brigade peut
effectivement vérifier dans l'année, combien de redressements le classement
permet-il d'atteindre, comparé à la programmation actuelle ? C'est cette
comparaison-là qui décide du déploiement.

**Étape 5 — Contrôle des effets indésirables.**
Vérifier que le classement ne concentre pas mécaniquement les contrôles sur un
secteur, une commune ou une taille d'entreprise. Un ciblage efficace mais
systématiquement biaisé serait juridiquement et politiquement intenable.

---

## 7. Références

- Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020). *Detecting Accounting
  Fraud in Publicly Traded U.S. Firms Using a Machine Learning Approach.*
  Journal of Accounting Research, 58(1), 199-235. Données :
  <https://github.com/JarFraud/FraudDetection>
- Royaume du Maroc, *Code Général des Impôts* — art. 210 (droit de contrôle),
  art. 212 (vérification de comptabilité), art. 220 et s. (procédure de
  rectification), art. 246 (secret professionnel).
- Loi n° 09-08 relative à la protection des personnes physiques à l'égard du
  traitement des données à caractère personnel ; Commission Nationale de
  Contrôle de la Protection des Données à Caractère Personnel (CNDP).
- Direction Générale des Impôts, portail de télédéclaration **SIMPL** ;
  télédéclaration et télépaiement obligatoires pour les personnes morales et
  physiques depuis le 1<sup>er</sup> janvier 2017.

> Les numéros d'articles du CGI sont à revérifier dans l'édition en vigueur à la
> date de dépôt du mémoire : la numérotation est stable, mais le contenu est
> modifié chaque année par la loi de finances.
