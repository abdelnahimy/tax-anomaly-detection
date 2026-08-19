# Bibliographie et sources

## 1. Détection de fraude comptable par apprentissage automatique

**Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020).** *Detecting Accounting
Fraud in Publicly Traded U.S. Firms Using a Machine Learning Approach.* Journal
of Accounting Research, 58(1), 199-235.
→ **Référence centrale du projet.** Source du jeu de données et du protocole de
validation temporelle répliqué ici. Données et code MATLAB d'origine :
<https://github.com/JarFraud/FraudDetection>

**Dechow, P. M., Ge, W., Larson, C. R., & Sloan, R. G. (2011).** *Predicting
Material Accounting Misstatements.* Contemporary Accounting Research, 28(1),
17-82.
→ Origine du F-score et des 28 variables comptables brutes reprises par Bao et
al. C'est la référence à citer pour justifier le **choix des variables**, en
amont de toute considération algorithmique.

**Beneish, M. D. (1999).** *The Detection of Earnings Manipulation.* Financial
Analysts Journal, 55(5), 24-36.
→ Le M-score, antécédent classique fondé sur des ratios. Utile comme point de
comparaison : un modèle d'apprentissage doit faire mieux qu'un score linéaire à
huit variables pour justifier sa complexité.

**Perols, J. (2011).** *Financial Statement Fraud Detection: An Analysis of
Statistical and Machine Learning Algorithms.* Auditing: A Journal of Practice &
Theory, 30(2), 19-50.
→ Comparaison systématique d'algorithmes sur données comptables déséquilibrées.

**Jones, J. J. (1991).** *Earnings Management During Import Relief
Investigations.* Journal of Accounting Research, 29(2), 193-228.
→ Modèle de Jones sur les *accruals*, dont s'inspire le ratio `accruals` du
projet.

---

## 2. Apprentissage sur classes déséquilibrées

**Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.*
Proceedings of the 22nd ACM SIGKDD, 785-794.
→ L'algorithme du modèle final. Sa gestion **native des valeurs manquantes**
(direction par défaut apprise à chaque nœud) est ce qui permet de laisser à
`NaN` les variables absentes de la liasse marocaine plutôt que de les combler
par des zéros.

**Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).** *Optuna: A
Next-generation Hyperparameter Optimization Framework.* Proceedings of the 25th
ACM SIGKDD, 2623-2631.
→ Optimisation des hyperparamètres du notebook 05.

**Saito, T., & Rehmsmeier, M. (2015).** *The Precision-Recall Plot Is More
Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced
Datasets.* PLoS ONE, 10(3), e0118432.
→ **Justification du choix de l'AUC-PR** comme métrique principale plutôt que
l'AUC-ROC. Référence à citer pour défendre le protocole d'évaluation.

**He, H., & Garcia, E. A. (2009).** *Learning from Imbalanced Data.* IEEE
Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.
→ Panorama des stratégies de rééquilibrage (dont le `scale_pos_weight` retenu).

**Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008).** *Isolation Forest.* Eighth
IEEE International Conference on Data Mining, 413-422.
→ L'un des trois modèles non supervisés testés au notebook 03 — et écartés.

---

## 3. Interprétabilité

**Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to Interpreting
Model Predictions.* Advances in Neural Information Processing Systems 30.
→ Méthode SHAP, utilisée pour produire les motifs de signalement.

**Lundberg, S. M., Erion, G., Chen, H., et al. (2020).** *From Local
Explanations to Global Understanding with Explainable AI for Trees.* Nature
Machine Intelligence, 2(1), 56-67.
→ TreeSHAP, la variante exacte et rapide employée sur les modèles à arbres.

**Rudin, C. (2019).** *Stop Explaining Black Box Machine Learning Models for
High Stakes Decisions and Use Interpretable Models Instead.* Nature Machine
Intelligence, 1(5), 206-215.
→ **Contrepoint utile.** Argument à connaître pour la soutenance : dans un
contexte administratif à enjeux, une explication *post hoc* ne vaut pas un
modèle intrinsèquement interprétable. La réponse du projet est que le score
n'a aucune valeur probante et que la décision reste humaine — mais l'objection
mérite d'être traitée, pas éludée.

---

## 4. Sources juridiques marocaines

**Code Général des Impôts** (institué par l'article 5 de la loi de finances
n° 43-06, modifié annuellement) :

| Article | Objet | Usage dans le mémoire |
|---|---|---|
| 210 | Droit de contrôle ; présentation des documents comptables, y compris sur support électronique | Fondement de l'exploitation des liasses à des fins de ciblage |
| 212 | Vérification de comptabilité ; avis préalable | Le score ne déclenche pas une vérification : la procédure a ses formes |
| 213 | Pouvoir d'appréciation de l'administration | — |
| 220 et s. | Procédure normale de rectification, contradictoire | **Un score ne peut fonder aucun redressement** |
| 232 | Délais de prescription | Détermine la profondeur d'historique exploitable |
| 246 | **Secret professionnel** | Fondement de l'inaccessibilité des données DRI |

→ Version en vigueur : <https://www.finances.gov.ma> (Direction Générale des
Impôts). **Vérifier la numérotation dans l'édition en vigueur à la date de
dépôt** : elle est stable, mais le contenu est modifié chaque année par la loi
de finances.

**Loi n° 09-08** relative à la protection des personnes physiques à l'égard du
traitement des données à caractère personnel, et son décret d'application
n° 2-09-165. Commission Nationale de Contrôle de la Protection des Données à
Caractère Personnel (**CNDP**) : <https://www.cndp.ma>
→ Encadre tout traitement automatisé portant sur des personnes physiques :
formalité préalable, finalité déterminée, proportionnalité, durée de
conservation bornée.

**Loi n° 17-95** relative aux sociétés anonymes.
→ Citée à propos des actions à dividende prioritaire (variable `pstk`), dont
l'absence de traitement comptable distinct dans le CGNC justifie la convention
retenue.

**Code Général de Normalisation Comptable (CGNC)** et Plan Comptable Général
des Entreprises — loi n° 9-88 relative aux obligations comptables des
commerçants.
→ Référentiel des tableaux de la liasse fiscale : bilan (tableaux n° 1 et 2),
CPC (n° 3), ESG (n° 4), tableau de financement (n° 5), ETIC. **Base de toute
la correspondance du document 02.**

**Télédéclaration et télépaiement.** Obligatoires pour les personnes morales et
les personnes physiques exerçant une activité professionnelle depuis le
1<sup>er</sup> janvier 2017 — portail **SIMPL** de la DGI.
→ <https://www.finances.gov.ma/fr/Pages/detail-actualite.aspx?fiche=2291>
→ Argument de faisabilité : la matière première du modèle est déjà numérique et
déjà détenue par l'administration.

---

## 5. Sources de données marocaines publiques

**Bourse de Casablanca** — <https://www.casablanca-bourse.com>
**Autorité Marocaine du Marché des Capitaux (AMMC)** — <https://www.ammc.ma>
→ États financiers annuels et semestriels des sociétés cotées. **Seule source
comptable marocaine publique, détaillée et pluriannuelle** — donc le seul
support possible pour l'illustration sur données réelles proposée au §9 du
document 02.

**Haut-Commissariat au Plan (HCP)** — <https://www.hcp.ma>
→ Statistiques sectorielles régionales, utiles pour situer le tissu économique
de la région Tanger-Tétouan-Al Hoceïma.

**Direction Générale des Impôts** — rapports d'activité annuels.
→ Statistiques agrégées sur le contrôle fiscal : nombre de vérifications,
recettes, taux de couverture du portefeuille. Utiles pour dimensionner le
« budget de contrôle » du §1 du document 03.

---

## 6. Outils

| Outil | Rôle | Référence |
|---|---|---|
| **scikit-learn** | Métriques, prétraitement | Pedregosa et al. (2011), JMLR 12, 2825-2830 |
| **XGBoost** | Modèle supervisé final | Chen & Guestrin (2016) |
| **Optuna** | Optimisation d'hyperparamètres | Akiba et al. (2019) |
| **SHAP** | Interprétabilité | Lundberg & Lee (2017) |
| **PyOD** | Détection d'anomalies non supervisée | Zhao, Nasrullah & Li (2019), JMLR 20(96) |
| **pandas** | Manipulation de données | McKinney (2010), Proc. 9th Python in Science Conf. |
| **Streamlit** | Interface de restitution | <https://streamlit.io> |

---

## 7. Note sur la citation du jeu de données

Le jeu JAR2020 doit être cité comme **source de données**, distinctement de
l'article qui l'accompagne :

> Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020). *Data for: Detecting
> Accounting Fraud in Publicly Traded U.S. Firms Using a Machine Learning
> Approach* [jeu de données]. <https://github.com/JarFraud/FraudDetection>

Les données comptables sous-jacentes proviennent de **Compustat North America**
(S&P Global Market Intelligence), les étiquettes de fraude des **AAER** de la
*Securities and Exchange Commission*.
