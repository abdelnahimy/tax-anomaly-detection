"""Package source du projet de détection d'anomalies fiscales.

Regroupe le code réutilisé par les notebooks et le dashboard (chemins de projet,
prétraitement, ingénierie de variables, scoring, inférence, visualisation) afin
d'éviter la duplication de fonctions identiques d'un notebook à l'autre.

Modules
-------
config
    Chemins du projet, listes de variables, constantes du protocole de
    validation temporelle.
preprocessing
    Log signé, indicateurs de valeurs manquantes et imputation.
feature_engineering
    Split temporel avec anti-fuite, variables de trajectoire (`_d1` / `_d2`),
    ratios fiscaux métier.
models
    Entraînement supervisé et non supervisé, boucle de validation temporelle.
scoring
    Métriques adaptées au fort déséquilibre de classes (AUC-PR, rappel@k).
inference
    Pont entre les modèles sauvegardés et leurs utilisateurs : reconstruit le
    vecteur de variables dans le bon ordre, score, et explique via SHAP.
liasse_mapping
    Passerelle entre la liasse fiscale marocaine (CGNC) et les variables
    Compustat attendues par le modèle. C'est la pièce qui rend le dispositif
    utilisable dans le contexte de la Direction Régionale des Impôts.
viz
    Fonctions de tracé partagées.

Point d'attention transversal
-----------------------------
Les modèles ont été entraînés sur des tableaux NumPy et ne portent pas les noms
de colonnes en interne : **l'ordre des variables est la seule chose qui relie
une valeur à la bonne feature**. Tout passe donc par `models/model_meta_v2.json`,
qui fait foi. Voir `inference.prepare_features` et `tests/test_inference.py`.
"""
