"""Boucle de validation temporelle partagée par tous les modèles.

Avant ce refactor, la boucle `for year in TEST_YEARS: split -> fit -> predict
-> evaluate` était réécrite (avec de petites variations) dans chaque fonction
`run_isolation_forest`, `run_lof`, `run_autoencoder`, `run_random_forest`,
`run_xgboost`, `run_rusboost`, `run_xgboost_smote`, `run_ensemble` des
notebooks 03, 04 et 05. `run_temporal_cv` factorise cette boucle une seule
fois ; chaque modèle n'a plus qu'à fournir une fonction
``fit_predict(df_train, df_test, features) -> np.ndarray`` qui l'entraîne et
retourne les scores d'anomalie/probabilités sur le test.
"""
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..feature_engineering import temporal_split
from ..scoring import evaluate_scores


def run_temporal_cv(
    df: pd.DataFrame,
    features: Sequence[str],
    test_years: Sequence[int],
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame, Sequence[str]], np.ndarray],
    min_train_fraud: int = 1,
    label: str = "",
    verbose_errors: bool = True,
    **split_kwargs,
) -> pd.DataFrame:
    """Exécute `fit_predict` sur chaque split temporel de `test_years` et agrège
    les métriques dans un DataFrame indexé par année.

    Parameters
    ----------
    fit_predict : callable(df_train, df_test, features) -> scores
        Doit entraîner un modèle sur `df_train` et retourner un score
        d'anomalie/probabilité de fraude pour chaque ligne de `df_test`
        (plus le score est élevé, plus l'observation est jugée suspecte).
    min_train_fraud : int
        Nombre minimum de fraudes dans le train pour tenter l'année (certains
        modèles, ex. SMOTE/RUSBoost, ont besoin d'un minimum de positifs).
    """
    results = {}
    for year in test_years:
        df_train, df_test = temporal_split(df, year_test=year, **split_kwargs)
        if df_test.empty or df_train.empty or df_train["misstate"].sum() < min_train_fraud:
            continue

        y_test = df_test["misstate"].values
        try:
            scores = fit_predict(df_train, df_test, features)
        except ValueError as e:
            if verbose_errors:
                print(f"[{label or fit_predict.__name__}] Année {year} ignorée : {e}")
            continue

        results[year] = evaluate_scores(y_test, scores)

    return pd.DataFrame(results).T
