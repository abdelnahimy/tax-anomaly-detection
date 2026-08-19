"""Modèles non supervisés : Isolation Forest, Local Outlier Factor, Autoencoder.

Chaque `fit_predict_*` a la signature attendue par
`src.models.common.run_temporal_cv` : ``(df_train, df_test, features) -> scores``,
où un score plus élevé signifie "plus anormal". Les hyperparamètres fixes
(nombre d'arbres, voisins, contamination...) sont exposés en paramètres de la
factory `make_*` pour rester ajustables depuis le notebook sans dupliquer la
boucle d'entraînement.
"""
from typing import Sequence

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ..config import CONTAMINATION, RANDOM_STATE


def _scale(df_train: pd.DataFrame, df_test: pd.DataFrame, features: Sequence[str]):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features].values)
    X_test = scaler.transform(df_test[features].values)
    return X_train, X_test


def make_isolation_forest_fit_predict(
    n_estimators: int = 300, contamination: float = CONTAMINATION, random_state: int = RANDOM_STATE
):
    """Isolation Forest standardisé. Le score `decision_function` est inversé
    (`-score`) pour que "plus grand = plus anormal", cohérent avec LOF/autoencoder."""

    def fit_predict(df_train, df_test, features):
        X_train, X_test = _scale(df_train, df_test, features)
        model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            random_state=random_state, n_jobs=-1,
        )
        model.fit(X_train)
        return -model.decision_function(X_test)

    return fit_predict


def make_lof_fit_predict(n_neighbors: int = 35, contamination: float = CONTAMINATION):
    """LOF en mode `novelty=True` (nécessaire pour scorer un `X_test` distinct
    du train, puisqu'on respecte le split temporel plutôt qu'un fit unique)."""

    def fit_predict(df_train, df_test, features):
        X_train, X_test = _scale(df_train, df_test, features)
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination,
            novelty=True, n_jobs=-1,
        )
        model.fit(X_train)
        return -model.decision_function(X_test)

    return fit_predict


def make_autoencoder_fit_predict(
    contamination: float = CONTAMINATION, epoch_num: int = 30,
    batch_size: int = 64, random_state: int = RANDOM_STATE,
):
    """Autoencoder PyOD : le score d'anomalie est l'erreur de reconstruction.
    Lève `ImportError` avec un message clair si PyOD n'est pas installé,
    plutôt que d'échouer sur un `NameError` obscur plus loin dans le notebook.
    """
    try:
        from pyod.models.auto_encoder import AutoEncoder
    except ImportError as e:
        raise ImportError(
            "PyOD n'est pas installé. Installez-le avec `pip install pyod` "
            "(inclus dans requirements.txt) pour utiliser l'autoencoder."
        ) from e

    def fit_predict(df_train, df_test, features):
        X_train, X_test = _scale(df_train, df_test, features)
        n_features = X_train.shape[1]
        hidden = [max(4, n_features // 2), max(2, n_features // 4), max(4, n_features // 2)]

        model = AutoEncoder(
            hidden_neuron_list=hidden, contamination=contamination,
            epoch_num=epoch_num, batch_size=batch_size,
            random_state=random_state, verbose=0,
        )
        model.fit(X_train)
        return model.decision_function(X_test)

    return fit_predict
