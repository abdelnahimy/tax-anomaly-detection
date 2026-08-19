"""Modèles supervisés : Random Forest, XGBoost, RUSBoost, XGBoost+SMOTE, ensemble.

Mêmes conventions que `unsupervised.py` : chaque `make_*_fit_predict` retourne
une fonction `(df_train, df_test, features) -> scores` compatible avec
`src.models.common.run_temporal_cv`.
"""
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from ..config import RANDOM_STATE

DEFAULT_XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
)


def make_random_forest_fit_predict(
    n_estimators: int = 500, max_depth: int = 8, random_state: int = RANDOM_STATE
):
    """Random Forest avec `class_weight='balanced'` : repondère automatiquement
    la classe minoritaire, sans sur/sous-échantillonnage explicite. Tolère les
    NaN structurels des features temporelles/ratios mal — on impute par la
    médiane du train si besoin.
    """

    def fit_predict(df_train, df_test, features):
        X_train = df_train[features].values
        y_train = df_train["misstate"].values
        X_test = df_test[features].values

        if np.isnan(X_train).any() or np.isnan(X_test).any():
            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]

    return fit_predict


def make_xgboost_fit_predict(
    spw_override: Optional[float] = None,
    params_override: Optional[dict] = None,
    random_state: int = RANDOM_STATE,
):
    """XGBoost avec `scale_pos_weight` dynamique (recalculé sur le train de
    chaque année, sauf si `spw_override` est fourni) et hyperparamètres
    configurables. Gère nativement les NaN — pas d'imputation nécessaire,
    contrairement à Random Forest.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "xgboost n'est pas installé. Installez-le avec `pip install xgboost` "
            "(inclus dans requirements.txt)."
        ) from e

    def fit_predict(df_train, df_test, features):
        X_train = df_train[features].values
        y_train = df_train["misstate"].values
        X_test = df_test[features].values

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = spw_override if spw_override is not None else n_neg / n_pos

        params = dict(DEFAULT_XGB_PARAMS)
        if params_override:
            params.update(params_override)

        model = XGBClassifier(
            **params, scale_pos_weight=spw, eval_metric="aucpr",
            random_state=random_state, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]

    return fit_predict


def make_rusboost_fit_predict(
    n_estimators: int = 300, learning_rate: float = 0.1,
    max_depth: int = 2, random_state: int = RANDOM_STATE,
):
    """RUSBoost (Random Under-Sampling + AdaBoost) : la méthodologie originale
    de Bao et al. (2020), via `imbalanced-learn`. Certaines années peuvent
    échouer (`BaseClassifier ... worse than random`) quand le sous-échantillon
    tiré est trop dégénéré pour AdaBoost — l'appelant (`run_temporal_cv`)
    capture ce `ValueError` et passe à l'année suivante plutôt que d'arrêter
    toute la validation.
    """
    try:
        from imblearn.ensemble import RUSBoostClassifier
    except ImportError as e:
        raise ImportError(
            "imbalanced-learn n'est pas installé. Installez-le avec "
            "`pip install imbalanced-learn` (inclus dans requirements.txt)."
        ) from e
    from sklearn.tree import DecisionTreeClassifier
    import inspect

    # `algorithm='SAMME'` est nécessaire avec sklearn >= 1.4 (SAMME.R a été retiré),
    # mais imbalanced-learn >= 0.12 déprécie ce paramètre et prévoit de le supprimer
    # en 0.14 (SAMME devient alors le seul comportement). On ne le passe que si le
    # constructeur l'accepte encore, pour rester compatible avec les deux générations
    # de la librairie sans changer ce module à chaque mise à jour.
    _rusboost_params = inspect.signature(RUSBoostClassifier.__init__).parameters
    _extra_kwargs = {"algorithm": "SAMME"} if "algorithm" in _rusboost_params else {}

    def fit_predict(df_train, df_test, features):
        X_train = df_train[features].values
        y_train = df_train["misstate"].values
        X_test = df_test[features].values

        model = RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=max_depth),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            sampling_strategy="auto",
            random_state=random_state,
            **_extra_kwargs,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1]

    return fit_predict


def make_xgboost_smote_fit_predict(random_state: int = RANDOM_STATE):
    """XGBoost précédé d'un sur-échantillonnage SMOTE de la classe minoritaire,
    appliqué **uniquement sur le train** de chaque split (jamais sur le test,
    pour ne pas fuiter d'observations synthétiques dans l'évaluation).
    """
    try:
        from imblearn.over_sampling import SMOTE
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "xgboost et imbalanced-learn sont requis pour XGBoost+SMOTE "
            "(inclus dans requirements.txt)."
        ) from e

    def fit_predict(df_train, df_test, features):
        X_train = df_train[features].values
        y_train = df_train["misstate"].values
        X_test = df_test[features].values

        n_pos = int(y_train.sum())
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, n_pos - 1))
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(
            **DEFAULT_XGB_PARAMS, eval_metric="aucpr",
            random_state=random_state, n_jobs=-1,
        )
        model.fit(X_train_res, y_train_res)
        return model.predict_proba(X_test)[:, 1]

    return fit_predict


def make_ensemble_fit_predict(
    w_rf: float = 0.3, w_xgb: float = 0.7,
    xgb_spw: Optional[float] = None, xgb_params: Optional[dict] = None,
    random_state: int = RANDOM_STATE,
):
    """Moyenne pondérée des scores Random Forest et XGBoost :
    ``score = w_rf * score_RF + w_xgb * score_XGB``.
    """
    rf_fit_predict = make_random_forest_fit_predict(random_state=random_state)
    xgb_fit_predict = make_xgboost_fit_predict(
        spw_override=xgb_spw, params_override=xgb_params, random_state=random_state
    )

    def fit_predict(df_train, df_test, features):
        scores_rf = rf_fit_predict(df_train, df_test, features)
        scores_xgb = xgb_fit_predict(df_train, df_test, features)
        return w_rf * scores_rf + w_xgb * scores_xgb

    return fit_predict
