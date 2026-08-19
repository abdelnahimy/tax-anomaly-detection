"""Pipeline d'inférence : des données comptables brutes vers un score de risque.

Ce module est le pont entre les modèles entraînés (`models/*.pkl`) et toute
application qui veut les utiliser (dashboard Streamlit, API FastAPI, script
batch). Il reconstruit exactement le vecteur de features attendu par le modèle
sauvegardé, dans le bon ordre.

⚠️ Point critique : les modèles ont été entraînés sur des tableaux NumPy
(`df[features].values`) et ne portent donc pas les noms de colonnes en interne
(`booster.feature_names is None`). **L'ordre des colonnes est donc la seule
chose qui relie une valeur à la bonne feature.** Si on construisait la matrice
dans un ordre différent, le modèle produirait des scores silencieusement faux
sans lever la moindre erreur. C'est pourquoi tout passe ici par
`meta["features"]`, qui fait foi, et par `assert` de contrôle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, RAW_ACCOUNTING_VARS, RATIO_VARS
from .preprocessing import add_log_features, add_missing_indicators_and_impute
from .feature_engineering import add_temporal_features, add_fiscal_features

# Colonnes minimales qu'un fichier d'entrée doit contenir pour être scorable.
# `gvkey` et `fyear` sont indispensables : sans eux, impossible de calculer les
# features de trajectoire (variations d'une année sur l'autre par entreprise).
REQUIRED_ID_COLS = ["gvkey", "fyear"]
REQUIRED_INPUT_COLS = REQUIRED_ID_COLS + list(RAW_ACCOUNTING_VARS)

MODEL_V2 = "xgboost_v2_best.pkl"
MODEL_V1 = "xgboost_fraud_detection.pkl"
META_V2 = "model_meta_v2.json"


class InferenceError(RuntimeError):
    """Erreur d'inférence explicite (modèle absent, colonnes manquantes, ...)."""


# --------------------------------------------------------------------------
# Chargement du modèle
# --------------------------------------------------------------------------

def load_model(models_dir: Path = MODELS_DIR, model_file: str = MODEL_V2,
               meta_file: str = META_V2) -> tuple:
    """Charge le modèle final et ses métadonnées (liste ordonnée des features).

    Returns
    -------
    (model, meta) : le classifieur XGBoost et le dict de métadonnées.
    """
    models_dir = Path(models_dir)
    model_path = models_dir / model_file
    meta_path = models_dir / meta_file

    if not model_path.exists():
        raise InferenceError(
            f"Modèle introuvable : {model_path}\n"
            "Lancez 05_modeling_supervised_v2.ipynb pour l'entraîner et le sauvegarder."
        )
    if not meta_path.exists():
        raise InferenceError(
            f"Métadonnées introuvables : {meta_path}\n"
            "Elles contiennent la liste ORDONNÉE des features et sont indispensables "
            "pour scorer correctement (le modèle ne stocke pas les noms de colonnes)."
        )

    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())

    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is not None and n_expected != len(meta["features"]):
        raise InferenceError(
            f"Incohérence modèle/métadonnées : le modèle attend {n_expected} features "
            f"mais model_meta_v2.json en décrit {len(meta['features'])}. "
            "Réentraînez le modèle et régénérez les métadonnées ensemble."
        )
    return model, meta


# --------------------------------------------------------------------------
# Validation des données d'entrée
# --------------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des problèmes bloquants d'un fichier d'entrée.

    Une liste vide signifie que le fichier est scorable.
    """
    problems = []

    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        problems.append(
            f"{len(missing)} colonne(s) requise(s) manquante(s) : {', '.join(missing[:12])}"
            + (" ..." if len(missing) > 12 else "")
        )

    if df.empty:
        problems.append("Le fichier ne contient aucune ligne.")

    for col in REQUIRED_ID_COLS:
        if col in df.columns and df[col].isnull().any():
            problems.append(f"La colonne identifiant '{col}' contient des valeurs vides.")

    present_numeric = [c for c in RAW_ACCOUNTING_VARS if c in df.columns]
    if present_numeric:
        non_numeric = [
            c for c in present_numeric
            if not pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
        ]
        if non_numeric:
            problems.append(
                f"Colonnes non numériques : {', '.join(non_numeric[:8])}"
            )

    return problems


def temporal_coverage(df_features: pd.DataFrame, temporal_features: Sequence[str]) -> dict:
    """Mesure la part des features de trajectoire réellement disponibles.

    Les features `_d1` / `_d2` comparent une entreprise à elle-même 1 ou 2 ans
    plus tôt. Si le fichier d'entrée ne contient qu'une seule année, elles sont
    toutes NaN : le modèle fonctionne quand même (XGBoost gère les NaN
    nativement) mais il score alors sur le niveau absolu uniquement, sans la
    dynamique — ce qui dégrade la qualité de détection. Cette fonction permet
    de le signaler à l'utilisateur au lieu de le laisser invisible.
    """
    present = [c for c in temporal_features if c in df_features.columns]
    if not present:
        return {"available_pct": 0.0, "n_temporal_features": 0, "rows_with_history": 0}

    non_nan_ratio = df_features[present].notna().mean().mean()
    rows_with_history = int(df_features[present].notna().any(axis=1).sum())
    return {
        "available_pct": float(non_nan_ratio * 100),
        "n_temporal_features": len(present),
        "rows_with_history": rows_with_history,
    }


# --------------------------------------------------------------------------
# Construction des features
# --------------------------------------------------------------------------

def prepare_features(df_raw: pd.DataFrame, feature_list: Sequence[str]) -> pd.DataFrame:
    """Reconstruit le tableau de features attendu par le modèle.

    Applique, dans l'ordre, la même chaîne que les notebooks 02 et 05 :
    1. log signé sur les 28 variables comptables brutes
    2. indicateurs de NaN + imputation sur les ratios (si présents)
    3. features temporelles `_d1` / `_d2` par `gvkey`
    4. ratios fiscaux métier

    Puis sélectionne `feature_list` **dans l'ordre exact fourni**, en créant
    au besoin les colonnes absentes (remplies de NaN, gérées nativement par
    XGBoost) plutôt que d'échouer.
    """
    df = df_raw.copy()

    # ⚠️ `add_temporal_features` et `add_fiscal_features` trient par (gvkey, fyear)
    # pour calculer les variations par entreprise. Sans précaution, le tableau
    # renvoyé serait donc dans un ORDRE DIFFÉRENT de l'entrée, et l'appelant
    # attribuerait les scores aux mauvaises lignes (vérifié : 16 418 lignes sur
    # 16 423 recevaient le score d'une autre entreprise). On marque donc l'ordre
    # d'origine avant toute transformation pour le restaurer à la fin.
    # Un compteur positionnel est utilisé plutôt que l'index pandas, pour rester
    # correct même si l'entrée a un index dupliqué ou non trivial.
    order_col = "__input_row_order__"
    df[order_col] = np.arange(len(df))

    # Les colonnes comptables doivent être numériques (un CSV peut les livrer en texte)
    for col in RAW_ACCOUNTING_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    present_raw = [c for c in RAW_ACCOUNTING_VARS if c in df.columns]
    if present_raw:
        df = add_log_features(df, present_raw)

    present_ratios = [c for c in RATIO_VARS if c in df.columns]
    if present_ratios:
        df = add_missing_indicators_and_impute(df, present_ratios)

    if {"gvkey", "fyear"}.issubset(df.columns):
        df, _ = add_temporal_features(df)

    df, _ = add_fiscal_features(df)

    # Restauration de l'ordre des lignes d'origine (cf. commentaire plus haut)
    df = df.sort_values(order_col)

    # Sélection ordonnée des COLONNES — c'est ici que se joue la correction des scores.
    missing_features = [f for f in feature_list if f not in df.columns]
    for f in missing_features:
        df[f] = np.nan

    X = df.reindex(columns=list(feature_list))
    X.index = df_raw.index  # aligne sur l'index d'entrée pour un usage naturel côté appelant

    assert list(X.columns) == list(feature_list), (
        "L'ordre des colonnes ne correspond pas à celui attendu par le modèle."
    )
    assert len(X) == len(df_raw), "Le nombre de lignes a changé pendant la préparation des features."
    return X


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_dataframe(
    df_raw: pd.DataFrame,
    model=None,
    meta: dict | None = None,
    models_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """Score un tableau de données brutes et retourne les résultats classés.

    Returns
    -------
    DataFrame trié par risque décroissant, avec les colonnes :
    `risk_score` (probabilité brute du modèle), `risk_rank` (1 = plus suspect),
    `risk_percentile` (position dans le lot, 100 = plus suspect), et les
    identifiants d'origine.
    """
    if model is None or meta is None:
        model, meta = load_model(models_dir)

    problems = validate_input(df_raw)
    if problems:
        raise InferenceError("Données d'entrée invalides :\n- " + "\n- ".join(problems))

    features = meta["features"]
    X = prepare_features(df_raw, features)

    scores = model.predict_proba(X.values)[:, 1]

    out = df_raw.copy()
    out["risk_score"] = scores
    out["risk_rank"] = (-out["risk_score"]).rank(method="first").astype(int)
    # pct=True donne 0-1 croissant ; on veut 100 = le plus suspect
    out["risk_percentile"] = out["risk_score"].rank(pct=True) * 100

    return out.sort_values("risk_score", ascending=False)


def score_with_diagnostics(df_raw: pd.DataFrame, models_dir: Path = MODELS_DIR) -> dict:
    """Comme `score_dataframe`, mais retourne aussi des diagnostics utiles à
    afficher dans une interface (couverture temporelle, nb de features
    reconstituées, seuils top-k)."""
    model, meta = load_model(models_dir)
    X = prepare_features(df_raw, meta["features"])
    coverage = temporal_coverage(X, meta.get("temporal_features", []))
    scored = score_dataframe(df_raw, model=model, meta=meta)

    n = len(scored)
    return {
        "scored": scored,
        "coverage": coverage,
        "n_rows": n,
        "n_features": len(meta["features"]),
        "top_1pct_threshold": float(scored["risk_score"].quantile(0.99)) if n else np.nan,
        "top_5pct_threshold": float(scored["risk_score"].quantile(0.95)) if n else np.nan,
    }


# --------------------------------------------------------------------------
# Interprétabilité
# --------------------------------------------------------------------------

def explain_row(
    df_raw: pd.DataFrame,
    row_index: int,
    model=None,
    meta: dict | None = None,
    models_dir: Path = MODELS_DIR,
    top_n: int = 12,
) -> pd.DataFrame:
    """Retourne les `top_n` features qui poussent le plus le score d'une ligne
    vers le haut (suspect) ou vers le bas (normal), via SHAP.

    Utilisé par le dashboard pour répondre à « pourquoi cette entreprise est-elle
    signalée ? » — indispensable dans un contexte d'audit, où une alerte non
    justifiable n'est pas actionnable.
    """
    try:
        import shap
    except ImportError as e:
        raise InferenceError(
            "SHAP n'est pas installé. `pip install shap` (inclus dans requirements.txt)."
        ) from e

    if model is None or meta is None:
        model, meta = load_model(models_dir)

    features = meta["features"]
    X = prepare_features(df_raw, features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values)

    row = np.asarray(shap_values)[row_index]
    contrib = pd.DataFrame({
        "feature": features,
        "valeur": X.iloc[row_index].values,
        "contribution_shap": row,
    })
    contrib["abs"] = contrib["contribution_shap"].abs()
    contrib = contrib.sort_values("abs", ascending=False).head(top_n).drop(columns="abs")
    contrib["effet"] = np.where(
        contrib["contribution_shap"] > 0, "augmente le risque", "diminue le risque"
    )
    return contrib.reset_index(drop=True)
