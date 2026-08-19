"""Split temporel et construction des jeux de features.

Toutes les fonctions ici sont pures (elles ne modifient pas leurs arguments
en place, sauf mention contraire) et sans état caché : les notebooks 02 à 05
les important au lieu d'en garder cinq copies légèrement divergentes, comme
c'était le cas avant ce refactor (`temporal_split` était redéfinie à
l'identique dans 02, 03, 04 et 05).
"""
from typing import List, Sequence, Tuple

import pandas as pd

from .config import RAW_ACCOUNTING_VARS, RATIO_VARS, SPLIT_GAP_YEARS, DATA_YEAR_START


def temporal_split(
    df: pd.DataFrame,
    year_test: int,
    gap: int = SPLIT_GAP_YEARS,
    year_start: int = DATA_YEAR_START,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporel avec gap, reproduisant la méthodologie de Bao et al. (2020).

    - train : années ``[year_start, year_test - gap]``
    - test  : année ``year_test`` uniquement
    - anti-fuite : les observations du train partageant un ``p_aaer`` avec une
      fraude du test sont retirées du train (même affaire de fraude en série
      chevauchant plusieurs années).

    Returns
    -------
    (df_train, df_test) : copies indépendantes (``.copy()``), sûres à modifier.
    """
    train_end = year_test - gap

    df_train = df[(df["fyear"] >= year_start) & (df["fyear"] <= train_end)].copy()
    df_test = df[df["fyear"] == year_test].copy()

    paaer_test = df_test.loc[df_test["misstate"] != 0, "p_aaer"].dropna().unique()
    df_train = df_train[~df_train["p_aaer"].isin(paaer_test)]

    return df_train, df_test


def split_summary(df: pd.DataFrame, year_test: int, **kwargs) -> dict:
    """Comme `temporal_split`, mais retourne un résumé chiffré au lieu des DataFrames
    (pratique pour construire un tableau récapitulatif sur plusieurs années sans
    garder toutes les données train/test en mémoire simultanément)."""
    train_end = year_test - kwargs.get("gap", SPLIT_GAP_YEARS)
    year_start = kwargs.get("year_start", DATA_YEAR_START)

    df_train_before = df[(df["fyear"] >= year_start) & (df["fyear"] <= train_end)]
    df_train, df_test = temporal_split(df, year_test, **kwargs)

    return {
        "year_test": year_test,
        "train_range": f"{year_start}-{train_end}",
        "train_obs": len(df_train),
        "train_fraud": int(df_train["misstate"].sum()),
        "n_removed_anti_leak": len(df_train_before) - len(df_train),
        "test_obs": len(df_test),
        "test_fraud": int(df_test["misstate"].sum()),
    }


def build_baseline_feature_lists(
    raw_accounting_vars: Sequence[str] = RAW_ACCOUNTING_VARS,
    ratio_vars: Sequence[str] = RATIO_VARS,
    missing_indicator_cols: Sequence[str] | None = None,
) -> dict:
    """Construit les jeux de variables ``FEATURES_BASELINE(_LOG)`` /
    ``FEATURES_EXTENDED(_LOG)`` à partir des listes de variables du projet.

    ``missing_indicator_cols`` doit être fourni une fois les indicateurs
    ``_was_missing`` créés (cf. `preprocessing.add_missing_indicators_and_impute`) ;
    si omis, on suppose la convention ``<ratio>_was_missing``.
    """
    if missing_indicator_cols is None:
        missing_indicator_cols = [c + "_was_missing" for c in ratio_vars]

    features_baseline = list(raw_accounting_vars)
    features_extended = list(raw_accounting_vars) + list(ratio_vars)
    features_baseline_log = [c + "_log" for c in raw_accounting_vars]
    features_extended_log = features_baseline_log + list(ratio_vars) + list(missing_indicator_cols)

    return {
        "FEATURES_BASELINE": features_baseline,
        "FEATURES_EXTENDED": features_extended,
        "FEATURES_BASELINE_LOG": features_baseline_log,
        "FEATURES_EXTENDED_LOG": features_extended_log,
        "missing_indicator_cols": missing_indicator_cols,
    }


# Les 10 variables comptables (déjà en log signé) jugées les plus discriminantes
# en EDA (01_EDA.ipynb, section 6) et utilisées pour les features de trajectoire.
TEMPORAL_VARS = [
    "txp_log", "txt_log", "sale_log", "ni_log", "at_log",
    "ib_log", "rect_log", "invt_log", "ceq_log", "dp_log",
]


def add_temporal_features(df: pd.DataFrame, vars_to_lag: Sequence[str] = TEMPORAL_VARS) -> Tuple[pd.DataFrame, List[str]]:
    """Ajoute les variations sur 1 an (``_d1``) et 2 ans (``_d2``) de `vars_to_lag`,
    calculées par entreprise (`gvkey`), triées par année fiscale.

    NaN pour les 1 (``_d1``) ou 2 (``_d2``) premières années d'observation de
    chaque entreprise — c'est structurel, pas une erreur, et XGBoost gère
    nativement ces NaN.
    """
    df = df.sort_values(["gvkey", "fyear"]).copy()
    new_cols = []
    for var in vars_to_lag:
        if var not in df.columns:
            continue
        col_d1 = var + "_d1"
        df[col_d1] = df.groupby("gvkey")[var].diff(1)
        new_cols.append(col_d1)

        col_d2 = var + "_d2"
        df[col_d2] = df.groupby("gvkey")[var].diff(2)
        new_cols.append(col_d2)
    return df, new_cols


FISCAL_FEATURE_NAMES = [
    "tax_burden", "effective_tax_rate", "tax_to_sales",
    "accruals", "asset_growth", "receivables_ratio",
]


def add_fiscal_features(df: pd.DataFrame, eps: float = 1e-6) -> Tuple[pd.DataFrame, List[str]]:
    """Ajoute 6 ratios fiscaux/comptables métier, construits sur les variables
    brutes (pas les versions log) pour rester interprétables :

    - ``tax_burden``          = txp / |txt|            — charge fiscale courante vs totale
    - ``effective_tax_rate``  = txt / |ib + txt|        — taux effectif d'imposition
    - ``tax_to_sales``        = txt / |sale|            — charge fiscale / chiffre d'affaires
    - ``accruals``            = (ni - dp) / |at|        — accruals totaux (proxy Jones model)
    - ``asset_growth``        = at(t)/at(t-1) - 1       — croissance des actifs, par gvkey
    - ``receivables_ratio``   = rect / |sale|           — créances / ventes (manipulation revenus)

    Chaque ratio est "clippé" pour limiter l'effet des valeurs extrêmes issues
    de dénominateurs proches de zéro.
    """
    df = df.copy()

    if {"txp", "txt"}.issubset(df.columns):
        df["tax_burden"] = (df["txp"] / (df["txt"].abs() + eps)).clip(-10, 10)

    if {"txt", "ib"}.issubset(df.columns):
        pretax = df["ib"] + df["txt"]
        df["effective_tax_rate"] = (df["txt"] / (pretax.abs() + eps)).clip(-5, 5)

    if {"txt", "sale"}.issubset(df.columns):
        df["tax_to_sales"] = (df["txt"] / (df["sale"].abs() + eps)).clip(-5, 5)

    if {"ni", "dp", "at"}.issubset(df.columns):
        df["accruals"] = ((df["ni"] - df["dp"]) / (df["at"].abs() + eps)).clip(-5, 5)

    if "at" in df.columns:
        df = df.sort_values(["gvkey", "fyear"])
        at_lag = df.groupby("gvkey")["at"].shift(1)
        df["asset_growth"] = ((df["at"] - at_lag) / (at_lag.abs() + eps)).clip(-5, 5)

    if {"rect", "sale"}.issubset(df.columns):
        df["receivables_ratio"] = (df["rect"] / (df["sale"].abs() + eps)).clip(0, 5)

    fiscal_features = [f for f in FISCAL_FEATURE_NAMES if f in df.columns]
    return df, fiscal_features
