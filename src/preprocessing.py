"""Chargement et transformation de base des données comptables brutes."""
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from .config import DATA_RAW


def load_raw_data(path: Path = DATA_RAW) -> pd.DataFrame:
    """Charge le CSV brut JAR2020 et vérifie qu'il n'est pas vide.

    Lève une erreur explicite si le fichier est absent plutôt que de laisser
    ``pandas`` échouer avec un ``FileNotFoundError`` peu informatif — utile
    car le CSV (47 Mo) n'est volontairement pas versionné dans le dépôt
    (cf. README, section Dataset).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de données introuvable : {path}\n"
            "Téléchargez data_FraudDetection_JAR2020.csv depuis "
            "github.com/JarFraud/FraudDetection et placez-le dans data/raw/."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Le fichier {path} a été chargé mais est vide.")
    return df


def signed_log1p(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Transformation log signée : ``sign(x) * log1p(|x|)``.

    Réduit l'asymétrie des grandes valeurs comptables tout en préservant le
    signe (nécessaire pour des variables pouvant être négatives comme ``ib``,
    ``ni``, ``re``, ``ceq``), contrairement à un simple ``log1p``.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def add_log_features(df: pd.DataFrame, cols: Iterable[str], suffix: str = "_log") -> pd.DataFrame:
    """Ajoute une colonne ``<col><suffix>`` = signed_log1p(col) pour chaque `col`."""
    df = df.copy()
    for col in cols:
        df[col + suffix] = signed_log1p(df[col])
    return df


def add_missing_indicators_and_impute(
    df: pd.DataFrame, cols: Iterable[str], fill_value: float = 0
) -> pd.DataFrame:
    """Impute les NaN par `fill_value` et ajoute un indicateur binaire par colonne.

    Les ratios ``ch_*``/``dch_*`` du dataset JAR2020 sont des variations d'une
    année sur l'autre : ils sont structurellement absents pour la première
    année d'observation de chaque entreprise. L'indicateur ``<col>_was_missing``
    conserve cette information plutôt que de la faire disparaître silencieusement
    dans l'imputation — utile pour les modèles qui n'acceptent pas les NaN
    (régression logistique, Isolation Forest, autoencoder).
    """
    df = df.copy()
    for col in cols:
        df[col + "_was_missing"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(fill_value)
    return df


def check_duplicates(df: pd.DataFrame, subset: List[str] = ("gvkey", "fyear")) -> pd.DataFrame:
    """Retourne les lignes dupliquées sur `subset` (attendu : aucune, un couple
    (entreprise, année fiscale) doit être unique). Utilisé comme test de qualité
    des données en EDA."""
    subset = list(subset)
    return df[df.duplicated(subset=subset, keep=False)].sort_values(subset)
