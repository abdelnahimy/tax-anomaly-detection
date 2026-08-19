"""Métriques d'évaluation adaptées au déséquilibre extrême de classes (~150:1).

Le seuil 0.5 est inutilisable sur ce problème (cf. README, "Métriques
d'évaluation") : les scores ne sont pas calibrés, et la classe positive est
trop rare pour qu'un seuil fixe ait un sens. On évalue donc par le **rang**
des scores (precision/recall@k%) en plus des métriques globales AUC-ROC /
AUC-PR.
"""
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_K_PCTS = (0.01, 0.05, 0.10)
DEFAULT_METRICS = ["auc_roc", "auc_pr", "precision@1%", "recall@1%", "precision@5%", "recall@5%"]


def precision_recall_at_k(y_true: Iterable, scores: Iterable, k_pct: float) -> tuple[float, float]:
    """Precision et recall parmi le top k% des scores les plus élevés.

    C'est la métrique la plus proche d'un cas d'usage réel d'audit : "si
    l'équipe ne peut investiguer que les k% dossiers les plus suspects,
    quelle fraction de fraudes réelles capte-t-on, et quelle fraction des
    alertes est-elle réellement fondée ?"
    """
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    n = len(scores)
    k = max(1, int(np.ceil(n * k_pct)))
    order = np.argsort(-scores)
    top_k_idx = order[:k]

    n_fraud_total = y_true.sum()
    n_fraud_in_top_k = y_true[top_k_idx].sum()

    precision = n_fraud_in_top_k / k
    recall = n_fraud_in_top_k / n_fraud_total if n_fraud_total > 0 else np.nan
    return precision, recall


def evaluate_scores(y_true: Iterable, scores: Iterable, k_pcts: Sequence[float] = DEFAULT_K_PCTS) -> dict:
    """Calcule AUC-ROC, AUC-PR, et precision/recall@k pour un vecteur de scores
    sur une seule année de test."""
    y_true = np.asarray(y_true)
    result = {}
    if 0 < y_true.sum() < len(y_true):
        result["auc_roc"] = roc_auc_score(y_true, scores)
        result["auc_pr"] = average_precision_score(y_true, scores)
    else:
        # une seule classe présente (arrive sur de très petits jeux de test) :
        # AUC non définie plutôt que de planter ou renvoyer une valeur trompeuse
        result["auc_roc"] = np.nan
        result["auc_pr"] = np.nan

    for k in k_pcts:
        p, r = precision_recall_at_k(y_true, scores, k)
        result[f"precision@{int(k * 100)}%"] = p
        result[f"recall@{int(k * 100)}%"] = r

    result["n_test"] = len(y_true)
    result["n_fraud_test"] = int(y_true.sum())
    return result


def weighted_mean(results_df: pd.DataFrame, col: str, weight_col: str = "n_test",
                   exclude_years: Sequence[int] = ()) -> float:
    """Moyenne de `col` pondérée par `weight_col` (nombre d'observations de test
    de chaque année), en excluant optionnellement certaines années (ex. 2011,
    cf. le biais documenté dans 05_modeling_supervised_v2.ipynb).
    """
    subset = results_df.drop(index=[y for y in exclude_years if y in results_df.index])
    valid = subset[[col, weight_col]].dropna()
    if valid.empty:
        return np.nan
    return np.average(valid[col], weights=valid[weight_col])


def summary(results_df: pd.DataFrame, name: str = "", metrics: Sequence[str] = DEFAULT_METRICS,
            exclude_years: Sequence[int] = (), verbose: bool = True) -> pd.Series:
    """Moyennes pondérées des métriques clés sur toutes les années de test.

    `exclude_years` permet de recalculer un résumé hors d'une année atypique
    sans dupliquer la boucle de calcul.
    """
    row = {m: weighted_mean(results_df, m, exclude_years=exclude_years) for m in metrics}
    label = name + (f" (hors {list(exclude_years)})" if exclude_years else "")
    s = pd.Series(row, name=label).round(4)
    if verbose:
        print(f"\n{'=' * 50}\n  {label}\n{'=' * 50}")
        print(s.to_string())
    return s
