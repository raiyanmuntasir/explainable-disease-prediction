"""Repeated stratified cross-validation with clinically relevant metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold


def metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "auprc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "sensitivity": recall_score(y_true, pred),
        "specificity": recall_score(y_true, pred, pos_label=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
    }


def cross_validate(pipelines: dict, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, n_repeats: int = 5,
                   seed: int = 0) -> pd.DataFrame:
    """Return one row per (model, fold) with all metrics."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        for name, pipe in pipelines.items():
            m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
            proba = m.predict_proba(X.iloc[te])[:, 1]
            rows.append({"model": name, "fold": fold, **metrics(y.iloc[te].values, proba)})
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    agg = folds.drop(columns="fold").groupby("model").agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for metric in folds.columns.drop(["model", "fold"]):
        out[metric] = agg[(metric, "mean")]
        out[metric + "_sd"] = agg[(metric, "std")]
    return out.sort_values("roc_auc", ascending=False)


def bootstrap_ci(y_true, proba, fn=roc_auc_score, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(fn(y_true[idx], proba[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(fn(y_true, proba)), float(lo), float(hi)
