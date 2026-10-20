"""Dataset loaders. Every loader returns ``(X: DataFrame, y: Series, info: dict)``
with ``y == 1`` meaning *disease present*.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_breast_cancer() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Wisconsin Diagnostic Breast Cancer (569 patients, 30 FNA-derived features).

    Ships with scikit-learn, so no download is required. Re-coded so that
    malignant = 1.
    """
    from sklearn.datasets import load_breast_cancer as _load

    d = _load(as_frame=True)
    y = (d.target == 0).astype(int).rename("malignant")
    return d.data, y, {"name": "Wisconsin Diagnostic Breast Cancer", "positive": "malignant",
                       "source": "UCI ML Repository via scikit-learn"}


def load_pima() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Pima Indians Diabetes (768 patients, 8 clinical features). Downloaded from OpenML.

    Zeros in physiologically impossible columns are treated as missing and
    imputed inside the model pipeline.
    """
    from sklearn.datasets import fetch_openml

    d = fetch_openml(data_id=37, as_frame=True, parser="auto")
    X = d.data.astype(float)
    for col in ["plas", "pres", "skin", "insu", "mass"]:
        X.loc[X[col] == 0, col] = float("nan")
    y = (d.target == "tested_positive").astype(int).rename("diabetes")
    return X, y, {"name": "Pima Indians Diabetes", "positive": "diabetes", "source": "OpenML #37"}


def load_csv(path: str | Path, target: str, positive=1) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Any tabular CSV with a binary target column (e.g. UCI Heart Disease)."""
    df = pd.read_csv(path)
    y = (df.pop(target) == positive).astype(int).rename(target)
    X = df.select_dtypes("number")
    return X, y, {"name": Path(path).stem, "positive": str(positive), "source": str(path)}


DATASETS = {"breast_cancer": load_breast_cancer, "pima": load_pima}
