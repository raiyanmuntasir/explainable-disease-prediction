"""Leakage-safe SMOTE.

:class:`SMOTEClassifier` wraps any scikit-learn classifier and oversamples the
minority class *inside* ``fit``. Because resampling happens only when the model
is fitted, cross-validation never sees synthetic samples in a validation fold, a
common source of optimistic results in the medical ML literature.

If `imbalanced-learn` is installed its SMOTE is used; otherwise a compact NumPy
implementation of Chawla et al. (2002) is used.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors


def smote(X: np.ndarray, y: np.ndarray, k: int = 5, random_state: int | None = None):
    """Oversample the minority class until both classes have equal counts."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or counts[0] == counts[1]:
        return X, y
    minority = classes[np.argmin(counts)]
    Xm = X[y == minority]
    n_new = counts.max() - counts.min()
    k = min(k, len(Xm) - 1)
    if k < 1:
        return X, y
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm)
    neigh = nn.kneighbors(Xm, return_distance=False)[:, 1:]
    base = rng.integers(0, len(Xm), n_new)
    partner = neigh[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1))
    synth = Xm[base] + gap * (Xm[partner] - Xm[base])
    return np.vstack([X, synth]), np.concatenate([y, np.full(n_new, minority)])


class SMOTEClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, k_neighbors: int = 5, random_state: int | None = 0):
        self.estimator = estimator
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        try:
            from imblearn.over_sampling import SMOTE

            Xr, yr = SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state).fit_resample(X, y)
        except ImportError:
            Xr, yr = smote(X, y, self.k_neighbors, self.random_state)
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X, dtype=float))
