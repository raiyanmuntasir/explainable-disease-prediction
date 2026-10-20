"""Model-agnostic explanations: Shapley values and permutation importance.

``shapley_values`` uses the `shap` package when it is installed. Otherwise it falls
back to a self-contained Monte Carlo permutation estimator (Štrumbelj & Kononenko,
2014). Each sampled permutation walks from a background patient to the explained
patient one feature at a time, so the attributions add up exactly to
``f(x) - E[f(background)]`` (the efficiency property of Shapley values).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402


def _sampling_shapley(f, X: np.ndarray, background: np.ndarray, n_permutations: int, seed: int):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    phi = np.zeros((n, d))
    base = np.zeros(n)
    for i in range(n):
        perms = np.argsort(rng.random((n_permutations, d)), axis=1)
        z = background[rng.integers(0, len(background), n_permutations)]
        # path[p, k] = point after switching the first k features of perm p from z to x
        path = np.repeat(z[:, None, :], d + 1, axis=1)
        for k in range(d):
            cols = perms[:, k]
            path[np.arange(n_permutations), k + 1:, cols] = X[i, cols][:, None]
        out = f(path.reshape(-1, d)).reshape(n_permutations, d + 1)
        deltas = np.diff(out, axis=1)
        np.add.at(phi[i], perms.ravel(), deltas.ravel())
        phi[i] /= n_permutations
        base[i] = out[:, 0].mean()
    return phi, base


def shapley_values(model, X: pd.DataFrame, background: pd.DataFrame, n_permutations: int = 100,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (phi [n x d], base values [n], backend name) for P(disease).

    For every patient i: ``base[i] + phi[i].sum() == P(disease | x_i)`` (exactly for
    the sampling backend, up to shap's own tolerance otherwise).
    """

    def f(a):
        return model.predict_proba(pd.DataFrame(a, columns=X.columns))[:, 1]

    try:
        import shap

        bg = shap.sample(background, min(100, len(background)), random_state=seed)
        ex = shap.Explainer(f, bg)
        sv = ex(X)
        return np.asarray(sv.values), np.broadcast_to(sv.base_values, (len(X),)).astype(float), "shap"
    except ImportError:
        bg = background.sample(min(100, len(background)), random_state=seed).values
        phi, base = _sampling_shapley(f, X.values, bg, n_permutations, seed)
        return phi, base, "sampling-shapley"


def plot_global(phi: np.ndarray, X: pd.DataFrame, path, top: int = 12, title: str = "") -> pd.Series:
    imp = pd.Series(np.abs(phi).mean(0), index=X.columns).sort_values(ascending=False)
    feats = imp.index[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * top + 1.2))
    for row, feat in enumerate(feats):
        j = X.columns.get_loc(feat)
        v = X[feat].values
        c = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)
        jitter = np.random.default_rng(row).normal(0, 0.08, len(v))
        ax.scatter(phi[:, j], row + jitter, c=c, cmap="coolwarm", s=10, alpha=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(feats)), feats)
    ax.set_xlabel("Shapley value (impact on predicted probability)")
    ax.set_title(title or "Global feature attribution (colour = feature value, low → high)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp


def plot_local(phi_row: np.ndarray, x_row: pd.Series, base: float, path, top: int = 10, title: str = ""):
    s = pd.Series(phi_row, index=x_row.index)
    order = s.abs().sort_values(ascending=False).index[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * top + 1.4))
    colors = ["#d62728" if s[k] > 0 else "#1f77b4" for k in order]
    ax.barh([f"{k} = {x_row[k]:.3g}" for k in order], s[order], color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    pred = base + s.sum()
    ax.set_xlabel("contribution to P(disease)")
    ax.set_title(title or f"Patient-level explanation: base {base:.2f} → predicted {pred:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def permutation_importances(model, X: pd.DataFrame, y: pd.Series, seed: int = 0, n_repeats: int = 20) -> pd.DataFrame:
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "auc_drop_mean": r.importances_mean, "auc_drop_sd": r.importances_std})
              .sort_values("auc_drop_mean", ascending=False).reset_index(drop=True))
