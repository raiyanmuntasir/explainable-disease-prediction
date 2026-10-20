"""End-to-end experiment: ``python -m diseasepred --dataset breast_cancer``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from .data import DATASETS, load_csv  # noqa: E402
from .evaluate import bootstrap_ci, cross_validate, metrics, summarise  # noqa: E402
from .explain import permutation_importances, plot_global, plot_local, shapley_values  # noqa: E402
from .models import all_pipelines, make_pipeline  # noqa: E402

PRETTY = ["roc_auc", "auprc", "f1", "mcc", "sensitivity", "specificity", "brier"]


def _md_table(df: pd.DataFrame, cols, index_name="model") -> str:
    head = f"| {index_name} | " + " | ".join(cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    rows = []
    for idx, r in df.iterrows():
        cells = []
        for c in cols:
            sd = r.get(c + "_sd")
            cells.append(f"{r[c]:.3f} ± {sd:.3f}" if sd is not None and not pd.isna(sd) else f"{r[c]:.3f}")
        rows.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="diseasepred")
    ap.add_argument("--dataset", default="breast_cancer", help=f"one of {list(DATASETS)} or 'csv'")
    ap.add_argument("--csv", help="path to CSV when --dataset csv")
    ap.add_argument("--target", help="target column when --dataset csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=5, help="repeats of stratified 5-fold CV")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.dataset == "csv":
        X, y, info = load_csv(args.csv, args.target)
    else:
        X, y, info = DATASETS[args.dataset]()
    out = args.out or Path("results") / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    print(f"{info['name']}: {len(X)} patients, {X.shape[1]} features, prevalence {y.mean():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)

    # 1. repeated CV on the development set ------------------------------------------------
    pipes = all_pipelines(args.seed)
    folds = cross_validate(pipes, X_tr, y_tr, n_repeats=args.repeats, seed=args.seed)
    cv = summarise(folds)
    cv.to_csv(out / "cv_metrics.csv")
    # Model selection: highest mean ROC-AUC; near-ties (within 0.002) are broken by
    # the lower Brier score, because well-calibrated probabilities matter clinically.
    near = cv[cv.roc_auc >= cv.roc_auc.max() - 0.002]
    best = near.brier.idxmin()
    print(cv[["roc_auc", "roc_auc_sd", "f1", "mcc"]].round(3))

    # 2. ablation: SMOTE and PCA on the best model ---------------------------------------
    best_clf = pipes[best].named_steps["clf"].estimator
    ablation_pipes = {
        f"{best} (SMOTE)": pipes[best],
        f"{best} (no SMOTE)": make_pipeline(best_clf, use_smote=False, seed=args.seed),
        f"{best} (SMOTE + PCA 95%)": make_pipeline(best_clf, use_smote=True, pca=0.95, seed=args.seed),
    }
    abl = summarise(cross_validate(ablation_pipes, X_tr, y_tr, n_repeats=args.repeats, seed=args.seed))
    abl.to_csv(out / "ablation.csv")

    # 3. held-out test set ----------------------------------------------------------------
    fitted = {k: p.fit(X_tr, y_tr) for k, p in pipes.items()}
    test_rows, fig, ax = [], *plt.subplots(figsize=(5.5, 5))
    for name, m in fitted.items():
        p = m.predict_proba(X_te)[:, 1]
        auc, lo, hi = bootstrap_ci(y_te, p, seed=args.seed)
        test_rows.append({"model": name, **metrics(y_te.values, p), "roc_auc_ci": f"[{lo:.3f}, {hi:.3f}]"})
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, lw=2 if name == best else 1, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title(f"Held-out ROC: {info['name']}")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_test.png", dpi=150)
    plt.close(fig)
    test = pd.DataFrame(test_rows).set_index("model").loc[cv.index]
    test.to_csv(out / "test_metrics.csv")

    cm = confusion_matrix(y_te, (fitted[best].predict_proba(X_te)[:, 1] >= 0.5).astype(int))

    # 4. explanations ---------------------------------------------------------------------
    # Explain the best model; also explain gradient boosting so tree-ensemble behaviour is visible.
    to_explain = [best] + (["gradient_boosting"] if best != "gradient_boosting" else [])
    explanations = {}
    for name in to_explain:
        phi, base, backend = shapley_values(fitted[name], X_te, X_tr, seed=args.seed)
        imp = plot_global(phi, X_te, out / f"shap_global_{name}.png",
                          title=f"Shapley attribution, {name} ({backend})")
        explanations[name] = {"backend": backend, "top_features": imp.head(10).round(4).to_dict()}
        if name == best:
            proba = fitted[name].predict_proba(X_te)[:, 1]
            # a correctly detected positive that is not trivially certain (P closest to 0.8)
            i = int(np.argmin(np.where((y_te.values == 1) & (proba >= 0.5), np.abs(proba - 0.8), np.inf)))
            plot_local(phi[i], X_te.iloc[i], base[i], out / f"shap_local_patient_{name}.png",
                       title=f"Why test patient #{X_te.index[i]} was flagged ({name}): "
                             f"P = {proba[i]:.2f}")
            # sanity check: efficiency property
            explanations[name]["additivity_max_error"] = float(np.max(np.abs(base + phi.sum(1) - proba)))
            explanations[name]["explained_patient"] = {"index": int(X_te.index[i]), "p_disease": round(float(proba[i]), 3)}
    perm = permutation_importances(fitted[best], X_te, y_te, seed=args.seed)
    perm.to_csv(out / f"permutation_importance_{best}.csv", index=False)

    # 5. report ---------------------------------------------------------------------------
    report = [
        f"# Results: {info['name']}",
        "",
        f"Source: {info['source']}. {len(X)} patients, {X.shape[1]} features, positive class = "
        f"{info['positive']} (prevalence {y.mean():.1%}). Stratified 80/20 development/test split, seed {args.seed}.",
        "",
        f"## Repeated stratified 5-fold CV on the development set ({args.repeats} repeats), mean ± sd",
        "",
        _md_table(cv, PRETTY),
        "",
        f"Selected model: **{best}** (highest CV ROC-AUC; near-ties within 0.002 broken by Brier score).",
        "",
        "## Ablation (selected model)",
        "",
        _md_table(abl, PRETTY),
        "",
        "## Held-out test set (models fitted once on the full development set)",
        "",
        _md_table(test, PRETTY) + "\n\nROC-AUC 95% bootstrap CIs: "
        + "; ".join(f"{k} {v}" for k, v in test.roc_auc_ci.items()),
        "",
        f"Confusion matrix for {best} at threshold 0.5 (rows = truth [neg, pos], cols = prediction): "
        f"`{cm.tolist()}`",
        "",
        "## Explanations",
        "",
        f"Top permutation importances for {best} (drop in test ROC-AUC):",
        "",
        "| feature | AUC drop |",
        "|---|---|",
        *[f"| {r.feature} | {r.auc_drop_mean:.4f} ± {r.auc_drop_sd:.4f} |" for r in perm.head(8).itertuples()],
        "",
        "```json",
        json.dumps(explanations, indent=2),
        "```",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(report))
    print(f"best model by CV ROC-AUC: {best}; wrote {out}/")


if __name__ == "__main__":
    main()
