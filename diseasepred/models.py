"""Model zoo: classical baselines, boosting ensembles and a stacked ensemble.

Every model is wrapped in the same leakage-safe pipeline::

    median imputation -> standardisation -> [optional PCA] -> SMOTE (fit only) -> classifier
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .smote import SMOTEClassifier


def _xgboost(seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=1)


def base_models(seed: int = 0) -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=5000, C=1.0),
        # Platt-scaled SVM (replacement for the deprecated SVC(probability=True))
        "svm_rbf": CalibratedClassifierCV(SVC(C=1.0, gamma="scale", random_state=seed), method="sigmoid",
                                          ensemble=False, cv=5),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "adaboost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                        subsample=0.9, random_state=seed),
    }
    xgb = _xgboost(seed)
    if xgb is not None:
        models["xgboost"] = xgb
    return models


def stacking(seed: int = 0) -> StackingClassifier:
    b = base_models(seed)
    level0 = [(k, b[k]) for k in ("svm_rbf", "random_forest", "gradient_boosting", "knn")]
    return StackingClassifier(estimators=level0, final_estimator=LogisticRegression(max_iter=5000),
                              stack_method="predict_proba", cv=5, n_jobs=1)


def make_pipeline(clf, use_smote: bool = True, pca: float | None = None, seed: int = 0) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if pca:
        steps.append(("pca", PCA(n_components=pca, random_state=seed)))
    steps.append(("clf", SMOTEClassifier(clf, random_state=seed) if use_smote else clf))
    return Pipeline(steps)


def all_pipelines(seed: int = 0, use_smote: bool = True, pca: float | None = None) -> dict[str, Pipeline]:
    models = {**base_models(seed), "stacking_ensemble": stacking(seed)}
    return {k: make_pipeline(v, use_smote, pca, seed) for k, v in models.items()}
