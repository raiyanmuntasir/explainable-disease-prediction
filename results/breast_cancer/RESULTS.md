# Results: Wisconsin Diagnostic Breast Cancer

Source: UCI ML Repository via scikit-learn. 569 patients, 30 features, positive class = malignant (prevalence 37.3%). Stratified 80/20 development/test split, seed 42.

## Repeated stratified 5-fold CV on the development set (5 repeats), mean ± sd

| model | roc_auc | auprc | f1 | mcc | sensitivity | specificity | brier |
|---|---|---|---|---|---|---|---|
| adaboost | 0.994 ± 0.006 | 0.993 ± 0.006 | 0.961 ± 0.021 | 0.939 ± 0.032 | 0.945 ± 0.030 | 0.987 ± 0.016 | 0.113 ± 0.005 |
| svm_rbf | 0.994 ± 0.006 | 0.993 ± 0.006 | 0.961 ± 0.023 | 0.938 ± 0.036 | 0.964 ± 0.030 | 0.975 ± 0.020 | 0.023 ± 0.010 |
| logistic_regression | 0.994 ± 0.007 | 0.993 ± 0.007 | 0.964 ± 0.021 | 0.943 ± 0.034 | 0.958 ± 0.028 | 0.982 ± 0.020 | 0.021 ± 0.009 |
| stacking_ensemble | 0.994 ± 0.006 | 0.992 ± 0.006 | 0.963 ± 0.022 | 0.941 ± 0.034 | 0.955 ± 0.030 | 0.982 ± 0.017 | 0.022 ± 0.011 |
| random_forest | 0.990 ± 0.007 | 0.988 ± 0.007 | 0.943 ± 0.027 | 0.910 ± 0.043 | 0.944 ± 0.034 | 0.966 ± 0.030 | 0.034 ± 0.009 |
| gradient_boosting | 0.990 ± 0.009 | 0.988 ± 0.009 | 0.945 ± 0.029 | 0.914 ± 0.046 | 0.947 ± 0.037 | 0.966 ± 0.025 | 0.031 ± 0.014 |
| knn | 0.987 ± 0.013 | 0.982 ± 0.016 | 0.952 ± 0.023 | 0.924 ± 0.036 | 0.941 ± 0.033 | 0.978 ± 0.018 | 0.030 ± 0.011 |

Selected model: **logistic_regression** (highest CV ROC-AUC; near-ties within 0.002 broken by Brier score).

## Ablation (selected model)

| model | roc_auc | auprc | f1 | mcc | sensitivity | specificity | brier |
|---|---|---|---|---|---|---|---|
| logistic_regression (no SMOTE) | 0.995 ± 0.007 | 0.994 ± 0.007 | 0.965 ± 0.021 | 0.946 ± 0.032 | 0.953 ± 0.029 | 0.987 ± 0.016 | 0.020 ± 0.009 |
| logistic_regression (SMOTE) | 0.994 ± 0.007 | 0.993 ± 0.007 | 0.964 ± 0.021 | 0.943 ± 0.034 | 0.958 ± 0.028 | 0.982 ± 0.020 | 0.021 ± 0.009 |
| logistic_regression (SMOTE + PCA 95%) | 0.993 ± 0.007 | 0.992 ± 0.006 | 0.963 ± 0.022 | 0.941 ± 0.034 | 0.954 ± 0.034 | 0.983 ± 0.017 | 0.021 ± 0.009 |

## Held-out test set (models fitted once on the full development set)

| model | roc_auc | auprc | f1 | mcc | sensitivity | specificity | brier |
|---|---|---|---|---|---|---|---|
| adaboost | 0.985 | 0.983 | 0.951 | 0.925 | 0.929 | 0.986 | 0.121 |
| svm_rbf | 0.995 | 0.993 | 0.964 | 0.943 | 0.952 | 0.986 | 0.017 |
| logistic_regression | 0.992 | 0.989 | 0.964 | 0.943 | 0.952 | 0.986 | 0.026 |
| stacking_ensemble | 0.996 | 0.995 | 0.963 | 0.944 | 0.929 | 1.000 | 0.020 |
| random_forest | 0.998 | 0.996 | 0.951 | 0.925 | 0.929 | 0.986 | 0.026 |
| gradient_boosting | 0.996 | 0.994 | 0.950 | 0.926 | 0.905 | 1.000 | 0.030 |
| knn | 0.982 | 0.974 | 0.938 | 0.906 | 0.905 | 0.986 | 0.035 |

ROC-AUC 95% bootstrap CIs: adaboost [0.960, 1.000]; svm_rbf [0.982, 1.000]; logistic_regression [0.977, 1.000]; stacking_ensemble [0.987, 1.000]; random_forest [0.992, 1.000]; gradient_boosting [0.989, 1.000]; knn [0.950, 0.999]

Confusion matrix for logistic_regression at threshold 0.5 (rows = truth [neg, pos], cols = prediction): `[[71, 1], [2, 40]]`

## Explanations

Top permutation importances for logistic_regression (drop in test ROC-AUC):

| feature | AUC drop |
|---|---|
| worst texture | 0.0131 ± 0.0053 |
| worst concavity | 0.0130 ± 0.0027 |
| worst symmetry | 0.0130 ± 0.0028 |
| mean concavity | 0.0047 ± 0.0025 |
| mean concave points | 0.0043 ± 0.0031 |
| worst concave points | 0.0041 ± 0.0023 |
| worst radius | 0.0027 ± 0.0026 |
| worst area | 0.0020 ± 0.0022 |

```json
{
  "logistic_regression": {
    "backend": "sampling-shapley",
    "top_features": {
      "worst texture": 0.0692,
      "radius error": 0.0476,
      "mean concave points": 0.0417,
      "worst concavity": 0.0417,
      "worst radius": 0.0392,
      "worst symmetry": 0.0388,
      "compactness error": 0.0348,
      "worst area": 0.0342,
      "worst concave points": 0.0318,
      "worst perimeter": 0.0304
    },
    "additivity_max_error": 3.3306690738754696e-16,
    "explained_patient": {
      "index": 39,
      "p_disease": 0.833
    }
  },
  "gradient_boosting": {
    "backend": "sampling-shapley",
    "top_features": {
      "worst concave points": 0.0885,
      "worst area": 0.0752,
      "area error": 0.069,
      "worst concavity": 0.0452,
      "worst texture": 0.0422,
      "mean concave points": 0.0378,
      "worst radius": 0.0347,
      "mean texture": 0.0241,
      "worst perimeter": 0.0179,
      "mean area": 0.0174
    }
  }
}
```
