import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from diseasepred.data import load_breast_cancer
from diseasepred.evaluate import bootstrap_ci, cross_validate, summarise
from diseasepred.explain import shapley_values
from diseasepred.models import all_pipelines, make_pipeline
from diseasepred.smote import SMOTEClassifier, smote


class TestDiseasePred(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, _ = load_breast_cancer()

    def test_positive_class_is_malignant(self):
        self.assertEqual(int(self.y.sum()), 212)

    def test_smote_balances_classes(self):
        X = np.random.default_rng(0).normal(size=(120, 4))
        y = np.r_[np.zeros(100), np.ones(20)].astype(int)
        Xr, yr = smote(X, y, random_state=0)
        self.assertEqual((yr == 0).sum(), (yr == 1).sum())
        # synthetic points lie inside the minority-class bounding box
        mino = X[y == 1]
        self.assertTrue((Xr[len(X):] >= mino.min(0) - 1e-9).all() and (Xr[len(X):] <= mino.max(0) + 1e-9).all())

    def test_smote_only_applied_at_fit_time(self):
        m = SMOTEClassifier(LogisticRegression(max_iter=1000)).fit(self.X.values, self.y.values)
        self.assertEqual(m.predict_proba(self.X.values).shape, (len(self.X), 2))

    def test_cross_validation_runs_and_is_sane(self):
        pipes = {"lr": make_pipeline(LogisticRegression(max_iter=2000))}
        cv = summarise(cross_validate(pipes, self.X, self.y, n_repeats=1))
        self.assertGreater(cv.loc["lr", "roc_auc"], 0.95)

    def test_all_pipelines_build(self):
        self.assertIn("stacking_ensemble", all_pipelines())

    def test_shapley_additivity(self):
        m = make_pipeline(LogisticRegression(max_iter=2000)).fit(self.X, self.y)
        Xs = self.X.iloc[:3]
        phi, base, _ = shapley_values(m, Xs, self.X, n_permutations=20)
        p = m.predict_proba(Xs)[:, 1]
        np.testing.assert_allclose(base + phi.sum(1), p, atol=1e-6 if _ == "sampling-shapley" else 1e-2)

    def test_bootstrap_ci_contains_point_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p = np.clip(y * 0.6 + rng.random(200) * 0.5, 0, 1)
        est, lo, hi = bootstrap_ci(y, p, n=300)
        self.assertTrue(lo <= est <= hi)


if __name__ == "__main__":
    unittest.main()
