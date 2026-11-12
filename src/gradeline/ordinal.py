"""Ordinal estimators built on scikit-learn: cumulative binary and threshold models."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class CumulativeBinary(BaseEstimator, ClassifierMixin):
    """Frank and Hall decomposition.

    One binary model per boundary estimates P(grade > k). The cumulative
    probabilities are made monotone and differenced into class probabilities,
    so the output is a proper distribution over ordered grades.
    """

    def __init__(self, base=None, n_classes: int = 5):
        self.base = base
        self.n_classes = n_classes

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.arange(self.n_classes)
        self.models_ = []
        for k in range(self.n_classes - 1):
            m = clone(self.base)
            m.fit(X, (y > k).astype(int))
            self.models_.append(m)
        return self

    def cumulative(self, X) -> np.ndarray:
        """P(grade > k) for k = 0..K-2, forced to be non-increasing in k."""
        gt = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        return np.minimum.accumulate(gt, axis=1)

    def predict_proba(self, X):
        gt = self.cumulative(X)
        upper = np.hstack([np.ones((len(gt), 1)), gt])
        lower = np.hstack([gt, np.zeros((len(gt), 1))])
        return np.clip(upper - lower, 0.0, 1.0)

    def predict(self, X):
        return decode_median(self.predict_proba(X))


class ThresholdRegressor(BaseEstimator, RegressorMixin):
    """Regress the grade, then learn cutpoints that maximise quadratic kappa.

    Cutpoints are fitted on out-of-fold predictions so they are not tuned on
    scores the regressor has already memorised.
    """

    def __init__(self, base=None, n_classes: int = 5, inner_folds: int = 5, seed: int = 0):
        self.base = base
        self.n_classes = n_classes
        self.inner_folds = inner_folds
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        self.model_ = clone(self.base).fit(X, y)
        cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        oof = cross_val_predict(clone(self.base), X, y, cv=cv)
        self.cutpoints_ = fit_cutpoints(oof, y, self.n_classes)
        return self

    def predict_score(self, X):
        return self.model_.predict(X)

    def predict(self, X):
        return apply_cutpoints(self.predict_score(X), self.cutpoints_)


def apply_cutpoints(score, cutpoints) -> np.ndarray:
    return np.searchsorted(np.sort(cutpoints), np.asarray(score), side="right")


def fit_cutpoints(score, y, n_classes: int) -> np.ndarray:
    """Search for sorted cutpoints that maximise quadratic weighted kappa."""
    init = np.arange(n_classes - 1) + 0.5

    def loss(c):
        pred = apply_cutpoints(score, c)
        return -cohen_kappa_score(y, pred, weights="quadratic")

    res = minimize(loss, init, method="Powell", options={"xtol": 1e-3, "ftol": 1e-4})
    return np.sort(res.x)


def decode_argmax(proba: np.ndarray) -> np.ndarray:
    return proba.argmax(axis=1)


def decode_median(proba: np.ndarray) -> np.ndarray:
    """Smallest grade whose cumulative probability reaches one half.

    The median minimises expected absolute grade error, which is the loss that
    matters when being two grades out is worse than being one grade out.
    """
    return (np.cumsum(proba, axis=1) < 0.5).sum(axis=1).clip(0, proba.shape[1] - 1)


def expected_grade(proba: np.ndarray) -> np.ndarray:
    return proba @ np.arange(proba.shape[1])
