"""Candidate model registry."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import FEATURES
from .ordinal import CumulativeBinary, ThresholdRegressor, decode_argmax, decode_median

SKEWED = [
    "residual_sugar", "chlorides", "free_sulfur_dioxide",
    "total_sulfur_dioxide", "sulphates", "volatile_acidity",
]
SEED = 0


def _logs():
    """Log-transform right-skewed columns and scale everything, inside the pipeline."""
    skew_idx = [FEATURES.index(c) for c in SKEWED]
    rest_idx = [i for i in range(len(FEATURES)) if i not in skew_idx]
    return ColumnTransformer(
        [("log", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), skew_idx),
         ("raw", StandardScaler(), rest_idx)]
    )


def _logit():
    return Pipeline([("prep", _logs()), ("clf", LogisticRegression(C=1.0, max_iter=2000))])


def _ridge():
    return Pipeline([("prep", _logs()), ("reg", Ridge(alpha=3.0))])


def _hgb_clf():
    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.06, max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=SEED)


def _hgb_reg():
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.06, max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=SEED)


def _alcohol_only():
    idx = [FEATURES.index("alcohol")]
    return Pipeline([("pick", FunctionTransformer(lambda X: np.asarray(X)[:, idx])),
                     ("reg", LinearRegression())])


# name -> (factory, kind, decoder, complexity rank used by the one-SE rule)
REGISTRY = {
    "mode_only": (lambda: DummyClassifier(strategy="prior"), "proba", decode_argmax, 0),
    "alcohol_cutpoints": (lambda: ThresholdRegressor(_alcohol_only(), seed=SEED), "score", None, 1),
    "ordinal_logit": (lambda: CumulativeBinary(_logit()), "proba", decode_median, 2),
    "multinomial_logit": (_logit, "proba", decode_argmax, 3),
    "ridge_cutpoints": (lambda: ThresholdRegressor(_ridge(), seed=SEED), "score", None, 3),
    "boosted_multiclass": (_hgb_clf, "proba", decode_argmax, 4),
    "boosted_cumulative": (lambda: CumulativeBinary(_hgb_clf()), "proba", decode_median, 5),
    "boosted_cutpoints": (lambda: ThresholdRegressor(_hgb_reg(), seed=SEED), "score", None, 5),
}


def build(name: str):
    return REGISTRY[name][0]()


def predict_grades(name: str, model, X) -> np.ndarray:
    _, kind, decoder, _ = REGISTRY[name]
    if kind == "score":
        return model.predict(X)
    return decoder(model.predict_proba(X))


def predict_proba(model, X) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None
