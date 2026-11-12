"""Metrics for ordered grades and for the low-tier probability."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score


def qwk(y, pred) -> float:
    return float(cohen_kappa_score(y, pred, weights="quadratic"))


def grade_metrics(y, pred) -> dict[str, float]:
    y, pred = np.asarray(y), np.asarray(pred)
    return {
        "qwk": qwk(y, pred),
        "mae": float(np.abs(y - pred).mean()),
        "exact": float((y == pred).mean()),
        "within1": float((np.abs(y - pred) <= 1).mean()),
    }


def ranked_probability_score(y, proba) -> float:
    """Mean squared distance between predicted and observed cumulative distributions."""
    y = np.asarray(y)
    k = proba.shape[1]
    obs = (np.arange(k)[None, :] >= y[:, None]).astype(float)
    cum = np.cumsum(proba, axis=1)
    return float(((cum - obs) ** 2)[:, :-1].sum(axis=1).mean() / (k - 1))


def low_tier_probability(proba: np.ndarray, low_max: int) -> np.ndarray:
    return proba[:, : low_max + 1].sum(axis=1)


def expected_calibration_error(y_true, p, bins: int = 10) -> float:
    y_true, p = np.asarray(y_true, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def bootstrap_ci(fn, *arrays, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    size = len(arrays[0])
    vals = []
    for _ in range(n):
        i = rng.integers(0, size, size)
        vals.append(fn(*[a[i] for a in arrays]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
