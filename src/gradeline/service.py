"""Scoring entry point shared by the app and the tests."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import FEATURES, GRADE_LABELS, LOW_TIER_MAX
from .explain import reasons
from .metrics import low_tier_probability
from .ordinal import decode_median, expected_grade

DEFAULT_BUNDLE = Path("outputs") / "gradeline_model.joblib"


def load_bundle(path: str | Path = DEFAULT_BUNDLE) -> dict:
    return joblib.load(path)


def validate_lot(values: dict, bundle: dict) -> pd.DataFrame:
    """Build a one-row frame, rejecting missing or non-numeric measurements."""
    missing = [c for c in FEATURES if c not in values]
    if missing:
        raise ValueError(f"missing measurements: {', '.join(missing)}")
    row = {}
    for c in FEATURES:
        try:
            row[c] = float(values[c])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{c} must be a number") from exc
        if not np.isfinite(row[c]):
            raise ValueError(f"{c} must be finite")
    return pd.DataFrame([row], columns=FEATURES)


def score_lot(values: dict, bundle: dict) -> dict:
    """Grade distribution, expected grade, triage call and reasons for one lot."""
    x = validate_lot(values, bundle)
    model = bundle["model"]
    proba = model.predict_proba(x)
    p_low = float(low_tier_probability(proba, LOW_TIER_MAX)[0])
    ranges = bundle.get("ranges", {})
    outside = [c for c, (lo, hi) in ranges.items() if not lo <= float(x[c].iloc[0]) <= hi]
    return {
        "grade_probabilities": {g: float(p) for g, p in zip(GRADE_LABELS, proba[0], strict=True)},
        "most_likely_grade": GRADE_LABELS[int(decode_median(proba)[0])],
        "expected_grade_index": float(expected_grade(proba)[0]),
        "p_low_tier": p_low,
        "send_to_panel": bool(p_low >= bundle["threshold"]),
        "threshold": float(bundle["threshold"]),
        "reasons": reasons(model, x, pd.Series(bundle["reference"])),
        "outside_training_range": outside,
    }
