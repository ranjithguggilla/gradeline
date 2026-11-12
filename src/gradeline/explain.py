"""Plain-language reasons for a single lot, computed by what-if substitution."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ordinal import expected_grade

NAMES = {
    "fixed_acidity": "fixed acidity", "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid", "residual_sugar": "residual sugar",
    "chlorides": "chlorides", "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide", "density": "density",
    "pH": "pH", "sulphates": "sulphates", "alcohol": "alcohol", "is_red": "colour",
}


def reasons(model, x: pd.DataFrame, reference: pd.Series, top: int = 3) -> list[str]:
    """Swap each measurement for the reference (median) lot and see how the expected grade moves."""
    base = expected_grade(model.predict_proba(x))[0]
    rows = []
    for col in x.columns:
        alt = x.copy()
        alt[col] = reference[col]
        shift = base - expected_grade(model.predict_proba(alt))[0]
        rows.append((col, shift))
    rows.sort(key=lambda r: -abs(r[1]))
    out = []
    for col, shift in rows[:top]:
        if abs(shift) < 0.005:
            continue
        val = x[col].iloc[0]
        shown = ("red" if val >= 0.5 else "white") if col == "is_red" else f"{val:g}"
        direction = "raises" if shift > 0 else "lowers"
        out.append(f"{NAMES[col]} ({shown}) {direction} the expected grade by {abs(shift):.2f} "
                   f"compared with a typical lot")
    return out


def global_importance(model, X: pd.DataFrame, y, scorer, n_repeats: int = 5, seed: int = 0):
    """Permutation importance measured as loss in a user-supplied score."""
    rng = np.random.default_rng(seed)
    base = scorer(y, model.predict(X))
    drops = {}
    for col in X.columns:
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            vals.append(base - scorer(y, model.predict(Xp)))
        drops[col] = float(np.mean(vals))
    return dict(sorted(drops.items(), key=lambda kv: -kv[1]))
