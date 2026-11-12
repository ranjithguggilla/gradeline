import numpy as np
import pandas as pd
import pytest

from gradeline.data import FEATURES


@pytest.fixture(scope="session")
def toy():
    """Small synthetic table with a monotone signal so ordinal models have something to learn."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    X["is_red"] = rng.integers(0, 2, n)
    for c in ("residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
              "sulphates", "volatile_acidity"):
        X[c] = np.exp(X[c] * 0.3)
    latent = 1.2 * X["alcohol"] - 0.8 * X["volatile_acidity"] + rng.normal(scale=0.8, size=n)
    y = np.digitize(latent, np.quantile(latent, [0.1, 0.4, 0.75, 0.93]))
    return X, y
