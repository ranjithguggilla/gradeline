import numpy as np
import pandas as pd
import pytest

from gradeline.data import FEATURES
from gradeline.explain import reasons
from gradeline.models import build
from gradeline.service import score_lot, validate_lot


@pytest.fixture(scope="module")
def bundle(toy):
    X, y = toy
    model = build("ordinal_logit").fit(X, y)
    return {"model": model, "reference": X.median().to_dict(), "threshold": 0.3,
            "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES}}


def lot(toy, **over):
    X, _ = toy
    d = X.iloc[0].to_dict()
    d.update(over)
    return d


def test_score_lot_shape(toy, bundle):
    out = score_lot(lot(toy), bundle)
    assert abs(sum(out["grade_probabilities"].values()) - 1) < 1e-6
    assert 0 <= out["p_low_tier"] <= 1
    assert out["send_to_panel"] == (out["p_low_tier"] >= 0.3)


def test_missing_and_bad_values_rejected(toy, bundle):
    d = lot(toy)
    d.pop("alcohol")
    with pytest.raises(ValueError, match="missing"):
        score_lot(d, bundle)
    with pytest.raises(ValueError, match="number"):
        validate_lot(lot(toy, alcohol="abc"), bundle)
    with pytest.raises(ValueError, match="finite"):
        validate_lot(lot(toy, alcohol=float("nan")), bundle)


def test_out_of_range_is_flagged(toy, bundle):
    out = score_lot(lot(toy, alcohol=500.0), bundle)
    assert "alcohol" in out["outside_training_range"]


def test_more_alcohol_lowers_low_tier_risk(toy, bundle):
    lo = score_lot(lot(toy, alcohol=-2.0), bundle)["p_low_tier"]
    hi = score_lot(lot(toy, alcohol=2.0), bundle)["p_low_tier"]
    assert hi < lo


def test_reasons_name_the_driving_feature(toy, bundle):
    X, _ = toy
    x = pd.DataFrame([X.iloc[0].to_dict()], columns=FEATURES)
    x["alcohol"] = 3.0
    out = reasons(bundle["model"], x, pd.Series(bundle["reference"]), top=2)
    assert out and out[0].startswith("alcohol")
    assert "raises" in out[0]
    assert np.isfinite(x.to_numpy()).all()
