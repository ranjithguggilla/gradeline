import numpy as np

from gradeline.models import build, predict_grades
from gradeline.ordinal import (
    CumulativeBinary,
    ThresholdRegressor,
    apply_cutpoints,
    decode_argmax,
    decode_median,
    expected_grade,
    fit_cutpoints,
)


def test_cumulative_probabilities_are_a_distribution(toy):
    X, y = toy
    m = build("ordinal_logit").fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X), 5)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p >= 0).all()


def test_cumulative_is_monotone(toy):
    X, y = toy
    m = build("ordinal_logit").fit(X, y)
    gt = m.cumulative(X)
    assert (np.diff(gt, axis=1) <= 1e-12).all()


def test_median_decode_known_case():
    p = np.array([[0.1, 0.2, 0.25, 0.25, 0.2], [0.6, 0.2, 0.1, 0.05, 0.05]])
    assert decode_median(p).tolist() == [2, 0]
    assert decode_argmax(p).tolist() == [2, 0]


def test_median_differs_from_argmax_on_bimodal():
    p = np.array([[0.34, 0.0, 0.0, 0.33, 0.33]])
    assert decode_argmax(p)[0] == 0
    assert decode_median(p)[0] == 3


def test_expected_grade():
    p = np.array([[0.0, 0.5, 0.5, 0.0, 0.0]])
    assert expected_grade(p)[0] == 1.5


def test_cutpoints_recover_a_clean_mapping():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 400)
    score = y + rng.normal(scale=0.15, size=400)
    cuts = fit_cutpoints(score, y, 5)
    assert (np.diff(cuts) > 0).all()
    assert (apply_cutpoints(score, cuts) == y).mean() > 0.9


def test_threshold_regressor_predicts_valid_classes(toy):
    X, y = toy
    m = build("ridge_cutpoints").fit(X, y)
    pred = m.predict(X)
    assert set(np.unique(pred)) <= set(range(5))
    assert len(m.cutpoints_) == 4


def test_models_beat_the_mode_on_toy(toy):
    X, y = toy
    from gradeline.metrics import qwk
    for name in ("ordinal_logit", "ridge_cutpoints"):
        m = build(name).fit(X, y)
        assert qwk(y, predict_grades(name, m, X)) > 0.4


def test_cumulative_binary_handles_custom_class_count(toy):
    from sklearn.linear_model import LogisticRegression
    X, y = toy
    y3 = np.minimum(y, 2)
    m = CumulativeBinary(LogisticRegression(max_iter=500), n_classes=3).fit(X, y3)
    assert m.predict_proba(X).shape[1] == 3
    assert ThresholdRegressor is not None
