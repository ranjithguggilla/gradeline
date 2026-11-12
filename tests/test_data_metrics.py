import numpy as np
import pandas as pd

from gradeline.data import FEATURES, collapse_grade, drop_duplicates, load_raw, prepare, split
from gradeline.economics import (
    Costs,
    best_threshold,
    budget_recall,
    hindsight_cost,
    policy_cost,
)
from gradeline.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    grade_metrics,
    low_tier_probability,
    qwk,
    ranked_probability_score,
)


def test_collapse_grade():
    assert collapse_grade([3, 4, 5, 6, 7, 8, 9]).tolist() == [0, 0, 1, 2, 3, 4, 4]


def test_load_and_dedupe_real_files():
    raw = load_raw()
    assert len(raw) == 6497
    assert set(FEATURES) <= set(raw.columns)
    assert len(drop_duplicates(raw)) == 5320


def test_split_is_disjoint_and_stratified():
    d = drop_duplicates(load_raw())
    parts = split(d, seed=0)
    assert sum(len(p) for p in parts.values()) == len(d)
    X, y = prepare(parts["test"])
    assert abs((y <= 1).mean() - (collapse_grade(d["quality"]) <= 1).mean()) < 0.02


def test_split_is_reproducible():
    d = drop_duplicates(load_raw())
    a, b = split(d, 3), split(d, 3)
    assert a["train"].equals(b["train"])


def test_qwk_penalises_distance():
    y = np.array([0, 1, 2, 3, 4] * 20)
    near = np.clip(y + 1, 0, 4)
    far = np.clip(y + 2, 0, 4)
    assert qwk(y, y) == 1.0
    assert qwk(y, near) > qwk(y, far)


def test_grade_metrics_keys():
    m = grade_metrics([0, 1, 2], [0, 2, 2])
    assert m["exact"] == 2 / 3 and m["within1"] == 1.0 and abs(m["mae"] - 1 / 3) < 1e-9


def test_rps_zero_for_perfect():
    y = np.array([0, 2, 4])
    p = np.eye(5)[y]
    assert ranked_probability_score(y, p) == 0.0


def test_rps_prefers_closer_miss():
    y = np.array([2])
    near = np.array([[0, 0, 0, 1.0, 0]])
    far = np.array([[0, 0, 0, 0, 1.0]])
    assert ranked_probability_score(y, near) < ranked_probability_score(y, far)


def test_low_tier_probability_sums_first_classes():
    p = np.array([[0.1, 0.2, 0.3, 0.2, 0.2]])
    assert abs(low_tier_probability(p, 1)[0] - 0.3) < 1e-12


def test_ece_perfect_and_bad():
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == 0.0
    assert expected_calibration_error([0, 0, 0, 0], [0.9] * 4) > 0.85


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(5, 1, 400)
    lo, hi = bootstrap_ci(lambda a: float(a.mean()), x, n=200)
    assert lo < x.mean() < hi


def test_policy_cost_arithmetic():
    c = Costs(review=10, missed_low=100)
    low = np.array([1, 1, 0, 0], bool)
    assert policy_cost(np.zeros(4, bool), low, c) == 200
    assert policy_cost(np.ones(4, bool), low, c) == 40
    assert hindsight_cost(low, c) == 20
    assert c.break_even == 0.1


def test_best_threshold_matches_break_even_on_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    low = rng.uniform(0, 1, 20000) < p
    c = Costs(review=100, missed_low=400)
    assert abs(best_threshold(p, low, c) - c.break_even) < 0.05


def test_budget_recall():
    score = np.array([0.9, 0.8, 0.1, 0.2])
    low = np.array([1, 0, 1, 0], bool)
    assert budget_recall(score, low, 0.25) == 0.5
    assert isinstance(pd.Series([1]).sum(), (int, np.integer))
