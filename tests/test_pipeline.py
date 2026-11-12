import json

import joblib
import numpy as np

from gradeline.pipeline import one_se_pick, run


def test_one_se_prefers_simpler_within_error():
    cv = {
        "ordinal_logit": {"qwk": np.array([0.51, 0.53, 0.52])},
        "boosted_cumulative": {"qwk": np.array([0.52, 0.53, 0.52])},
    }
    assert one_se_pick(cv, ["ordinal_logit", "boosted_cumulative"]) == "ordinal_logit"


def test_one_se_keeps_clear_winner():
    cv = {
        "ordinal_logit": {"qwk": np.array([0.40, 0.41, 0.40])},
        "boosted_cumulative": {"qwk": np.array([0.60, 0.61, 0.60])},
    }
    assert one_se_pick(cv, ["ordinal_logit", "boosted_cumulative"]) == "boosted_cumulative"


def test_quick_run_writes_outputs(tmp_path):
    res = run(tmp_path, tmp_path / "charts", quick=True)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "charts" / "triage_cost.png").exists()
    bundle = joblib.load(tmp_path / "gradeline_model.joblib")
    assert bundle["name"] == res["shipped"]
    on_disk = json.loads((tmp_path / "metrics.json").read_text())
    assert on_disk["rows"]["deduped"] == 5320
    assert 0 < on_disk["triage"]["review_share"] <= 1
    t = res["test"][res["shipped"]]
    assert t["qwk"] > 0.3 and t["within1"] > 0.9


def test_app_builds_from_bundle(tmp_path):
    from gradeline.app import build_app
    run(tmp_path, None, quick=True)
    demo = build_app(str(tmp_path / "gradeline_model.joblib"))
    assert demo is not None
