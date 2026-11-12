"""End-to-end experiment: selection, test evaluation, studies, charts, bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import krippendorff
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from mlxtend.evaluate import mcnemar, mcnemar_table
from scipy.stats import friedmanchisquare
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from . import __version__
from .data import FEATURES, LOW_TIER_MAX, drop_duplicates, load_raw, prepare, split
from .economics import Costs, best_threshold, budget_recall, hindsight_cost, policy_cost
from .explain import global_importance
from .metrics import (
    bootstrap_ci,
    expected_calibration_error,
    grade_metrics,
    low_tier_probability,
    qwk,
    ranked_probability_score,
)
from .models import REGISTRY, build, predict_grades
from .ordinal import decode_argmax, decode_median, expected_grade

SHIP_POOL = [n for n, v in REGISTRY.items() if v[1] == "proba" and n != "mode_only"]


def cv_scores(name: str, X, y, folds) -> dict[str, np.ndarray]:
    out = {"qwk": [], "mae": [], "rps": []}
    for tr, te in folds:
        m = build(name).fit(X.iloc[tr], y[tr])
        pred = predict_grades(name, m, X.iloc[te])
        gm = grade_metrics(y[te], pred)
        out["qwk"].append(gm["qwk"])
        out["mae"].append(gm["mae"])
        proba = m.predict_proba(X.iloc[te]) if hasattr(m, "predict_proba") else None
        out["rps"].append(ranked_probability_score(y[te], proba) if proba is not None else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def one_se_pick(cv: dict[str, dict], pool: list[str]) -> str:
    """Simplest model whose mean CV kappa sits within one standard error of the best."""
    stats = {n: (cv[n]["qwk"].mean(), cv[n]["qwk"].std(ddof=1) / np.sqrt(len(cv[n]["qwk"])))
             for n in pool}
    best = max(stats, key=lambda n: stats[n][0])
    cutoff = stats[best][0] - stats[best][1]
    ok = [n for n in pool if stats[n][0] >= cutoff]
    return min(ok, key=lambda n: (REGISTRY[n][3], -stats[n][0]))


def _fit_eval(name, tr, te, y_tr, y_te):
    m = build(name).fit(tr, y_tr)
    return m, predict_grades(name, m, te)


def leakage_study(raw: pd.DataFrame, seeds=range(5)) -> dict:
    """Random row split on the raw table vs the same split after removing repeated rows."""
    out = {"raw_random": [], "deduped_random": []}
    dedup = drop_duplicates(raw)
    for s in seeds:
        for key, frame in (("raw_random", raw), ("deduped_random", dedup)):
            rng = np.random.default_rng(s)
            idx = rng.permutation(len(frame))
            cut = int(0.75 * len(frame))
            tr, te = frame.iloc[idx[:cut]], frame.iloc[idx[cut:]]
            Xtr, ytr = prepare(tr)
            Xte, yte = prepare(te)
            m = build("boosted_multiclass").fit(Xtr, ytr)
            out[key].append(qwk(yte, m.predict(Xte)))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["raw_share_duplicated"] = float(raw.duplicated().mean())
    return res


def colour_study(dedup: pd.DataFrame, name: str) -> dict:
    parts = split(dedup, seed=1)
    train = pd.concat([parts["train"], parts["val"]])
    test = parts["test"]
    X_tr, y_tr = prepare(train)
    X_te, y_te = prepare(test)
    pooled = build(name).fit(X_tr, y_tr)
    res = {"pooled": qwk(y_te, predict_grades(name, pooled, X_te))}
    for colour, flag in (("red", 1), ("white", 0)):
        mtr, mte = train[train.is_red == flag], test[test.is_red == flag]
        Xa, ya = prepare(mtr)
        Xb, yb = prepare(mte)
        own = build(name).fit(Xa, ya)
        res[f"{colour}_pooled_model"] = qwk(yb, predict_grades(name, pooled, Xb))
        res[f"{colour}_own_model"] = qwk(yb, predict_grades(name, own, Xb))
    red = dedup[dedup.is_red == 1]
    white = dedup[dedup.is_red == 0]
    Xw, yw = prepare(white)
    Xr, yr = prepare(red)
    wmodel = build(name).fit(Xw, yw)
    res["white_to_red_transfer"] = qwk(yr, predict_grades(name, wmodel, Xr))
    return res


def calibration_study(name, parts) -> dict:
    """Isotonic recalibration of the low-tier probability, judged by two-way split of validation."""
    Xtr, ytr = prepare(parts["train"])
    Xv, yv = prepare(parts["val"])
    Xte, yte = prepare(parts["test"])
    m = build(name).fit(Xtr, ytr)
    pv = low_tier_probability(m.predict_proba(Xv), LOW_TIER_MAX)
    pt = low_tier_probability(m.predict_proba(Xte), LOW_TIER_MAX)
    lv, lt = (yv <= LOW_TIER_MAX).astype(int), (yte <= LOW_TIER_MAX).astype(int)
    rng = np.random.default_rng(0)
    half = rng.permutation(len(pv)) < len(pv) // 2
    raw_e, iso_e = [], []
    for a, b in ((half, ~half), (~half, half)):
        iso = IsotonicRegression(out_of_bounds="clip").fit(pv[a], lv[a])
        raw_e.append(expected_calibration_error(lv[b], pv[b]))
        iso_e.append(expected_calibration_error(lv[b], iso.predict(pv[b])))
    iso_full = IsotonicRegression(out_of_bounds="clip").fit(pv, lv)
    return {
        "val_ece_raw": float(np.mean(raw_e)), "val_ece_isotonic": float(np.mean(iso_e)),
        "test_ece_raw": expected_calibration_error(lt, pt),
        "test_ece_isotonic": expected_calibration_error(lt, iso_full.predict(pt)),
        "shipped": "isotonic" if np.mean(iso_e) < 0.8 * np.mean(raw_e) else "raw",
    }


def run(out_dir: Path, assets_dir: Path | None, quick: bool = False, seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    dedup = drop_duplicates(raw)
    parts = split(dedup, seed=seed)
    Xtr, ytr = prepare(parts["train"])
    Xva, yva = prepare(parts["val"])
    Xte, yte = prepare(parts["test"])
    names = list(REGISTRY)
    if quick:
        names = ["mode_only", "ordinal_logit", "multinomial_logit"]

    # --- cross-validated selection on the training fold only
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1 if quick else 3, random_state=seed)
    folds = list(rskf.split(Xtr, ytr))
    cv = {n: cv_scores(n, Xtr, ytr, folds) for n in names}
    pool = [n for n in SHIP_POOL if n in names]
    shipped = one_se_pick(cv, pool)

    # --- validation (fit on train) and test (fit on train + val)
    val_tab, test_tab, test_preds = {}, {}, {}
    Xtv = pd.concat([Xtr, Xva]).reset_index(drop=True)
    ytv = np.concatenate([ytr, yva])
    for n in names:
        _, pv = _fit_eval(n, Xtr, Xva, ytr, yva)
        val_tab[n] = grade_metrics(yva, pv)
        m, pt = _fit_eval(n, Xtv, Xte, ytv, yte)
        test_tab[n] = grade_metrics(yte, pt)
        test_preds[n] = pt
        if n == shipped:
            final = m
    pred = test_preds[shipped]
    ci = {
        "qwk": bootstrap_ci(qwk, yte, pred, n=300 if quick else 1000),
        "mae": bootstrap_ci(lambda a, b: float(np.abs(a - b).mean()), yte, pred,
                            n=300 if quick else 1000),
    }
    proba = final.predict_proba(Xte)
    p_low = low_tier_probability(proba, LOW_TIER_MAX)
    is_low = yte <= LOW_TIER_MAX

    results: dict = {
        "version": __version__,
        "rows": {"raw": int(len(raw)), "deduped": int(len(dedup)),
                 **{k: len(v) for k, v in parts.items()}},
        "low_tier_rate": float(is_low.mean()),
        "shipped": shipped,
        "cv": {n: {"qwk_mean": float(v["qwk"].mean()),
                   "qwk_se": float(v["qwk"].std(ddof=1) / np.sqrt(len(v["qwk"]))),
                   "mae_mean": float(v["mae"].mean()),
                   "rps_mean": None if np.isnan(v["rps"]).all() else float(np.nanmean(v["rps"]))}
               for n, v in cv.items()},
        "validation": val_tab,
        "test": test_tab,
        "test_shipped_ci": ci,
        "test_rps": ranked_probability_score(yte, proba),
    }

    # --- agreement between model and panel (ordinal Krippendorff alpha)
    results["alpha_model_vs_panel"] = float(krippendorff.alpha(
        reliability_data=np.vstack([yte, pred]), level_of_measurement="ordinal"))

    # --- decoding study on the shipped model
    results["decoding"] = {
        "argmax": grade_metrics(yte, decode_argmax(proba)),
        "median": grade_metrics(yte, decode_median(proba)),
        "rounded_expectation": grade_metrics(yte, np.rint(expected_grade(proba)).astype(int)),
    }

    # --- McNemar: shipped vs multinomial logit on exact-grade correctness
    if "multinomial_logit" in test_preds and shipped != "multinomial_logit":
        tb = mcnemar_table(yte, pred, test_preds["multinomial_logit"])
        chi2, p = mcnemar(tb, corrected=True)
        results["mcnemar_vs_multinomial"] = {"chi2": float(chi2), "p": float(p),
                                              "table": tb.tolist()}

    # --- Friedman + Nemenyi on fold-level kappa
    if len(names) > 3:
        mat = np.column_stack([cv[n]["qwk"] for n in names])
        stat, p = friedmanchisquare(*mat.T)
        nem = sp.posthoc_nemenyi_friedman(mat)
        nem.index = nem.columns = names
        results["friedman"] = {"stat": float(stat), "p": float(p)}
        results["nemenyi_p"] = {a: {b: float(nem.loc[a, b]) for b in names} for a in names}

    # --- calibration, triage economics
    results["calibration"] = calibration_study(shipped, parts)
    costs = Costs()
    p_val = low_tier_probability(
        build(shipped).fit(Xtr, ytr).predict_proba(Xva), LOW_TIER_MAX)
    t_val = best_threshold(p_val, yva <= LOW_TIER_MAX, costs)
    alcohol_rank = -parts["test"]["alcohol"].to_numpy()
    n = len(yte)
    pol = {
        "review_none": policy_cost(np.zeros(n, bool), is_low, costs),
        "review_all": policy_cost(np.ones(n, bool), is_low, costs),
        "model_break_even": policy_cost(p_low >= costs.break_even, is_low, costs),
        "model_val_tuned": policy_cost(p_low >= t_val, is_low, costs),
        "hindsight": hindsight_cost(is_low, costs),
    }
    share = float((p_low >= t_val).mean())
    rng = np.random.default_rng(seed)
    pol["random_same_share"] = float(np.mean([
        policy_cost(rng.permutation(n) < round(share * n), is_low, costs) for _ in range(200)]))
    k = int(round(share * n))
    alc_mask = np.zeros(n, bool)
    alc_mask[np.argsort(-alcohol_rank, kind="stable")[:k]] = True
    pol["alcohol_same_share"] = policy_cost(alc_mask, is_low, costs)
    saving = np.where(p_low >= t_val, costs.review * -1 + costs.missed_low * is_low, 0.0)
    # per-lot saving relative to review-all
    per_lot = np.where(p_low >= t_val, 0.0, costs.review - costs.missed_low * is_low)
    results["triage"] = {
        "assumptions": {"review": costs.review, "missed_low": costs.missed_low,
                        "break_even_probability": costs.break_even},
        "threshold_val_tuned": t_val, "review_share": share,
        "cost_total": pol, "cost_per_lot": {k_: v / n for k_, v in pol.items()},
        "saving_vs_review_all_per_lot": float(per_lot.mean()),
        "saving_ci": bootstrap_ci(lambda a: float(a.mean()), per_lot, n=300 if quick else 1000),
        "low_found_share": float(is_low[p_low >= t_val].sum() / is_low.sum()),
        "budget_recall_20": {"model": budget_recall(p_low, is_low, 0.2),
                             "alcohol": budget_recall(alcohol_rank, is_low, 0.2),
                             "random": 0.2},
    }
    del saving
    sweep = {}
    for rr in (0.1, 0.2, 0.25, 0.4, 0.6, 0.8):
        c = Costs(review=costs.missed_low * rr, missed_low=costs.missed_low)
        sweep[str(rr)] = {"model": policy_cost(p_low >= rr, is_low, c) / n,
                          "review_all": policy_cost(np.ones(n, bool), is_low, c) / n,
                          "review_none": policy_cost(np.zeros(n, bool), is_low, c) / n}
    results["triage"]["ratio_sweep"] = sweep

    if not quick:
        results["leakage"] = leakage_study(raw)
        results["colour"] = colour_study(dedup, shipped)
        imp = global_importance(final, Xte, yte, qwk, n_repeats=5)
        results["permutation_importance_qwk"] = imp

    # --- bundle and outputs
    bundle = {"model": final, "features": FEATURES,
              "reference": Xtv.median().to_dict(),
              "threshold": t_val, "costs": costs.__dict__, "name": shipped,
              "ranges": {c: [float(Xtv[c].min()), float(Xtv[c].max())] for c in FEATURES}}
    joblib.dump(bundle, out_dir / "gradeline_model.joblib", compress=3)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    preds = parts["test"].copy()
    preds["pred_class"] = pred
    preds["p_low_tier"] = p_low
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    if assets_dir is not None:
        from .plots import make_charts
        assets_dir.mkdir(parents=True, exist_ok=True)
        make_charts(results, yte, pred, p_low, is_low, assets_dir)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Train gradeline and write metrics and charts.")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--quick", action="store_true", help="small run for smoke tests")
    a = ap.parse_args()
    res = run(Path(a.out), Path(a.assets), quick=a.quick)
    print(json.dumps({"shipped": res["shipped"], "test": res["test"][res["shipped"]]}, indent=2))


if __name__ == "__main__":
    main()
