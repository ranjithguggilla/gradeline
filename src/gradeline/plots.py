"""Charts used in the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import GRADE_LABELS  # noqa: E402

BLUE, GREY, RED, GREEN = "#2f5d8a", "#9aa5b1", "#b5483a", "#4d8f5b"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25)


def make_charts(res, y, pred, p_low, is_low, out: Path) -> None:
    out = Path(out)
    names = list(res["cv"])
    order = sorted(names, key=lambda n: res["cv"][n]["qwk_mean"])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cols = [GREEN if n == res["shipped"] else GREY for n in order]
    ax.barh(order, [res["cv"][n]["qwk_mean"] for n in order],
            xerr=[res["cv"][n]["qwk_se"] for n in order], color=cols)
    ax.set_xlabel("cross-validated quadratic weighted kappa (train folds)")
    ax.set_title("Model comparison, shipped model in green")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)

    k = len(GRADE_LABELS)
    cm = np.zeros((k, k), int)
    for a, b in zip(y, pred, strict=True):
        cm[a, b] += 1
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax.set_xticks(range(k), GRADE_LABELS)
    ax.set_yticks(range(k), GRADE_LABELS)
    ax.set_xlabel("predicted grade")
    ax.set_ylabel("panel grade")
    ax.set_title("Test lots, counts (colour is row share)")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=150)
    plt.close(fig)

    t = res["triage"]
    labels = ["review none", "review all", "model, break-even cut", "model, tuned cut",
              "random, same share", "alcohol, same share", "hindsight"]
    keys = ["review_none", "review_all", "model_break_even", "model_val_tuned",
            "random_same_share", "alcohol_same_share", "hindsight"]
    vals = [t["cost_per_lot"][k_] for k_ in keys]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(labels[::-1], vals[::-1],
            color=[GREEN if "tuned" in lab else GREY for lab in labels[::-1]])
    ax.set_xlabel("expected cost per lot (USD, stated assumptions)")
    ax.set_title("Panel triage cost on the test fold")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "triage_cost.png", dpi=150)
    plt.close(fig)

    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p_low, bins[1:-1]), 0, 9)
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 5:
            xs.append(p_low[m].mean())
            ys.append(is_low[m].mean())
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--")
    ax.plot(xs, ys, "o-", color=BLUE)
    ax.set_xlabel("predicted probability of grade 5 or below")
    ax.set_ylabel("observed share")
    ax.set_title("Low-tier reliability, test fold")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)

    if "leakage" in res:
        lk = res["leakage"]
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.bar(["raw files,\nrandom split", "repeats removed,\nrandom split"],
               [lk["raw_random"], lk["deduped_random"]], color=[RED, GREEN])
        ax.set_ylabel("test quadratic weighted kappa")
        ax.set_title("Repeated rows inflate the score")
        fig.tight_layout()
        fig.savefig(out / "duplicate_leak.png", dpi=150)
        plt.close(fig)
