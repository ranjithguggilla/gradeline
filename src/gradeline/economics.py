"""Panel-triage economics. Dollar inputs are assumptions, not data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Costs:
    review: float = 100.0      # tasting panel review of one lot
    missed_low: float = 400.0  # an unreviewed low-tier lot reaching the shelf

    @property
    def break_even(self) -> float:
        return self.review / self.missed_low


def policy_cost(review_mask: np.ndarray, is_low: np.ndarray, costs: Costs) -> float:
    """Total cost when reviewed lots are graded correctly and the rest ship as-is."""
    review_mask = np.asarray(review_mask, bool)
    is_low = np.asarray(is_low, bool)
    missed = (is_low & ~review_mask).sum()
    return float(costs.review * review_mask.sum() + costs.missed_low * missed)


def best_threshold(p_low, is_low, costs: Costs, grid=None) -> float:
    grid = np.linspace(0.02, 0.9, 89) if grid is None else grid
    totals = [policy_cost(p_low >= t, is_low, costs) for t in grid]
    return float(grid[int(np.argmin(totals))])


def hindsight_cost(is_low, costs: Costs) -> float:
    return policy_cost(np.asarray(is_low, bool), is_low, costs)


def budget_recall(score, is_low, share: float) -> float:
    """Share of low-tier lots found when only the top ``share`` of lots are reviewed."""
    score, is_low = np.asarray(score), np.asarray(is_low, bool)
    k = int(round(share * len(score)))
    top = np.argsort(-score, kind="stable")[:k]
    return float(is_low[top].sum() / max(is_low.sum(), 1))
