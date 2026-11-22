"""Score an existing prediction for explanations using the same rules as the UI."""
from __future__ import annotations

import operator

from .schema import ScoringRule

COMPARISONS = {"gte": operator.ge, "gt": operator.gt, "lte": operator.le, "lt": operator.lt}


def weighted_lux(lux: float, weight: int, target: float | None) -> float:
    return weight * min(lux / target, 1) if target else 0


def rank(rows: list[dict], rule: ScoringRule) -> list[dict]:
    if rule.udi + rule.f300 + rule.p10 + rule.mean != 100:
        raise ValueError("Daylight weights must sum to 100%.")
    if (rule.p10 and rule.p10_target is None) or (rule.mean and rule.mean_target is None):
        raise ValueError("Weighted illuminance metrics need a lux target.")
    daylight = [
        row["F_UDI_fraction"] * rule.udi
        + row["F_300_fraction"] * rule.f300
        + (weighted_lux(row["E_P10_lux"], rule.p10, rule.p10_target) if rule.p10 else 0)
        + (weighted_lux(row["E_mean_lux"], rule.mean, rule.mean_target) if rule.mean else 0)
        for row in rows
    ]
    eligible = [all(COMPARISONS[cutoff.operator](row[cutoff.metric], cutoff.value) for cutoff in rule.cutoffs) for row in rows]
    top_daylight = max((score for score, allowed in zip(daylight, eligible) if allowed), default=0)
    top_pv = max((row["PV_AC_kWh"] for row, allowed in zip(rows, eligible) if allowed), default=0)
    scores = [
        ((score / top_daylight if top_daylight else 0) * (100 - rule.power)
         + (row["PV_AC_kWh"] / top_pv if top_pv else 0) * rule.power) if allowed else None
        for score, row, allowed in zip(daylight, rows, eligible)
    ]
    best = max((score for score in scores if score is not None), default=0)
    return [
        {**row, "daylight_score": score, "preference_score": total, "eligible": allowed,
         "recommended": total is not None and abs(best - total) <= 1e-9}
        for row, score, total, allowed in zip(rows, daylight, scores, eligible)
    ]
