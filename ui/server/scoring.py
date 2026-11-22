"""Score an existing prediction for explanations using the same rules as the UI."""
from __future__ import annotations

from .schema import ScoringRule


def rank(rows: list[dict], rule: ScoringRule) -> list[dict]:
    weights = rule.udi + rule.f300 + rule.p10
    if weights != 100:
        raise ValueError("采光指标权重的总和必须为 100%。")
    daylight = [
        row["F_UDI_fraction"] * rule.udi
        + row["F_300_fraction"] * rule.f300
        + min(row["E_P10_lux"] / rule.p10_target, 1) * rule.p10
        for row in rows
    ]
    top_daylight = max(daylight, default=0)
    top_pv = max((row["PV_AC_kWh"] for row in rows), default=0)
    scores = [
        (score / top_daylight if top_daylight else 0) * (100 - rule.power)
        + (row["PV_AC_kWh"] / top_pv if top_pv else 0) * rule.power
        for score, row in zip(daylight, rows)
    ]
    best = max(scores, default=0)
    return [
        {**row, "daylight_score": score, "preference_score": total,
         "recommended": abs(best - total) <= 1e-9}
        for row, score, total in zip(rows, daylight, scores)
    ]
