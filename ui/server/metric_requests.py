"""Resolve metric scopes and thresholds into model units."""
from __future__ import annotations

import re

from pydantic import ValidationError

from .schema import Metric, MetricFilter, MetricSort

EXPLICIT_COVERAGE = (
    r"(?:coverage|share of points)\s+(?:at or above|at|above|over)\s+300\s*(?:lux|lx)"
    r"|300\s*(?:lux|lx)\s*(?:coverage|达标(?:比例)?)"
    r"|of (?:the )?(?:room|(?:sampling )?points)\s+(?:at or above|at|above|over)\s+300\s*(?:lux|lx)"
)
METRICS = {
    "E_mean_lux": r"mean illuminance|average (?:illuminance|light)|平均照度",
    "E_P10_lux": r"(?<![a-z0-9])p10(?![a-z0-9])|10th percentile|第\s*10\s*百分位",
    "F_300_fraction": rf"(?<![a-z0-9])f300(?![a-z0-9])|{EXPLICIT_COVERAGE}|300\s*lux|达标比例",
    "F_UDI_fraction": r"(?<![a-z0-9])udi(?![a-z0-9])|useful daylight|有效(?:照度|采光)",
    "PV_AC_kWh": r"\bpv\b|electricity|solar (?:energy|generation|power)|power generation|发电|光伏",
}
LABELS = {
    "E_mean_lux": ("Mean illuminance", "平均照度", 1, "lux"),
    "E_P10_lux": ("P10 illuminance", "P10 照度", 1, "lux"),
    "F_300_fraction": ("300 lux coverage", "300 lux 达标比例", 100, "%"),
    "F_UDI_fraction": ("Useful daylight", "有效照度比例", 100, "%"),
    "PV_AC_kWh": ("PV energy", "光伏发电量", 1000, "Wh"),
}

OPERATORS = {
    "gte": r">=|≥|\bat least\b|\bno less than\b|\bat or above\b|至少|不低于|不少于|大于等于|达到",
    "lte": r"<=|≤|\bat most\b|\bno more than\b|\bat or below\b|至多|不高于|不超过|小于等于",
    "gt": r">|\babove\b|\bover\b|\bmore than\b|\bgreater than\b|高于|超过|大于",
    "lt": r"<|\bbelow\b|\bunder\b|\bless than\b|低于|小于|少于",
}
OPERATOR = '|'.join(f"(?P<{name}>{pattern})" for name, pattern in OPERATORS.items())
LIMIT = re.compile(rf"(?:{OPERATOR})\s*(?P<value>-?\d+(?:\.\d+)?)"
                   r"(?![\d.]|\s*(?:°|deg(?:ree)?s?\b|度|[ap]\.?m\b|[:点时]))\s*"
                   r"(?P<unit>%|percent\b|lux\b|lx\b|kwh\b|wh\b|勒克斯|千瓦时|瓦时)?")
CLAUSE = re.compile(r"\b(?:with|where|when|if|and|or|having|require(?:s|d)?|requiring|only consider|(?:sorted|ranked|ordered) by)\b|[,，;；]|并且|且|或者|满足|要求|仅考虑|只考虑")
REQUIRED_LIMIT = re.compile(r"\bat least\b|\bat most\b|\bno (?:less|more) than\b|[<>≤≥]|至少|至多|不低于|不高于|不少于|不超过")
SORT_BY = re.compile(r"\b(?:rank|sort|order)(?:ed)?\b.*?\bby\b|按")
HIGH = r"\b(?:highest|most|maximum|maximize|descending)\b|最高|最多|最大|降序"
LOW = r"\b(?:lowest|least|minimum|minimize|ascending)\b|最低|最少|最小|升序"

UNITS = {
    "%": ({"F_300_fraction", "F_UDI_fraction"}, 100),
    "percent": ({"F_300_fraction", "F_UDI_fraction"}, 100),
    "lux": ({"E_mean_lux", "E_P10_lux"}, 1),
    "lx": ({"E_mean_lux", "E_P10_lux"}, 1),
    "勒克斯": ({"E_mean_lux", "E_P10_lux"}, 1),
    "wh": ({"PV_AC_kWh"}, 1000),
    "kwh": ({"PV_AC_kWh"}, 1),
    "瓦时": ({"PV_AC_kWh"}, 1000),
    "千瓦时": ({"PV_AC_kWh"}, 1),
}


def mentioned_metrics(text: str) -> list[Metric]:
    return [metric for metric, pattern in METRICS.items() if re.search(pattern, text)]


def scoped_metrics(text: str) -> list[Metric]:
    explicit = mentioned_metrics(text)
    remaining = re.sub(METRICS["F_UDI_fraction"], "", text)
    if re.search(r"\bday\s*light(?:ing)?\b(?!\s+(?:score|first|priority|preference))|采光(?!分|优先)", remaining):
        return [metric for metric in METRICS if metric != "PV_AC_kWh" or metric in explicit]
    return explicit


def ranking_request(text: str) -> dict | None:
    normalized = re.sub(EXPLICIT_COVERAGE, " f300 ", text)
    limits = list(LIMIT.finditer(normalized))
    leftover = LIMIT.sub("", normalized)
    missing_limit = bool(REQUIRED_LIMIT.search(leftover) or re.search(
        r"\b(?:between|exactly)\s+\d+(?:\.\d+)?\s*(?:%|percent|lux|lx|wh|kwh)", leftover
    ))
    sort_match = SORT_BY.search(normalized)
    objective = CLAUSE.split(normalized[sort_match.end():] if sort_match
                             else normalized[:limits[0].start()] if limits else normalized, maxsplit=1)[0]
    lower = re.search(LOW, objective) or (sort_match and re.search(r"ascending|lowest first|升序", normalized))
    higher = re.search(HIGH, objective)
    metric_names = mentioned_metrics(objective)
    rank_intent = bool(sort_match or lower or higher)
    if not limits and not missing_limit and not (rank_intent and metric_names):
        return None
    has_metric_context = mentioned_metrics(normalized) or re.search(r"\b(?:angles?|louvers?|coverage)\b|角度|百叶|覆盖", normalized)
    if not has_metric_context and not any(limit["unit"] in {"wh", "kwh", "瓦时", "千瓦时"} for limit in limits):
        return None
    if missing_limit:
        return {"ranking_issue": "filter"}
    if rank_intent and len(metric_names) != 1:
        return {"ranking_issue": "sort"}
    sort = MetricSort(metric=metric_names[0], direction="asc" if lower else "desc") if rank_intent else None

    filters = []
    if len(limits) > 5 or (limits and re.search(r"\bor\b|或者|或", leftover)):
        return {"ranking_issue": "filter"}
    for index, limit in enumerate(limits):
        unit = UNITS.get(limit["unit"])
        if not unit:
            return {"ranking_issue": "filter"}
        accepted, divisor = unit
        start = limits[index - 1].end() if index else 0
        end = limits[index + 1].start() if index + 1 < len(limits) else len(normalized)
        before = CLAUSE.split(normalized[start:limit.start()])[-1]
        after = CLAUSE.split(normalized[limit.end():end], maxsplit=1)[0]
        possible = [metric for metric in mentioned_metrics(before) if metric in accepted]
        possible = possible or [metric for metric in mentioned_metrics(after) if metric in accepted]
        if not possible and len(accepted) == 1 and not mentioned_metrics(before + " " + after):
            possible = list(accepted)
        if len(possible) != 1:
            return {"ranking_issue": "filter"}
        operator = next(name for name in OPERATORS if limit[name] is not None)
        try:
            filters.append(MetricFilter(metric=possible[0], operator=operator, value=float(limit["value"]) / divisor))
        except ValidationError:
            return {"ranking_issue": "filter"}
    return {"filters": [item.model_dump() for item in filters], "sort": sort.model_dump() if sort else None}
