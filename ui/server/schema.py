from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Preference = Literal["power", "daylight", "balanced", "custom"]
Angle = Literal[0, 30, 60, 90]


class Conditions(BaseModel):
    room_id: str = Field(min_length=1, max_length=30)
    day: int = Field(strict=True, ge=1, le=365)
    hour: int = Field(strict=True, ge=7, le=18)


class Draft(BaseModel):
    room_id: str | None = Field(default=None, max_length=30)
    day: int | None = Field(default=None, strict=True, ge=1, le=365)
    hour: int | None = Field(default=None, strict=True, ge=7, le=18)
    preference: Preference | None = None


class ResultRef(Conditions):
    id: str = Field(min_length=1, max_length=40)


Action = Literal["predict", "explain", "rule", "compare", "rank", "chart", "export", "help"]
DaylightMetric = Literal["E_mean_lux", "E_P10_lux", "F_300_fraction", "F_UDI_fraction"]
Metric = Literal["E_mean_lux", "E_P10_lux", "F_300_fraction", "F_UDI_fraction", "PV_AC_kWh"]
Topic = Literal["general", "p10", "udi", "weather", "hours", "scoring", "rooms", "coverage", "metrics"]


class MetricFilter(BaseModel):
    metric: Metric
    operator: Literal["gte", "gt", "lte", "lt"]
    value: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_fraction(self) -> MetricFilter:
        if self.metric in {"F_300_fraction", "F_UDI_fraction"} and self.value > 1:
            raise ValueError("Fraction thresholds must be between zero and one")
        return self


class DaylightCutoff(MetricFilter):
    metric: DaylightMetric


class MetricSort(BaseModel):
    metric: Metric
    direction: Literal["asc", "desc"]


class RuleEdit(BaseModel):
    power: int | None = None
    udi: int | None = None
    f300: int | None = None
    p10: int | None = None
    p10_target: float | None = None
    mean: int | None = None
    mean_target: float | None = None
    cutoffs: list[DaylightCutoff] | None = Field(default=None, max_length=5)
    remove_cutoffs: list[DaylightMetric] | None = Field(default=None, max_length=4)
    clear_cutoffs: bool | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    draft: Draft = Field(default_factory=Draft)
    language: str = Field(default="en", max_length=40)
    results: list[ResultRef] = Field(default_factory=list, max_length=20)
    active_result_id: str | None = Field(default=None, max_length=40)


class Intent(BaseModel):
    room_query: str | None = Field(default=None, max_length=100)
    floor: int | None = Field(default=None, ge=0, le=100)
    days: list[int] = Field(default_factory=list, max_length=5)
    hours: list[int] = Field(default_factory=list, max_length=5)
    needs_day: bool = False
    needs_hour: bool = False
    preference: Literal["power", "daylight", "balanced"] | None = None
    explain: bool = False
    action: Action | None = None
    result_id: str | None = Field(default=None, max_length=40)
    against_result_id: str | None = Field(default=None, max_length=40)
    angles: list[int] = Field(default_factory=list, max_length=4)
    rule: RuleEdit | None = None
    rule_issue: Literal["cutoff"] | None = None
    metrics: list[Metric] = Field(default_factory=list, max_length=5)
    filters: list[MetricFilter] = Field(default_factory=list, max_length=5)
    sort: MetricSort | None = None
    ranking_issue: Literal["filter", "sort"] | None = None
    visibility: Literal["only", "show", "hide"] | None = None
    table_open: bool | None = None
    format: Literal["csv", "json"] | None = None
    topic: Topic | None = None
    reply_language: str | None = Field(default=None, max_length=40)
    language_only: bool = False


class RoomSelection(BaseModel):
    room_ids: list[str] = Field(default_factory=list, max_length=6)


class ExplanationRefs(BaseModel):
    summary: str = Field(min_length=1, max_length=550)


class ScoringRule(BaseModel):
    power: int = Field(strict=True, ge=0, le=100)
    udi: int = Field(strict=True, ge=0, le=100)
    f300: int = Field(strict=True, ge=0, le=100)
    p10: int = Field(strict=True, ge=0, le=100)
    p10_target: float | None = Field(default=None, gt=0, le=10000)
    mean: int = Field(default=0, strict=True, ge=0, le=100)
    mean_target: float | None = Field(default=None, gt=0, le=10000)
    cutoffs: list[DaylightCutoff] = Field(default_factory=list, max_length=5)


class ExplainRequest(BaseModel):
    conditions: Conditions
    rule: ScoringRule | None = None
    angles: list[Angle] = Field(default_factory=list, max_length=4)
    metrics: list[Metric] = Field(default_factory=list, max_length=5)
    question: str = Field(default="Explain the recommended angle.", min_length=1, max_length=1000)
    language: str = Field(default="en", max_length=40)
