from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    draft: Draft = Field(default_factory=Draft)
    language: str = Field(default="en", max_length=40)


class Intent(BaseModel):
    room_query: str | None = Field(default=None, max_length=100)
    floor: int | None = Field(default=None, ge=0, le=100)
    days: list[int] = Field(default_factory=list, max_length=5)
    hours: list[int] = Field(default_factory=list, max_length=5)
    needs_day: bool = False
    needs_hour: bool = False
    preference: Literal["power", "daylight", "balanced"] | None = None
    explain: bool = False
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
    p10_target: int = Field(strict=True, ge=1, le=10000)


class ExplainRequest(BaseModel):
    conditions: Conditions
    rule: ScoringRule | None = None
    question: str = Field(default="Explain the recommended angle.", min_length=1, max_length=1000)
    language: str = Field(default="en", max_length=40)
