"""Local API for the Tower A conversation and four-angle inference."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
import logging
import os
from pathlib import Path
import sys
from threading import Lock
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agents import Agents
from .catalog import Catalog, DATASET, UI_ROOT
from .conversation import clarify
from .language import language_code
from .metric_requests import LABELS
from .schema import ChatRequest, Conditions, ExplainRequest
from .scoring import rank

logger = logging.getLogger(__name__)
PREDICTION_LOCK = Lock()
WEATHER_SOURCE = "original_simulation_weather"
PROJECT = UI_ROOT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


@asynccontextmanager
async def lifespan(application: FastAPI):
    directory = Path(os.environ.get("PVSD_DATASET") or DATASET)
    application.state.catalog = Catalog(directory)
    application.state.agents = Agents()
    application.state.model = None
    try:
        from pvsd_trial import TrialPredictor

        application.state.model = TrialPredictor(version="0.2")
    except (ImportError, OSError, RuntimeError, ValueError):
        logger.exception("Trial predictor could not load")
    prediction.cache_clear()
    yield
    prediction.cache_clear()


app = FastAPI(title="Tower A Louver Predictions", lifespan=lifespan)


@app.get("/api/health")
def health():
    model = app.state.model
    return {
        "prediction_ready": model is not None,
        "model_version": model.version if model is not None else None,
        "weather_source": WEATHER_SOURCE,
        "agents_available": app.state.agents.enabled,
        "weather_hours": len(app.state.catalog.weather),
        "room_count": len(app.state.catalog.rooms),
    }


@app.get("/api/rooms")
def rooms(q: str = Query(default="", max_length=100), floor: int | None = Query(default=None, ge=0, le=100)):
    catalog: Catalog = app.state.catalog
    return {"rooms": [room.card() for room in catalog.search(q, floor)]}


@app.post("/api/chat")
async def chat(body: ChatRequest):
    today = datetime.now(ZoneInfo("Asia/Hong_Kong")).date()
    return await clarify(body.message, body.draft, app.state.catalog, app.state.agents, today, body.language,
                         body.results, body.active_result_id)


@lru_cache(maxsize=128)
def prediction(room_id: str, day: int, hour: int) -> dict:
    catalog: Catalog = app.state.catalog
    if room_id not in catalog.rooms:
        raise HTTPException(status_code=404, detail="roomError")
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="modelError")
    if (day, hour) not in catalog.weather:
        raise HTTPException(status_code=404, detail="weatherError")
    # threadpool_limits in TrialPredictor affects process-wide numerical libraries.
    with PREDICTION_LOCK:
        frame = app.state.model.predict([catalog.inputs(room_id, day, hour)])
    return {
        "room": catalog.rooms[room_id].card(),
        "day": day,
        "hour": hour,
        "weather_source": WEATHER_SOURCE,
        "version": frame["Model_Version"].iat[0],
        "model_sha256": frame["Model_SHA256"].iat[0],
        "coverage": frame["Coverage_Flags"].iat[0].split(";"),
        "rows": frame[["PVSD_Angle_deg", "E_mean_lux", "E_P10_lux",
                       "F_300_fraction", "F_UDI_fraction", "PV_AC_kWh"]].to_dict("records"),
    }


@app.post("/api/predict")
def predict(body: Conditions):
    return prediction(body.room_id.strip().upper(), body.day, body.hour)


@app.post("/api/explain")
async def explain(body: ExplainRequest):
    from starlette.concurrency import run_in_threadpool

    result = await run_in_threadpool(predict, body.conditions)
    try:
        scored = rank(result["rows"], body.rule) if body.rule else result["rows"]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="scoreError") from exc
    facts = {"requested_angles": body.angles, "requested_metrics": body.metrics, "conditions": body.conditions.model_dump(), "angles": scored,
             "rule": body.rule.model_dump() if body.rule else None}
    language = language_code(body.language) or "en"
    summary = await app.state.agents.explain(facts, body.question, language)
    if summary:
        return {"text": summary, "source": "agent", "language": language}
    if body.metrics:
        details = describe_metrics(scored, body, language)
        return {"text": details, "source": "local", "language": "zh" if language == "zh" else "en"}
    if body.angles:
        details = []
        for row in scored:
            angle = row["PVSD_Angle_deg"]
            if angle not in body.angles:
                continue
            pv = row["PV_AC_kWh"] * 1000
            p10 = row["E_P10_lux"]
            udi = row["F_UDI_fraction"] * 100
            if language == "zh":
                detail = f"{angle}° 的预测发电量为 {pv:.2f} Wh，有效照度比例为 {udi:.1f}%，P10 为 {p10:.1f} lux。"
                if row.get("preference_score") is not None:
                    detail += f"当前偏好评分为 {row['preference_score']:.2f}/100。"
            else:
                detail = (f"At {angle}°, predicted PV energy is {pv:.2f} Wh, useful daylight "
                          f"is {udi:.1f}%, and P10 is {p10:.1f} lux.")
                if row.get("preference_score") is not None:
                    detail += f" The current preference score is {row['preference_score']:.2f}/100."
            details.append(detail)
        return {"text": " ".join(details), "source": "local", "language": "zh" if language == "zh" else "en"}
    eligible = [row for row in scored if row.get("eligible", True)]
    if not eligible:
        text = "没有角度满足全部限值，因此没有推荐角度。" if language == "zh" else "No angle meets every cutoff. The rule yields no recommended angle."
        return {"text": text, "source": "local", "language": "zh" if language == "zh" else "en"}
    power = max(eligible, key=lambda row: row["PV_AC_kWh"])
    udi = max(eligible, key=lambda row: row["F_UDI_fraction"])
    recommended = [str(row["PVSD_Angle_deg"]) + "°" for row in eligible if row.get("recommended")]
    top_score = max((row.get("preference_score", 0) for row in eligible), default=0)
    pv_wh = 1000 * power["PV_AC_kWh"]
    fraction = 100 * udi["F_UDI_fraction"]
    if language == "zh":
        prefix = "在符合限值的角度中，" if body.rule and body.rule.cutoffs else "这个时段，"
        text = (f"{prefix}{power['PVSD_Angle_deg']}° 的预测发电量最高（{pv_wh:.2f} Wh），"
                f"{udi['PVSD_Angle_deg']}° 的有效照度比例最高（{fraction:.1f}%）。")
        if body.rule and recommended:
            text += (f"当前偏好按 {body.rule.power}% 发电、{100 - body.rule.power}% 采光计分，"
                     f"{'、'.join(recommended)} 得分最高（{top_score:.2f}/100）。")
    else:
        prefix = "Among qualifying angles" if body.rule and body.rule.cutoffs else "For this hour"
        text = (f"{prefix}, {power['PVSD_Angle_deg']}° has the highest predicted PV energy ({pv_wh:.2f} Wh), "
                f"and {udi['PVSD_Angle_deg']}° has the highest useful daylight fraction ({fraction:.1f}%).")
        if body.rule and recommended:
            angles = ", ".join(recommended)
            text += (f" With {body.rule.power}% PV and {100 - body.rule.power}% daylight, "
                     f"the highest score is {top_score:.2f}/100 at {angles}.")
    return {"text": text, "source": "local", "language": "zh" if language == "zh" else "en"}



def describe_metrics(rows: list[dict], request: ExplainRequest, language: str) -> str:
    chinese = language == "zh"
    by_angle = {row["PVSD_Angle_deg"]: row for row in rows}
    selected = [by_angle[angle] for angle in dict.fromkeys(request.angles)]
    descriptions = []
    for metric in dict.fromkeys(request.metrics):
        english, chinese_label, scale, unit = LABELS[metric]
        label = chinese_label if chinese else english
        precision = 2 if unit == "Wh" else 1
        suffix = "%" if unit == "%" else " " + unit
        if selected:
            values = [(row["PVSD_Angle_deg"], row[metric] * scale) for row in selected]
            snippets = ([f"{angle}° 为 {value:.{precision}f}{suffix}" for angle, value in values] if chinese else
                        [f"{value:.{precision}f}{suffix} at {angle}°" for angle, value in values])
            summary = f"{label}在" + "，".join(snippets) + "。" if chinese else f"{label} is " + " and ".join(snippets) + "."
            if len(values) == 2:
                (start, first), (end, second) = values
                difference = second - first
                if abs(difference) < 0.5 * 10**-precision:
                    difference = 0
                change_unit = ("百分点" if chinese else "percentage points") if unit == "%" else unit
                summary += (f" 从 {start}° 到 {end}° 变化了 {difference:+.{precision}f} {change_unit}。" if chinese else
                            f" The change from {start}° to {end}° is {difference:+.{precision}f} {change_unit}.")
        else:
            best = max(row[metric] for row in rows)
            angles = ("、" if chinese else ", ").join(f"{row['PVSD_Angle_deg']}°" for row in rows if row[metric] == best)
            value = f"{best * scale:.{precision}f}{suffix}"
            summary = (f"{label}在 {angles} 最高，为 {value}。" if chinese else
                       f"{label} peaks at {value} for {angles}.")
        descriptions.append(summary)
    return " ".join(descriptions)


dist = UI_ROOT / "dist"
if (dist / "index.html").is_file():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(dist / "index.html")
