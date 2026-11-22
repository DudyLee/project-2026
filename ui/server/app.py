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
from .schema import ChatRequest, Conditions, ExplainRequest
from .scoring import rank

logger = logging.getLogger(__name__)
PREDICTION_LOCK = Lock()
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
    return {
        "prediction_ready": app.state.model is not None,
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
    return await clarify(body.message, body.draft, app.state.catalog, app.state.agents, today, body.language)


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
        "weather_source": "original_simulation_weather",
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
    facts = {"conditions": body.conditions.model_dump(), "angles": scored,
             "rule": body.rule.model_dump() if body.rule else None}
    language = language_code(body.language) or "en"
    summary = await app.state.agents.explain(facts, body.question, language)
    if summary:
        return {"text": summary, "source": "agent", "language": language}
    power = max(scored, key=lambda row: row["PV_AC_kWh"])
    udi = max(scored, key=lambda row: row["F_UDI_fraction"])
    recommended = [str(row["PVSD_Angle_deg"]) + "°" for row in scored if row.get("recommended")]
    pv_wh = 1000 * power["PV_AC_kWh"]
    fraction = 100 * udi["F_UDI_fraction"]
    if language == "zh":
        text = (f"这个时段，{power['PVSD_Angle_deg']}° 的预测发电量最高（{pv_wh:.2f} Wh），"
                f"{udi['PVSD_Angle_deg']}° 的有效照度比例最高（{fraction:.1f}%）。")
        if recommended:
            text += f"按照当前规则，推荐 {'、'.join(recommended)}。"
    else:
        text = (f"For this hour, {power['PVSD_Angle_deg']}° has the highest predicted PV energy ({pv_wh:.2f} Wh), "
                f"and {udi['PVSD_Angle_deg']}° has the highest useful daylight fraction ({fraction:.1f}%).")
        if recommended:
            text += f" The current rule recommends {', '.join(recommended)}."
    return {"text": text, "source": "local", "language": "zh" if language == "zh" else "en"}



dist = UI_ROOT / "dist"
if (dist / "index.html").is_file():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(dist / "index.html")
