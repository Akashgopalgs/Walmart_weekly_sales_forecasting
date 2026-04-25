from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODELS_DIR
from src.forecasting_engine import forecast_store, load_package


STATE = {}


class ForecastRequest(BaseModel):
    store_id: int = Field(..., ge=1)
    target_date: Optional[str] = Field(None, description="Forecast up to this date")
    weeks_ahead: Optional[int] = Field(4, ge=1, le=52)
    model_name: Optional[str] = Field(None)
    temperature_avg: Optional[float] = None
    fuel_price: Optional[float] = None
    cpi: Optional[float] = None
    unemployment: Optional[float] = None
    markdown_total: Optional[float] = None


class ForecastItem(BaseModel):
    date: str
    store_id: int
    model_used: str
    prediction: float
    lower_95: float
    upper_95: float


class ForecastResponse(BaseModel):
    store_id: int
    model_used: str
    forecasts: List[ForecastItem]


@asynccontextmanager
async def lifespan(app: FastAPI):
    pkg = load_package()
    STATE["pkg"] = pkg
    yield
    STATE.clear()


app = FastAPI(
    title="Walmart Forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stores")
def stores():
    pkg = STATE["pkg"]
    return {"stores": sorted(list(pkg["store_histories"].keys()))}


@app.get("/leaderboard")
def leaderboard():
    pkg = STATE["pkg"]
    return pkg["leaderboard"].to_dict(orient="records")


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    pkg = STATE["pkg"]
    if req.store_id not in pkg["store_histories"]:
        raise HTTPException(status_code=404, detail="Unknown store_id")

    overrides = {}
    if req.temperature_avg is not None:
        overrides["Temperature"] = req.temperature_avg
    if req.fuel_price is not None:
        overrides["Fuel_Price"] = req.fuel_price
    if req.cpi is not None:
        overrides["CPI"] = req.cpi
    if req.unemployment is not None:
        overrides["Unemployment"] = req.unemployment
    if req.markdown_total is not None:
        overrides["markdown_total"] = req.markdown_total

    preds = forecast_store(
        pkg,
        store_id=req.store_id,
        target_date=req.target_date,
        weeks_ahead=req.weeks_ahead,
        model_name=req.model_name,
        overrides=overrides,
    )

    return ForecastResponse(
        store_id=req.store_id,
        model_used=preds[0]["model_used"] if preds else pkg["best_model_name"],
        forecasts=[ForecastItem(**p) for p in preds],
    )


@app.get("/forecast")
def forecast_get(store_id: int, weeks_ahead: int = 4, model_name: Optional[str] = None):
    pkg = STATE["pkg"]
    preds = forecast_store(pkg, store_id=store_id, weeks_ahead=weeks_ahead, model_name=model_name)
    return {"store_id": store_id, "forecasts": preds}