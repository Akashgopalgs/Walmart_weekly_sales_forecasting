from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, RETRAIN_THRESHOLD_WMAPE, N_TEST_WEEKS
from .features import build_next_row, feature_columns
from .metrics import wmape


def load_package(model_path: Path | None = None):
    model_path = model_path or (MODELS_DIR / "model_package.joblib")
    return joblib.load(model_path)


def build_future_steps(last_date: pd.Timestamp, target_date: str | None = None, weeks_ahead: int | None = None) -> list[pd.Timestamp]:
    if target_date is not None:
        tgt = pd.to_datetime(target_date)
        steps = max(1, int(np.ceil((tgt - last_date).days / 7)))
    else:
        steps = int(weeks_ahead or 1)

    return [last_date + pd.Timedelta(weeks=i) for i in range(1, steps + 1)]


def forecast_store(
    package: dict,
    store_id: int,
    target_date: str | None = None,
    weeks_ahead: int | None = None,
    model_name: str | None = None,
    overrides: dict | None = None,
) -> list[dict]:
    overrides = overrides or {}
    models = package["models"]
    chosen_name = model_name or package["best_model_name"]
    model = models[chosen_name]

    history = package["store_histories"][int(store_id)].copy().sort_values("ds").reset_index(drop=True)
    store_meta = package["store_meta"][int(store_id)].copy()

    last_date = pd.Timestamp(history["ds"].iloc[-1])
    future_dates = build_future_steps(last_date, target_date=target_date, weeks_ahead=weeks_ahead)

    preds = []
    for dt in future_dates:
        row_meta = store_meta.copy()
        row_meta["IsHoliday"] = int(dt.isocalendar().week in {1, 6, 36, 47, 52})
        row_meta["Temperature"] = overrides.get("Temperature", history["Temperature"].iloc[-1] if "Temperature" in history else 0.0)
        row_meta["Fuel_Price"] = overrides.get("Fuel_Price", history["Fuel_Price"].iloc[-1] if "Fuel_Price" in history else 0.0)
        row_meta["CPI"] = overrides.get("CPI", history["CPI"].iloc[-1] if "CPI" in history else 0.0)
        row_meta["Unemployment"] = overrides.get("Unemployment", history["Unemployment"].iloc[-1] if "Unemployment" in history else 0.0)
        row_meta["markdown_total"] = overrides.get("markdown_total", history["markdown_total"].iloc[-1] if "markdown_total" in history else 0.0)
        row_meta["markdown_active"] = int(row_meta["markdown_total"] > 0)

        next_row = build_next_row(history, pd.Timestamp(dt), row_meta)
        pred = float(model.predict(next_row)[0])
        pred = max(0.0, pred)

        resid = package.get("best_residuals", np.array([0.0]))
        alpha = (1 - 0.95) / 2.0
        q_low, q_high = np.quantile(resid, alpha), np.quantile(resid, 1 - alpha)
        lower = max(0.0, pred + q_low)
        upper = max(0.0, pred + q_high)

        preds.append(
            {
                "date": str(pd.Timestamp(dt).date()),
                "store_id": int(store_id),
                "model_used": chosen_name,
                "prediction": float(pred),
                "lower_95": float(lower),
                "upper_95": float(upper),
            }
        )

        append_row = history.iloc[-1:].copy()
        append_row["ds"] = pd.Timestamp(dt)
        append_row["y"] = pred
        append_row["Temperature"] = row_meta["Temperature"]
        append_row["Fuel_Price"] = row_meta["Fuel_Price"]
        append_row["CPI"] = row_meta["CPI"]
        append_row["Unemployment"] = row_meta["Unemployment"]
        append_row["markdown_total"] = row_meta["markdown_total"]
        append_row["markdown_active"] = row_meta["markdown_active"]
        append_row["IsHoliday"] = row_meta["IsHoliday"]
        history = pd.concat([history, append_row], ignore_index=True)

    return preds