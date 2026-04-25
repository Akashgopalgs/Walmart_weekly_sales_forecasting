from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLS


def build_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy().sort_values(["Store", "ds"]).reset_index(drop=True)

    for lag in [1, 2, 4, 52]:
        df[f"lag_{lag}"] = df.groupby("Store")["y"].shift(lag)

    shifted = df.groupby("Store")["y"].shift(1)
    df["roll_4_mean"] = shifted.groupby(df["Store"]).rolling(4).mean().reset_index(level=0, drop=True)
    df["roll_4_std"] = shifted.groupby(df["Store"]).rolling(4).std().reset_index(level=0, drop=True)
    df["roll_13_mean"] = shifted.groupby(df["Store"]).rolling(13).mean().reset_index(level=0, drop=True)
    df["roll_26_mean"] = shifted.groupby(df["Store"]).rolling(26).mean().reset_index(level=0, drop=True)

    df["week"] = df["ds"].dt.isocalendar().week.astype(int)
    df["month"] = df["ds"].dt.month
    df["quarter"] = df["ds"].dt.quarter
    df["is_q4"] = (df["quarter"] == 4).astype(int)

    df["year_trend"] = df.groupby("Store").cumcount()

    df["sin_w1"] = np.sin(2 * np.pi * df["week"] / 52.0)
    df["cos_w1"] = np.cos(2 * np.pi * df["week"] / 52.0)
    df["sin_w2"] = np.sin(4 * np.pi * df["week"] / 52.0)
    df["cos_w2"] = np.cos(4 * np.pi * df["week"] / 52.0)
    df["sin_m1"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_m1"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df = df.dropna().reset_index(drop=True)
    return df


def feature_columns() -> list[str]:
    return FEATURE_COLS


def build_next_row(history: pd.DataFrame, next_date: pd.Timestamp, store_row: pd.Series) -> pd.DataFrame:
    hist = history.sort_values("ds").reset_index(drop=True)
    y = hist["y"].astype(float)

    def lag(k: int) -> float:
        return float(y.iloc[-k]) if len(y) >= k else float(y.mean())

    def roll_mean(k: int) -> float:
        return float(y.iloc[-k:].mean()) if len(y) >= k else float(y.mean())

    def roll_std(k: int) -> float:
        return float(y.iloc[-k:].std()) if len(y) >= k else 0.0

    row = {
        "Store": int(store_row["Store"]),
        "TypeCode": int(store_row["TypeCode"]),
        "Size": float(store_row["Size"]),
        "lag_1": lag(1),
        "lag_2": lag(2),
        "lag_4": lag(4),
        "lag_52": lag(52),
        "roll_4_mean": roll_mean(4),
        "roll_4_std": roll_std(4),
        "roll_13_mean": roll_mean(13),
        "roll_26_mean": roll_mean(26),
        "week": int(next_date.isocalendar().week),
        "month": int(next_date.month),
        "quarter": int(next_date.quarter),
        "is_q4": int(next_date.quarter == 4),
        "year_trend": len(hist),
        "sin_w1": float(np.sin(2 * np.pi * int(next_date.isocalendar().week) / 52.0)),
        "cos_w1": float(np.cos(2 * np.pi * int(next_date.isocalendar().week) / 52.0)),
        "sin_w2": float(np.sin(4 * np.pi * int(next_date.isocalendar().week) / 52.0)),
        "cos_w2": float(np.cos(4 * np.pi * int(next_date.isocalendar().week) / 52.0)),
        "sin_m1": float(np.sin(2 * np.pi * int(next_date.month) / 12.0)),
        "cos_m1": float(np.cos(2 * np.pi * int(next_date.month) / 12.0)),
        "Temperature": float(store_row.get("Temperature", hist["Temperature"].iloc[-1] if "Temperature" in hist else 0.0)),
        "Fuel_Price": float(store_row.get("Fuel_Price", hist["Fuel_Price"].iloc[-1] if "Fuel_Price" in hist else 0.0)),
        "CPI": float(store_row.get("CPI", hist["CPI"].iloc[-1] if "CPI" in hist else 0.0)),
        "Unemployment": float(store_row.get("Unemployment", hist["Unemployment"].iloc[-1] if "Unemployment" in hist else 0.0)),
        "IsHoliday": int(store_row.get("IsHoliday", 0)),
        "markdown_total": float(store_row.get("markdown_total", hist["markdown_total"].iloc[-1] if "markdown_total" in hist else 0.0)),
        "markdown_active": int(store_row.get("markdown_active", 0)),
    }

    for md in [f"MarkDown{i}" for i in range(1, 6)]:
        if md in hist.columns:
            row[md] = float(store_row.get(md, hist[md].iloc[-1]))
        else:
            row[md] = float(store_row.get(md, 0.0))

    return pd.DataFrame([row])[feature_columns()]