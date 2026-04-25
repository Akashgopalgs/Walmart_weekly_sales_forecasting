from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def _safe_array(arr):
    arr = np.asarray(arr, dtype=float)
    # Replace NaN with 0, +Inf with large finite, -Inf with large finite
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    return arr

def rmse(actual, predicted) -> float:
    actual = _safe_array(actual)
    predicted = _safe_array(predicted)
    return float(np.sqrt(mean_squared_error(actual, predicted)))

def mae(actual, predicted) -> float:
    actual = _safe_array(actual)
    predicted = _safe_array(predicted)
    return float(mean_absolute_error(actual, predicted))

def mape(actual, predicted) -> float:
    a = _safe_array(actual)
    p = _safe_array(predicted)
    mask = np.abs(a) > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)

def smape(actual, predicted) -> float:
    a = _safe_array(actual)
    p = _safe_array(predicted)
    denom = (np.abs(a) + np.abs(p)) / 2.0
    mask = denom > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(a[mask] - p[mask]) / denom[mask]) * 100)

def wmape(actual, predicted) -> float:
    a = _safe_array(actual)
    p = _safe_array(predicted)
    denom = np.sum(np.abs(a))
    if denom <= 1e-8:
        return float("nan")
    return float(np.sum(np.abs(a - p)) / denom * 100)


def score_all(actual, predicted, name: str) -> dict:
    return {
        "Model": name,
        "RMSE": rmse(actual, predicted),
        "MAE": mae(actual, predicted),
        "MAPE": mape(actual, predicted),
        "SMAPE": smape(actual, predicted),
        "WMAPE": wmape(actual, predicted),
    }


def leaderboard_df(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values(["WMAPE", "RMSE"]).reset_index(drop=True)