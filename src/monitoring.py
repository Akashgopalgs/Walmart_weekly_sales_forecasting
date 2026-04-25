from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import RETRAIN_THRESHOLD_WMAPE
from .utils import savefig


class PredictionMonitor:
    def __init__(self, threshold: float = RETRAIN_THRESHOLD_WMAPE, window: int = 4):
        self.threshold = threshold
        self.window = window
        self.records: list[dict] = []

    def log(self, date, actual: float, predicted: float) -> None:
        err = abs(actual - predicted) / max(abs(actual), 1e-8) * 100
        self.records.append({"date": date, "actual": actual, "predicted": predicted, "abs_pct_err": err})

    def rolling_wmape(self) -> pd.Series:
        if not self.records:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.records).set_index("date")
        num = (df["actual"] - df["predicted"]).abs().rolling(self.window, min_periods=1).sum()
        den = df["actual"].abs().rolling(self.window, min_periods=1).sum().replace(0, np.nan)
        return (num / den) * 100.0

    def needs_retraining(self) -> bool:
        rw = self.rolling_wmape()
        if rw.empty:
            return False
        latest = float(rw.iloc[-1])
        return latest > self.threshold

    def plot(self, out_path) -> None:
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        rw = self.rolling_wmape().values

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(df["date"], df["actual"], label="Actual")
        axes[0].plot(df["date"], df["predicted"], label="Predicted", ls="--")
        axes[0].legend()
        axes[0].set_title("Actual vs Predicted")

        axes[1].plot(df["date"], rw, label=f"{self.window}-week Rolling WMAPE")
        axes[1].axhline(self.threshold, color="red", ls="--", label="Retrain threshold")
        axes[1].legend()
        axes[1].set_title("Drift Monitoring")

        savefig(out_path)


def drift_report(actual, predicted, threshold: float = RETRAIN_THRESHOLD_WMAPE) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    wmape = np.sum(np.abs(actual - predicted)) / max(np.sum(np.abs(actual)), 1e-8) * 100
    return {"wmape": float(wmape), "retrain": bool(wmape > threshold), "threshold": threshold}