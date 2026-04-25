from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .config import (
    RANDOM_STATE,
    N_TEST_WEEKS,
    MODELS_DIR,
    ARTIFACT_DIR,
    PLOTS_DIR,
    EDA_DIR,
    DATA_DIR
)
from .utils import ensure_dirs, set_seed
from .data_loader import load_raw, merge_datasets, clean_merged, make_store_week_panel, split_by_time, global_total_series
from .features import build_panel_features, feature_columns
from .eda import (
    plot_sales_overview,
    plot_seasonal_decomposition,
    plot_acf_pacf,
    plot_per_store_analysis,
    plot_model_comparison,
    adf_summary,
)
from .models import (
    train_statistical_baselines,
    train_ml_models,
    evaluate_models,
    optuna_tune_xgb,
    recursive_forecast_panel,
)
from .metrics import score_all, leaderboard_df
from .monitoring import PredictionMonitor
from .features import feature_columns


def run_pipeline(data_dir=None, n_test_weeks: int = N_TEST_WEEKS, n_trials: int = 25):
    ensure_dirs()
    set_seed(RANDOM_STATE)
    # If no directory is passed, use the one from config.py
    if data_dir is None:
        data_dir = DATA_DIR

    train, features, stores = load_raw(data_dir)
    merged = merge_datasets(train, features, stores)
    merged = clean_merged(merged)
    panel = make_store_week_panel(merged)

    plot_sales_overview(panel)
    plot_seasonal_decomposition(panel)
    plot_acf_pacf(panel)
    plot_per_store_analysis(panel)

    panel_feat = build_panel_features(panel)
    train_panel, test_panel = split_by_time(panel_feat, n_test_weeks)

    # Global statistical baselines are trained store-by-store to keep the panel structure honest.
    stat_preds = train_statistical_baselines(train_panel, test_panel)

    # Tune XGB once on the training panel.
    best_xgb_params = optuna_tune_xgb(train_panel, n_trials=n_trials)

    # Train ML models.
    ml_models = train_ml_models(train_panel, test_panel, xgb_best_params=best_xgb_params)

    # Evaluate ML + statistical baselines on the same panel holdout.
    results = []
    actual = test_panel.sort_values(["ds", "Store"])["y"].values

    for name, pred in stat_preds.items():
        results.append(score_all(actual, pred, name))

    for name, (_, pred) in ml_models.items():
        results.append(score_all(actual, pred, name))

    lb = leaderboard_df(results)
    plot_model_comparison(lb)

    # Pick best ML model for serving.
    best_ml_name = lb.iloc[0]["Model"]
    if best_ml_name in ml_models:
        best_model_obj = ml_models[best_ml_name][0]
        best_pred = ml_models[best_ml_name][1]
    else:
        best_model_obj = ml_models["XGBoost"][0]
        best_pred = ml_models["XGBoost"][1]
        best_ml_name = "XGBoost"

    # Backtest residuals for intervals.
    best_residuals = actual - best_pred

    monitor = PredictionMonitor()
    for d, a, p in zip(test_panel["ds"].values, actual, best_pred):
        monitor.log(d, float(a), float(p))
    monitor_report = pd.DataFrame(monitor.records)

    # Store histories for API inference.
    store_histories = {
        int(sid): grp.sort_values("ds").copy().reset_index(drop=True)
        for sid, grp in train_panel.groupby("Store")
    }

    store_meta = {}
    for sid, grp in train_panel.groupby("Store"):
        first = grp.iloc[-1]
        meta = {
            "Store": int(sid),
            "TypeCode": int(first["TypeCode"]),
            "Size": float(first["Size"]),
            "Temperature": float(first.get("Temperature", 0.0)),
            "Fuel_Price": float(first.get("Fuel_Price", 0.0)),
            "CPI": float(first.get("CPI", 0.0)),
            "Unemployment": float(first.get("Unemployment", 0.0)),
            "markdown_total": float(first.get("markdown_total", 0.0)),
        }
        store_meta[int(sid)] = meta

    package = {
        "best_model_name": best_ml_name,
        "models": {name: obj for name, (obj, _) in ml_models.items()},
        "feature_cols": feature_columns(),
        "store_histories": store_histories,
        "store_meta": store_meta,
        "leaderboard": lb,
        "monitor_report": monitor_report,
        "best_residuals": best_residuals,
        "best_xgb_params": best_xgb_params,
        "train_panel": train_panel,
        "test_panel": test_panel,
        "panel": panel_feat,
        "n_test_weeks": n_test_weeks,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, MODELS_DIR / "model_package.joblib")

    lb.to_csv(ARTIFACT_DIR / "leaderboard.csv", index=False)
    monitor_report.to_csv(ARTIFACT_DIR / "monitoring_report.csv", index=False)

    return package


if __name__ == "__main__":
    run_pipeline(data_dir=None)