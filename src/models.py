from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import RANDOM_STATE, FEATURE_COLS, STAT_EXOG_COLS
from .metrics import rmse, mae, mape, smape, wmape, score_all
from .features import build_next_row, feature_columns


def make_ridge_model() -> TransformedTargetRegressor:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.5)),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def make_rf_model() -> TransformedTargetRegressor:
    rf = RandomForestRegressor(
        n_estimators=700,
        max_depth=12,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)


def make_xgb_model(params: dict | None = None) -> TransformedTargetRegressor:
    params = params or {}
    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)


def date_time_split(unique_dates: list[pd.Timestamp], n_splits=4, test_size=8):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    for tr_idx, val_idx in tscv.split(unique_dates):
        yield [unique_dates[i] for i in tr_idx], [unique_dates[i] for i in val_idx]


def optuna_tune_xgb(train_feat: pd.DataFrame, n_trials: int = 25) -> dict:
    unique_dates = sorted(train_feat["ds"].unique())
    n_splits = 4
    test_size = 8

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        fold_scores = []
        for tr_dates, val_dates in date_time_split(unique_dates, n_splits=n_splits, test_size=test_size):
            tr_df = train_feat[train_feat["ds"].isin(tr_dates)].copy().sort_values(["ds", "Store"])
            val_df = train_feat[train_feat["ds"].isin(val_dates)].copy().sort_values(["ds", "Store"])

            model = make_xgb_model(params)
            model.fit(tr_df[feature_columns()], tr_df["y"].values)

            pred = recursive_forecast_panel(model, tr_df, val_df)
            fold_scores.append(wmape(val_df["y"].values, pred))

        return float(np.mean(fold_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    return best


def recursive_forecast_panel(model, train_panel: pd.DataFrame, future_panel: pd.DataFrame) -> np.ndarray:
    histories = {
        sid: grp.sort_values("ds").copy().reset_index(drop=True)
        for sid, grp in train_panel.groupby("Store")
    }

    future_panel = future_panel.sort_values(["ds", "Store"]).reset_index(drop=True)
    preds = []

    for ds, day_rows in future_panel.groupby("ds", sort=True):
        for _, row in day_rows.sort_values("Store").iterrows():
            sid = int(row["Store"])
            hist = histories[sid]
            next_row = build_next_row(hist, pd.Timestamp(ds), row)
            pred = float(model.predict(next_row)[0])
            pred = max(0.0, pred)
            preds.append(pred)

            new_hist_row = row.copy()
            new_hist_row["ds"] = pd.Timestamp(ds)
            new_hist_row["y"] = pred
            histories[sid] = pd.concat([hist, pd.DataFrame([new_hist_row])], ignore_index=True)

    return np.asarray(preds, dtype=float)


def train_statistical_baselines(train_panel: pd.DataFrame, test_panel: pd.DataFrame) -> dict:
    results = {}

    actual_all = test_panel.sort_values(["ds", "Store"])["y"].values

    arima_preds, sarima_preds, sarimax_preds = [], [], []

    for sid in sorted(train_panel["Store"].unique()):
        tr = train_panel[train_panel["Store"] == sid].sort_values("ds").copy()
        te = test_panel[test_panel["Store"] == sid].sort_values("ds").copy()

        y_tr = np.log1p(tr["y"].values)
        y_te = te["y"].values

        ex_tr = tr[STAT_EXOG_COLS].fillna(0)
        ex_te = te[STAT_EXOG_COLS].fillna(0)

        try:
            arima_fit = ARIMA(y_tr, order=(2, 1, 2)).fit()
            arima_p = np.expm1(np.asarray(arima_fit.forecast(len(te)), dtype=float))
        except Exception:
            arima_p = np.repeat(tr["y"].iloc[-1], len(te))
        arima_preds.extend(arima_p.tolist())

        try:
            sarima_fit = SARIMAX(
                y_tr,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            sarima_p = np.expm1(np.asarray(sarima_fit.forecast(len(te)), dtype=float))
        except Exception:
            sarima_p = np.repeat(tr["y"].iloc[-1], len(te))
        sarima_preds.extend(sarima_p.tolist())

        try:
            sarimax_fit = SARIMAX(
                y_tr,
                exog=ex_tr,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            sarimax_p = np.expm1(np.asarray(sarimax_fit.forecast(len(te), exog=ex_te), dtype=float))
        except Exception:
            sarimax_p = np.repeat(tr["y"].iloc[-1], len(te))
        sarimax_preds.extend(sarimax_p.tolist())

    results["ARIMA(2,1,2)"] = np.asarray(arima_preds, dtype=float)
    results["SARIMA(1,1,1)(1,0,1,52)"] = np.asarray(sarima_preds, dtype=float)
    results["SARIMAX(exog)"] = np.asarray(sarimax_preds, dtype=float)

    return results


def train_ml_models(train_feat: pd.DataFrame, test_feat: pd.DataFrame, xgb_best_params: dict | None = None):
    xgb_best_params = xgb_best_params or {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }

    ridge = make_ridge_model()
    rf = make_rf_model()
    xgb_model = make_xgb_model(xgb_best_params)

    X_train = train_feat[feature_columns()]
    y_train = train_feat["y"].values
    X_test = test_feat[feature_columns()]

    ridge.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    ridge_pred = recursive_forecast_panel(ridge, train_feat, test_feat)
    rf_pred = recursive_forecast_panel(rf, train_feat, test_feat)
    xgb_pred = recursive_forecast_panel(xgb_model, train_feat, test_feat)

    return {
        "Ridge": (ridge, ridge_pred),
        "Random Forest": (rf, rf_pred),
        "XGBoost": (xgb_model, xgb_pred),
    }


def evaluate_models(actual, preds_dict: dict) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    score_rows = []
    for name, pred in preds_dict.items():
        row = score_all(actual, pred, name)
        rows.append(row)
        score_rows.append(row)
    lb = pd.DataFrame(rows).sort_values(["WMAPE", "RMSE"]).reset_index(drop=True)
    return lb, score_rows