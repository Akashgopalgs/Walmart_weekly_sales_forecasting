from __future__ import annotations

import pandas as pd
import numpy as np

from .config import MARKDOWN_COLS


def load_raw(data_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    features = pd.read_csv(data_dir / "features.csv")
    stores = pd.read_csv(data_dir / "stores.csv")

    for df in [train, features]:
        df["Date"] = pd.to_datetime(df["Date"])

    return train, features, stores


def merge_datasets(train: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    df = (
        train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
        .merge(stores, on="Store", how="left")
        .sort_values(["Store", "Dept", "Date"])
        .reset_index(drop=True)
    )
    return df


def clean_merged(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in MARKDOWN_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if col in df.columns:
            df[col] = df.groupby("Store")[col].transform(lambda s: s.ffill().bfill())
        else:
            df[col] = 0.0

    df = df[df["Weekly_Sales"] >= 0].copy()

    q1 = df["Weekly_Sales"].quantile(0.25)
    q3 = df["Weekly_Sales"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    df["Weekly_Sales"] = df["Weekly_Sales"].clip(lower=0.0, upper=upper)

    return df


def make_store_week_panel(df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {"Weekly_Sales": "sum"}

    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        if col in df.columns:
            agg_map[col] = "mean"

    for col in MARKDOWN_COLS:
        if col in df.columns:
            agg_map[col] = "mean"

    if "IsHoliday" in df.columns:
        agg_map["IsHoliday"] = "max"

    if "Type" in df.columns:
        agg_map["Type"] = "first"
    if "Size" in df.columns:
        agg_map["Size"] = "first"

    panel = (
        df.groupby(["Store", "Date"])
        .agg(agg_map)
        .reset_index()
        .rename(columns={"Date": "ds", "Weekly_Sales": "y"})
        .sort_values(["Store", "ds"])
        .reset_index(drop=True)
    )

    panel["markdown_total"] = panel[[c for c in MARKDOWN_COLS if c in panel.columns]].sum(axis=1)
    panel["markdown_active"] = (panel["markdown_total"] > 0).astype(int)

    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"] + MARKDOWN_COLS:
        panel[col] = panel[col].ffill().bfill()

    panel["IsHoliday"] = panel["IsHoliday"].fillna(0).astype(int)
    panel["markdown_total"] = panel["markdown_total"].fillna(0.0)
    panel["markdown_active"] = panel["markdown_active"].fillna(0).astype(int)

    type_map = {"A": 0, "B": 1, "C": 2}
    if "Type" not in panel.columns:
        panel["Type"] = "A"
    if "Size" not in panel.columns:
        panel["Size"] = panel["y"].median()

    panel["TypeCode"] = panel["Type"].map(type_map).fillna(0).astype(int)

    return panel


def global_total_series(panel: pd.DataFrame) -> pd.DataFrame:
    total = (
        panel.groupby("ds", as_index=False)["y"]
        .sum()
        .sort_values("ds")
        .reset_index(drop=True)
    )
    return total


def split_by_time(panel: pd.DataFrame, n_test_weeks: int):
    unique_dates = sorted(panel["ds"].unique())
    test_dates = unique_dates[-n_test_weeks:]
    train_dates = unique_dates[:-n_test_weeks]

    train = panel[panel["ds"].isin(train_dates)].copy().reset_index(drop=True)
    test = panel[panel["ds"].isin(test_dates)].copy().reset_index(drop=True)
    return train, test