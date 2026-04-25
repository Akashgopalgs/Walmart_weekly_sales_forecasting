from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller

from .config import EDA_DIR, PLOTS_DIR, HOLIDAY_WEEKS
from .utils import savefig


def plot_sales_overview(panel: pd.DataFrame) -> None:
    total = panel.groupby("ds", as_index=False)["y"].sum()

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(total["ds"], total["y"] / 1e6, color="#0A2540", lw=1.5, label="Total Sales")
    ax0.plot(total["ds"], total["y"].rolling(4).mean() / 1e6, color="#00C896", lw=2, label="4-week MA")
    ax0.plot(total["ds"], total["y"].rolling(13).mean() / 1e6, color="#FF6B35", lw=2, label="13-week MA")
    ax0.plot(total["ds"], total["y"].rolling(52).mean() / 1e6, color="#C62828", lw=2, ls="--", label="52-week MA")
    ax0.set_title("Sales Overview + Moving Averages")
    ax0.set_ylabel("Sales (M$)")
    ax0.legend()

    total["month"] = total["ds"].dt.month
    ax1 = fig.add_subplot(gs[1, 0])
    monthly = total.groupby("month")["y"].mean() / 1e6
    ax1.bar(range(1, 13), monthly.values)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(list("JFMAMJJASOND"))
    ax1.set_title("Average Sales by Month")
    ax1.set_ylabel("Avg Sales (M$)")

    ax2 = fig.add_subplot(gs[1, 1])
    holiday_avg = panel.groupby("IsHoliday")["y"].mean() / 1e3
    ax2.bar(["Non-Holiday", "Holiday"], holiday_avg.values, color=["#1565C0", "#C62828"])
    ax2.set_title("Holiday vs Non-Holiday Sales")
    ax2.set_ylabel("Avg Weekly Sales ($K)")

    ax3 = fig.add_subplot(gs[1, 2])
    type_avg = panel.groupby("Type")["y"].mean() / 1e3 if "Type" in panel.columns else pd.Series([1, 2, 3], index=["A", "B", "C"])
    ax3.pie(type_avg.values, labels=type_avg.index, autopct="%1.1f%%")
    ax3.set_title("Sales Share by Store Type")

    ax4 = fig.add_subplot(gs[2, :2])
    top_store = panel.groupby("Store")["y"].sum().sort_values(ascending=False).head(12) / 1e6
    ax4.barh(top_store.index.astype(str), top_store.values)
    ax4.invert_yaxis()
    ax4.set_title("Top 12 Stores by Total Sales")
    ax4.set_xlabel("Total Sales (M$)")

    ax5 = fig.add_subplot(gs[2, 2])
    corr_cols = [c for c in ["y", "Temperature", "Fuel_Price", "CPI", "Unemployment"] if c in panel.columns]
    corr = panel[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax5)
    ax5.set_title("Correlation Heatmap")

    savefig(EDA_DIR / "sales_overview.png")


def plot_seasonal_decomposition(panel: pd.DataFrame) -> None:
    total = panel.groupby("ds", as_index=False)["y"].sum()
    result = seasonal_decompose(total.set_index("ds")["y"], model="multiplicative", period=52, extrapolate_trend="freq")

    fig = result.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle("Seasonal Decomposition (Multiplicative, period=52)")
    savefig(EDA_DIR / "seasonal_decomposition.png")


def plot_acf_pacf(panel: pd.DataFrame) -> None:
    total = panel.groupby("ds", as_index=False)["y"].sum()
    series = total["y"].values
    nlag = min(60, len(series) - 1)

    acf_vals = acf(series, nlags=nlag, fft=True)
    pacf_vals = pacf(series, nlags=nlag, method="ywm")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].stem(range(len(acf_vals)), acf_vals)
    axes[0].set_title("ACF")
    axes[1].stem(range(len(pacf_vals)), pacf_vals)
    axes[1].set_title("PACF")
    savefig(EDA_DIR / "acf_pacf.png")


def plot_per_store_analysis(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    store_total = panel.groupby("Store")["y"].sum().sort_values(ascending=False)
    store_total.head(10).plot(kind="bar", ax=axes[0])
    axes[0].set_title("Top 10 Stores by Total Sales")

    store_mean = panel.groupby("Store")["y"].mean().sort_values(ascending=False)
    store_mean.head(10).plot(kind="bar", ax=axes[1])
    axes[1].set_title("Top 10 Stores by Avg Weekly Sales")

    store_std = panel.groupby("Store")["y"].std().sort_values(ascending=False)
    store_std.head(10).plot(kind="bar", ax=axes[2])
    axes[2].set_title("Top 10 Stores by Sales Volatility")

    savefig(EDA_DIR / "per_store_analysis.png")

    if "Type" in panel.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=panel, x="Type", y="y")
        plt.title("Weekly Sales by Store Type")
        savefig(EDA_DIR / "sales_by_store_type.png")


def plot_model_comparison(leaderboard: pd.DataFrame) -> None:
    df = leaderboard.sort_values("WMAPE").copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric in zip(axes, ["RMSE", "MAE", "WMAPE"]):
        sns.barplot(data=df, x=metric, y="Model", ax=ax)
        ax.set_title(f"Model Comparison: {metric}")
    savefig(PLOTS_DIR / "model_comparison.png")


def plot_forecast_vs_actual(dates, actual, forecasts: dict, title: str, out_name: str) -> None:
    plt.figure(figsize=(16, 7))
    plt.plot(dates, actual, label="Actual", color="black", lw=2)

    for name, pred in forecasts.items():
        plt.plot(dates, pred, label=name, lw=1.8, ls="--")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    savefig(PLOTS_DIR / out_name)


def adf_summary(panel: pd.DataFrame) -> dict:
    total = panel.groupby("ds", as_index=False)["y"].sum()
    stat = adfuller(total["y"].values)
    return {"statistic": float(stat[0]), "p_value": float(stat[1])}