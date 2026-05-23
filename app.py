API_URL = "https://walmart-weekly-sales-forecasting.onrender.com"
import requests

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

from src.config import MODELS_DIR, ARTIFACT_DIR
from src.forecasting_engine import forecast_store

if run_btn:
    payload = {
        "store_id": store_id,
        "target_date": target_date if target_date.strip() else None,
        "weeks_ahead": weeks_ahead,
        "model_name": model_name,
        "temperature_avg": temperature,
        "fuel_price": fuel,
        "cpi": cpi,
        "unemployment": unemployment,
        "markdown_total": markdown_total,
    }

    response = requests.post(f"{API_URL}/forecast", json=payload)

    if response.status_code == 200:
        forecast_df = pd.DataFrame(response.json()["forecasts"])
        st.success("Forecast generated")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # plotting code stays the same
    else:
        st.error(f"API error: {response.status_code} - {response.text}")


st.set_page_config(page_title="Walmart Forecast Dashboard", page_icon="📈", layout="wide")

@st.cache_resource
def load_pkg():
    return joblib.load(MODELS_DIR / "model_package.joblib")


pkg = load_pkg()
leaderboard = pkg["leaderboard"].copy()
stores = sorted(pkg["store_histories"].keys())

st.title("📊 Walmart Weekly Sales Forecasting Dashboard")
st.caption("Multi-store forecasting, model comparison, and real-time prediction UI")

with st.sidebar:
    st.header("Forecast Controls")
    store_id = st.selectbox("Store", stores)
    model_name = st.selectbox("Model", [None] + list(pkg["models"].keys()), index=0)
    weeks_ahead = st.slider("Weeks ahead", 1, 26, 8)
    target_date = st.text_input("Target date (optional, YYYY-MM-DD)", "")
    temperature = st.number_input("Temperature override", value=0.0)
    fuel = st.number_input("Fuel price override", value=0.0)
    cpi = st.number_input("CPI override", value=0.0)
    unemployment = st.number_input("Unemployment override", value=0.0)
    markdown_total = st.number_input("Markdown total override", value=0.0)

    run_btn = st.button("Generate Forecast")

col1, col2 = st.columns([1.15, 0.85])

with col2:
    st.subheader("Model Leaderboard")
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    st.subheader("Saved EDA / Comparison Plots")
    st.image(str((ARTIFACT_DIR.parent / "plots" / "model_comparison.png")), use_container_width=True)

    for img in [
        ARTIFACT_DIR.parent / "eda" / "sales_overview.png",
        ARTIFACT_DIR.parent / "eda" / "seasonal_decomposition.png",
        ARTIFACT_DIR.parent / "eda" / "acf_pacf.png",
        ARTIFACT_DIR.parent / "eda" / "per_store_analysis.png",
    ]:
        if img.exists():
            st.image(str(img), use_container_width=True)

with col1:
    st.subheader(f"Store {store_id} Forecast")

    if run_btn:
        overrides = {
            "Temperature": temperature,
            "Fuel_Price": fuel,
            "CPI": cpi,
            "Unemployment": unemployment,
            "markdown_total": markdown_total,
        }

        preds = forecast_store(
            pkg,
            store_id=store_id,
            target_date=target_date if target_date.strip() else None,
            weeks_ahead=weeks_ahead,
            model_name=model_name,
            overrides=overrides,
        )

        forecast_df = pd.DataFrame(preds)

        st.success("Forecast generated")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        hist = pkg["store_histories"][store_id]
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], mode="lines", name="History"))

        fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["prediction"], mode="lines+markers", name="Forecast"))
        fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["lower_95"], mode="lines", name="Lower 95%", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["upper_95"], mode="lines", name="Upper 95%", line=dict(dash="dot"), fill="tonexty", opacity=0.2))

        fig.update_layout(height=520, title="Forecast vs History")
        st.plotly_chart(fig, use_container_width=True)

        st.line_chart(forecast_df.set_index("date")[["prediction"]])

    else:
        st.info("Choose a store and click Generate Forecast.")
