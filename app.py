from __future__ import annotations

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

from src.config import MODELS_DIR, ARTIFACT_DIR

# -------------------------------
# API CONFIG
# -------------------------------
API_URL = "https://walmart-weekly-sales-forecasting.onrender.com"

# -------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Walmart Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# -------------------------------
# LOAD LOCAL PACKAGE
# -------------------------------
@st.cache_resource
def load_pkg():
    return joblib.load(MODELS_DIR / "model_package.joblib")


pkg = load_pkg()

leaderboard = pkg["leaderboard"].copy()
stores = sorted(pkg["store_histories"].keys())

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Walmart Weekly Sales Forecasting Dashboard")

st.caption(
    "Multi-store forecasting system using ARIMA, SARIMAX, "
    "Random Forest, Ridge, and XGBoost."
)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:

    st.header("Forecast Controls")

    store_id = st.selectbox(
        "Select Store",
        stores
    )

    model_name = st.selectbox(
        "Model",
        [None] + list(pkg["models"].keys()),
        index=0
    )

    weeks_ahead = st.slider(
        "Weeks Ahead",
        1,
        26,
        8
    )

    target_date = st.text_input(
        "Target Date (optional - YYYY-MM-DD)",
        ""
    )

    temperature = st.number_input(
        "Temperature Override",
        value=0.0
    )

    fuel = st.number_input(
        "Fuel Price Override",
        value=0.0
    )

    cpi = st.number_input(
        "CPI Override",
        value=0.0
    )

    unemployment = st.number_input(
        "Unemployment Override",
        value=0.0
    )

    markdown_total = st.number_input(
        "Markdown Total Override",
        value=0.0
    )

    run_btn = st.button("Generate Forecast")


# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns([1.2, 0.8])

# =========================================================
# RIGHT COLUMN
# =========================================================
with col2:

    st.subheader("🏆 Model Leaderboard")

    st.dataframe(
        leaderboard,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 Model Comparison")

    comparison_plot = ARTIFACT_DIR.parent / "plots" / "model_comparison.png"

    if comparison_plot.exists():
        st.image(
            str(comparison_plot),
            use_container_width=True
        )

    st.subheader("📊 EDA Visualizations")

    eda_images = [
        ARTIFACT_DIR.parent / "eda" / "sales_overview.png",
        ARTIFACT_DIR.parent / "eda" / "seasonal_decomposition.png",
        ARTIFACT_DIR.parent / "eda" / "acf_pacf.png",
        ARTIFACT_DIR.parent / "eda" / "per_store_analysis.png",
    ]

    for img in eda_images:
        if img.exists():
            st.image(
                str(img),
                use_container_width=True
            )


# =========================================================
# LEFT COLUMN
# =========================================================
with col1:

    st.subheader(f"📍 Store {store_id} Forecast")

    if run_btn:

        payload = {
            "store_id": int(store_id),
            "target_date": target_date if target_date.strip() else None,
            "weeks_ahead": int(weeks_ahead),
            "model_name": model_name,
            "temperature_avg": float(temperature),
            "fuel_price": float(fuel),
            "cpi": float(cpi),
            "unemployment": float(unemployment),
            "markdown_total": float(markdown_total),
        }

        # -------------------------------
        # CALL LIVE API
        # -------------------------------
        try:

            with st.spinner("Generating forecast..."):

                response = requests.post(
                    f"{API_URL}/forecast",
                    json=payload,
                    timeout=120
                )

            # -------------------------------
            # SUCCESS
            # -------------------------------
            if response.status_code == 200:

                forecast_df = pd.DataFrame(
                    response.json()["forecasts"]
                )

                st.success("Forecast generated successfully!")

                st.dataframe(
                    forecast_df,
                    use_container_width=True,
                    hide_index=True
                )

                # -------------------------------
                # HISTORY
                # -------------------------------
                hist = pkg["store_histories"][store_id]

                # -------------------------------
                # PLOT
                # -------------------------------
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=hist["ds"],
                        y=hist["y"],
                        mode="lines",
                        name="Historical Sales"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["prediction"],
                        mode="lines+markers",
                        name="Forecast"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["lower_95"],
                        mode="lines",
                        name="Lower 95%",
                        line=dict(dash="dot")
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["upper_95"],
                        mode="lines",
                        name="Upper 95%",
                        line=dict(dash="dot"),
                        fill="tonexty",
                        opacity=0.2
                    )
                )

                fig.update_layout(
                    height=550,
                    title="Forecast vs Historical Sales",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # -------------------------------
                # QUICK LINE CHART
                # -------------------------------
                st.subheader("📉 Forecast Trend")

                st.line_chart(
                    forecast_df.set_index("date")[["prediction"]]
                )

            # -------------------------------
            # API ERROR
            # -------------------------------
            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.code(response.text)

        # -------------------------------
        # REQUEST ERROR
        # -------------------------------
        except Exception as e:

            st.error("Failed to connect to API")

            st.exception(e)

    else:

        st.info(
            "Select inputs and click 'Generate Forecast'"
        )
