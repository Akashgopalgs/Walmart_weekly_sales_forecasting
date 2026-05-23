# 📈 Walmart Weekly Sales Forecasting

End-to-end production-style time series forecasting system for Walmart weekly sales prediction using statistical models, machine learning, FastAPI, and Streamlit deployment.

<div style="display: flex; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/df86dba8-e840-411c-9c39-d0a3506ad290" width="32%" />
  <img src="https://github.com/user-attachments/assets/935cd97a-c1a6-466f-b84b-af1cea650ae8" width="32%" />
  <img  src="https://github.com/user-attachments/assets/76c3a855-c4f1-4d99-934f-2c75e9430232" width="32%" />
</div>


---

## 🚀 Live Demo

### 🌐 Dashboard
https://walmartweeklysalesforecasting-dashboard.streamlit.app/

### ⚡ API Docs
https://walmart-weekly-sales-forecasting.onrender.com/docs

---

## 📌 Project Overview

This project forecasts weekly Walmart store sales using:

- Time Series Forecasting
- Machine Learning
- Recursive Multi-step Forecasting
- Real-time Prediction API
- Interactive Dashboard Deployment

The system supports **multi-store forecasting**, confidence intervals, model comparison, and real-time forecasting through deployed APIs.

---

## 🧠 Models Used

### Statistical Models
- ARIMA
- SARIMA
- SARIMAX

### Machine Learning Models
- Ridge Regression
- Random Forest
- XGBoost

### Optimization & Validation
- Optuna Hyperparameter Tuning
- TimeSeriesSplit Cross Validation
- Recursive Forecasting
- Walk-forward Backtesting

---

## ⚙️ Features

✅ Multi-store forecasting  
✅ Advanced feature engineering  
✅ Rolling statistics & lag features  
✅ Seasonal decomposition  
✅ ACF / PACF analysis  
✅ Confidence intervals  
✅ Drift monitoring & retraining logic  
✅ FastAPI deployment  
✅ Streamlit dashboard  
✅ Docker-ready architecture  
✅ CI/CD workflow support  

---

## 📊 Evaluation Metrics

Models are evaluated using:

- RMSE
- MAE
- MAPE
- SMAPE
- WMAPE

A complete leaderboard comparison is generated automatically.

---

## 🏗️ Project Architecture

```text
Streamlit Dashboard
        ↓
FastAPI Backend
        ↓
Forecasting Engine
        ↓
ML / Statistical Models
        ↓
Recursive Forecast Generation
