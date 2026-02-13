# Power Forecast ML Pipeline

This repository is a demonstrating an end‑to‑end, reproducible **time series regression pipeline** for energy consumption forecasting.

It is designed to be:
- **Teach-ready** (clear structure, tasks, extensions),
- **Research-aware** (time-series splitting, leakage notes),
- **Industry-relevant** (sklearn pipeline + optional LightGBM baseline).

## What’s inside
- 12 months **synthetic daily energy data** (generated in the notebook)
- Feature Engineering (calendar, seasonality proxies, exogenous variables)
- Reproducible preprocessing + model training via `sklearn.Pipeline`
- Evaluation with **TimeSeriesSplit**, reporting **MAE** and **RMSE**
- Forecasting next 30 days
- Optional comparison: `HistGradientBoostingRegressor` vs. `LightGBM`

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
```

### 2) Run notebook
```bash
jupyter notebook
```
Open:
- `notebooks/ML_Pipeline_PowerForecast.ipynb`

> The notebook will export artifacts into `exports/` (CSV + JSON).

## Metrics
We evaluate using:

- **MAE (Mean Absolute Error)**  
  $$\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|$$

- **RMSE (Root Mean Squared Error)**  
  $$\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}$$

## Portfolio framing
This repo demonstrates:
- **Curriculum fit**: Applied ML / Data Analytics / Time Series
- **Best practices**: pipelines, reproducibility, clear evaluation
- **Assessment readiness**: suggested assignments + extension points (see `docs/teaching_notes.md`)
- **Research awareness**: notes on leakage and backtesting, plus baseline comparison.

## Suggested extensions (for teaching & research)
- Hyperparameter tuning (Optuna)
- Feature importance + SHAP
- Robust backtesting framework
- Incorporate real sensor data + data quality checks
- Deployment (Streamlit dashboard / REST API)

## Author
Andreas Hirnsperger
