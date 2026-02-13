# Teaching & Assessment Notes

## Recommended course placement
- BSc (Sem 4–6): Applied Machine Learning / Data Analytics
- MSc: Advanced ML / Forecasting & MLOps (as baseline & extension)

## Learning outcomes
Students can:
1. Prepare and validate time series data
2. Engineer calendar & seasonal features
3. Build reproducible pipelines (preprocessing + model)
4. Evaluate with time-aware splits (TimeSeriesSplit)
5. Compare models and interpret metrics (MAE/RMSE)
6. Produce a short forecast and discuss limitations

## Suggested student tasks (graded)
### Task A — Baseline & metrics (easy)
- Add a naive baseline (last value / seasonal mean)
- Compare MAE/RMSE to the ML model

### Task B — Feature engineering (medium)
- Add lag features (y_{t-1}, y_{t-7}) carefully (avoid leakage)
- Add rolling mean features (7-day, 30-day)

### Task C — Model comparison (medium)
- Compare HGB vs LightGBM
- Discuss differences and computational trade-offs

### Task D — Backtesting (advanced)
- Implement expanding-window backtesting
- Plot error over time and analyze drift/seasonality

## Research/industry discussion prompts
- What is data leakage in time series?
- Why does random train/test split break time order?
- When do you prefer MAE vs RMSE?
- How would you monitor model drift in production?

## Extension: MLOps-ready packaging
- Convert feature engineering into `src/` functions
- Add CLI entry point (predict next N days)
- Add unit tests (pytest) for feature functions
