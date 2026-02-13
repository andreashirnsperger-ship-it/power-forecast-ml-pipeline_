from __future__ import annotations
import numpy as np
import pandas as pd

def generate_synthetic_power_data(start: str = "2025-02-01", months: int = 12, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily energy consumption data for `months` months."""
    rng = np.random.default_rng(seed)
    start_dt = pd.Timestamp(start).normalize()
    end_dt = (start_dt + pd.DateOffset(months=months)) - pd.Timedelta(days=1)
    dates = pd.date_range(start_dt, end_dt, freq="D")

    df = pd.DataFrame({"date": dates})
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear

    seasonal = 1.0 + 0.18 * np.cos(2 * np.pi * (df["dayofyear"].to_numpy() - 20) / 365.25)
    weekend = (df["dow"] >= 5).astype(int)
    weekly = np.where(weekend == 1, 0.92, 1.03)

    temp = 12 + 10 * np.sin(2 * np.pi * (df["dayofyear"].to_numpy() - 172) / 365.25)
    temp += rng.normal(0, 2.0, size=len(df))
    df["temp_c"] = temp

    fixed_holidays = {(1, 1), (5, 1), (8, 15), (10, 26), (12, 25), (12, 26)}
    df["is_holiday"] = df["date"].apply(lambda d: int((d.month, d.day) in fixed_holidays))
    df["is_summer_break"] = df["month"].isin([7, 8]).astype(int)

    holiday_factor = np.where(df["is_holiday"].to_numpy() == 1, 0.88, 1.0)
    summer_factor = np.where(df["is_summer_break"].to_numpy() == 1, 0.93, 1.0)

    base = 520
    noise = rng.normal(0, 25, size=len(df))

    outlier = np.ones(len(df))
    outlier_idx = rng.choice(np.arange(len(df)), size=max(2, len(df) // 60), replace=False)
    outlier[outlier_idx] += rng.normal(0.25, 0.08, size=len(outlier_idx))

    y = base * seasonal * weekly * holiday_factor * summer_factor
    y = y * (1.0 - 0.006 * (df["temp_c"].to_numpy() - 10))
    y = y * outlier + noise
    y = np.clip(y, a_min=250, a_max=None)

    df["y_kwh"] = y.round(1)
    return df
