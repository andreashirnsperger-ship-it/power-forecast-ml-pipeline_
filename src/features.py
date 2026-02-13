from __future__ import annotations
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    return out
