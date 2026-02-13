from __future__ import annotations
from sklearn.ensemble import HistGradientBoostingRegressor

def get_hgb():
    return HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.07,
        max_iter=400,
    )

def get_lgb():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        random_state=42,
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=-1
    )
