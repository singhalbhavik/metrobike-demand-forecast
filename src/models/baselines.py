"""Zero-cost naive baselines, computed directly from Stage 1's lag columns -- no training."""
import numpy as np
import pandas as pd

BASELINES = {
    "Naive persistence (lag_1h)": "lag_1h",
    "Seasonal naive (lag_168h)": "lag_168h",
}


def evaluate_baseline(df: pd.DataFrame, pred_col: str, target_col: str = "demand") -> dict[str, float]:
    valid = df[[pred_col, target_col]].dropna()
    err = valid[pred_col].to_numpy() - valid[target_col].to_numpy()
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
    }
