"""Strict time-based train / val / test splits with no shuffling."""
from pathlib import Path

import pandas as pd
import yaml

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def split_data(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Split on unique hours (not rows) so all stations share the same
    temporal boundary — no station can leak future hours into training.
    """
    hours = df["hour_utc"].sort_values().unique()
    n = len(hours)

    train_end_idx = int(n * (1 - val_frac - test_frac))
    val_end_idx   = int(n * (1 - test_frac))

    train_end  = hours[train_end_idx - 1]
    val_start  = hours[train_end_idx]
    val_end    = hours[val_end_idx - 1]
    test_start = hours[val_end_idx]

    train = df[df["hour_utc"] <= train_end].copy()
    val   = df[(df["hour_utc"] >= val_start) & (df["hour_utc"] <= val_end)].copy()
    test  = df[df["hour_utc"] >= test_start].copy()

    boundaries = {
        "train": {"start": str(hours[0]),   "end": str(train_end)},
        "val":   {"start": str(val_start),  "end": str(val_end)},
        "test":  {"start": str(test_start), "end": str(hours[-1])},
    }

    _CONFIG_DIR.mkdir(exist_ok=True)
    with open(_CONFIG_DIR / "splits.yaml", "w") as f:
        yaml.dump(boundaries, f, default_flow_style=False, sort_keys=True)

    return train, val, test, boundaries
