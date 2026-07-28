"""Stage 2 preprocessing: cyclic features, train-only scaling, and sliding-window construction.

Windows are built from the full feature matrix (continuous per-station timeline, same precedent
as Stage 1's lag/rolling features) and assigned to train/val/test by their *target* hour against
`config/splits.yaml` boundaries -- a window's lookback may dip into hours that are chronologically
"train", which is not leakage (it's real past data available at prediction time), exactly how
lag_1h/lag_168h were computed across the full timeline before Stage 1 split the data.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Continuous columns that get log1p (count-derived, zero-heavy/skewed) + standardized.
LOG1P_SCALE_COLS = ["demand", "lag_168h", "rolling_mean_168h"]
# Continuous columns that get standardized without log1p.
LINEAR_SCALE_COLS = ["temp_f", "precip_in"]
SCALE_COLS = LOG1P_SCALE_COLS + LINEAR_SCALE_COLS

# Per-timestep sequence input to the LSTM (24h lookback window).
SEQ_FEATURE_COLS = [
    "demand_scaled", "temp_f_scaled", "precip_in_scaled",
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
    "is_weekend", "is_holiday",
]
# Static features concatenated at the FC head: weekly-scale signal (lag_168h / rolling_mean_168h)
# plus the *target* hour's own calendar attributes (known in advance -- you always know what
# day/hour you're forecasting, even before observing the lookback window).
STATIC_FEATURE_COLS = [
    "lag_168h_scaled", "rolling_mean_168h_scaled",
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "is_weekend", "is_holiday",
]


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sin/cos encodings for hour-of-day and day-of-week (periodic, not ordinal)."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def fit_scaler(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean/std per scaled column, computed only from train rows (log1p first where applicable)."""
    scaler = {}
    for col in SCALE_COLS:
        values = train_df[col].dropna()
        if col in LOG1P_SCALE_COLS:
            values = np.log1p(values)
        std = float(values.std())
        scaler[col] = {"mean": float(values.mean()), "std": std if std > 1e-8 else 1.0}
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """Writes `<col>_scaled` columns; leaves originals untouched (still needed for the raw target)."""
    df = df.copy()
    for col in SCALE_COLS:
        values = df[col]
        if col in LOG1P_SCALE_COLS:
            values = np.log1p(values)
        stats = scaler[col]
        df[f"{col}_scaled"] = (values - stats["mean"]) / stats["std"]
    return df


def build_station_index(df: pd.DataFrame) -> dict[str, int]:
    return {sid: i for i, sid in enumerate(sorted(df["station_id"].unique()))}


class StationSequenceDataset(Dataset):
    """
    Slices windows lazily from compact per-station arrays instead of materializing every
    overlapping window up front. With millions of overlapping 24h windows, pre-stacking them
    would duplicate the data ~seq_len-fold (each hour appears in ~24 windows) -- on an 8GB
    laptop that's the difference between a dataset that fits and one that doesn't. Each
    station's full timeline is stored once; `__getitem__` slices out the requested window.
    """

    def __init__(
        self,
        station_arrays: dict[int, dict[str, np.ndarray]],
        index: np.ndarray,  # (N, 2) int64: [station_idx, target_row_idx]
        seq_len: int,
    ):
        self.station_arrays = station_arrays
        self.index = index
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        station_idx, target_idx = self.index[i]
        arrays = self.station_arrays[int(station_idx)]
        x_seq = arrays["seq"][target_idx - self.seq_len : target_idx]
        x_static = arrays["static"][target_idx]
        y = arrays["demand"][target_idx]
        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(x_static),
            torch.tensor(station_idx, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )


def build_windows(
    df: pd.DataFrame,
    boundaries: dict,
    station_index: dict[str, int],
    seq_len: int,
    train_stride: int,
    eval_stride: int,
) -> dict[str, StationSequenceDataset]:
    """
    df: full feature matrix with cyclic + scaled columns already added, any row order.
    boundaries: {"train": {"start": ..., "end": ...}, "val": {...}, "test": {...}} (as in splits.yaml)
    Returns {"train": StationSequenceDataset, "val": ..., "test": ...}, all sharing the same
    underlying per-station arrays (only the index of valid (station, target) pairs differs).
    """
    bounds = {
        split: (pd.Timestamp(b["start"]), pd.Timestamp(b["end"]))
        for split, b in boundaries.items()
    }
    strides = {"train": train_stride, "val": eval_stride, "test": eval_stride}

    station_arrays: dict[int, dict[str, np.ndarray]] = {}
    index_by_split: dict[str, list[tuple[int, int]]] = {split: [] for split in bounds}

    for station_id, g in df.sort_values("hour_utc").groupby("station_id", sort=False):
        g = g.reset_index(drop=True)
        idx = station_index[station_id]
        station_arrays[idx] = {
            "seq": g[SEQ_FEATURE_COLS].to_numpy(dtype=np.float32),
            "static": g[STATIC_FEATURE_COLS].to_numpy(dtype=np.float32),
            "demand": g["demand"].to_numpy(dtype=np.float32),
        }
        hour_series = g["hour_utc"]  # tz-aware; use pandas' own searchsorted, not numpy's
        seq_ok = g[SEQ_FEATURE_COLS].notna().all(axis=1).to_numpy()
        static_ok = g[STATIC_FEATURE_COLS].notna().all(axis=1).to_numpy()

        for split, (start_ts, end_ts) in bounds.items():
            lo = max(hour_series.searchsorted(start_ts, side="left"), seq_len)
            hi = hour_series.searchsorted(end_ts, side="right")
            stride = strides[split]
            for target_idx in range(lo, hi, stride):
                window_start = target_idx - seq_len
                if not static_ok[target_idx] or not seq_ok[window_start:target_idx].all():
                    continue
                index_by_split[split].append((idx, target_idx))

    return {
        split: StationSequenceDataset(
            station_arrays, np.array(idx_list, dtype=np.int64), seq_len
        )
        for split, idx_list in index_by_split.items()
    }
