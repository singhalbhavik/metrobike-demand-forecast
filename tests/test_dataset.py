"""Offline tests for Stage 2 preprocessing: no BigQuery, no torch training."""
import numpy as np
import pandas as pd
import pytest

from src.models.dataset import (
    SEQ_FEATURE_COLS,
    add_cyclic_features,
    apply_scaler,
    build_station_index,
    build_windows,
    fit_scaler,
)

N_HOURS = 40
SEQ_LEN = 5


def _station_frame(station_id: str, demand_offset: float) -> pd.DataFrame:
    hours = pd.date_range("2024-01-01", periods=N_HOURS, freq="h", tz="UTC")
    idx = np.arange(N_HOURS, dtype=np.float32)
    demand = demand_offset + idx
    return pd.DataFrame(
        {
            "station_id": station_id,
            "hour_utc": hours,
            "demand": demand,
            "temp_f": 70.0,
            "precip_in": 0.0,
            "hour": hours.hour,
            "dayofweek": hours.dayofweek,
            "month": hours.month,
            "is_weekend": (hours.dayofweek >= 5).astype(int),
            "is_holiday": 0,
            # Fabricated (not causally derived) but non-null and station-distinguishable,
            # which is all build_windows/scaling need.
            "lag_168h": demand * 0.1,
            "rolling_mean_168h": demand * 0.05,
        }
    )


@pytest.fixture
def synthetic_df():
    return pd.concat(
        [_station_frame("A", 0.0), _station_frame("B", 1000.0)], ignore_index=True
    )


@pytest.fixture
def boundaries(synthetic_df):
    hours = pd.date_range("2024-01-01", periods=N_HOURS, freq="h", tz="UTC")
    return {
        "train": {"start": hours[0], "end": hours[23]},
        "val": {"start": hours[24], "end": hours[31]},
        "test": {"start": hours[32], "end": hours[39]},
    }


def _prepare(synthetic_df):
    df = add_cyclic_features(synthetic_df)
    scaler = fit_scaler(df[df["station_id"] == "A"].iloc[:24])  # arbitrary train-like slice
    return apply_scaler(df, scaler), scaler


def _collect(dataset):
    """Materializes a StationSequenceDataset's __getitem__ outputs for assertions."""
    x_seq = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])
    x_static = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])
    station_idx = np.array([dataset[i][2].item() for i in range(len(dataset))])
    y = np.array([dataset[i][3].item() for i in range(len(dataset))])
    return x_seq, x_static, station_idx, y


def test_add_cyclic_features_bounded(synthetic_df):
    df = add_cyclic_features(synthetic_df)
    for col in ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]:
        assert df[col].between(-1.0, 1.0).all()


def test_build_windows_split_assignment_matches_target_hour(synthetic_df, boundaries):
    df_scaled, _ = _prepare(synthetic_df)
    station_index = build_station_index(df_scaled)
    windows = build_windows(
        df_scaled, boundaries, station_index, seq_len=SEQ_LEN, train_stride=1, eval_stride=1
    )

    assert len(windows["train"]) == 19 * 2  # target idx 5..23 inclusive, x2 stations
    assert len(windows["val"]) == 8 * 2      # target idx 24..31
    assert len(windows["test"]) == 8 * 2     # target idx 32..39

    expected_ranges = {"train": (5, 23), "val": (24, 31), "test": (32, 39)}
    for split, (lo, hi) in expected_ranges.items():
        _, _, station_idx, y = _collect(windows[split])
        y_a = np.sort(y[station_idx == station_index["A"]])
        y_b = np.sort(y[station_idx == station_index["B"]])
        np.testing.assert_allclose(y_a, np.arange(lo, hi + 1))
        np.testing.assert_allclose(y_b, 1000 + np.arange(lo, hi + 1))


def test_build_windows_never_mixes_stations(synthetic_df, boundaries):
    df_scaled, scaler = _prepare(synthetic_df)
    station_index = build_station_index(df_scaled)
    windows = build_windows(
        df_scaled, boundaries, station_index, seq_len=SEQ_LEN, train_stride=1, eval_stride=1
    )
    demand_col = SEQ_FEATURE_COLS.index("demand_scaled")
    stats = scaler["demand"]

    for split in ("train", "val", "test"):
        x_seq, _, station_idx, _ = _collect(windows[split])
        raw_demand = np.expm1(x_seq[:, :, demand_col] * stats["std"] + stats["mean"])
        for i in range(len(station_idx)):
            is_station_a = station_idx[i] == station_index["A"]
            if is_station_a:
                assert np.all(raw_demand[i] < 500)
            else:
                assert np.all(raw_demand[i] >= 500)


def test_fit_scaler_uses_train_rows_only(synthetic_df):
    train_only = synthetic_df[synthetic_df["station_id"] == "A"].iloc[:24]
    scaler = fit_scaler(train_only)
    expected_mean = float(np.log1p(train_only["demand"]).mean())
    assert scaler["demand"]["mean"] == pytest.approx(expected_mean)

    # Stats must not shift if we hand it a df containing "future" rows outside what was passed in --
    # i.e. fit_scaler is a pure function of whatever df it's given, so the caller (pipeline.py) is
    # responsible for passing only the train split. Verify that contract holds for a larger slice.
    larger = synthetic_df[synthetic_df["station_id"] == "A"]
    scaler_larger = fit_scaler(larger)
    assert scaler_larger["demand"]["mean"] != pytest.approx(scaler["demand"]["mean"])
