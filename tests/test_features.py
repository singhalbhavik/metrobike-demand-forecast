"""Unit tests for feature engineering — no BigQuery required."""
import pandas as pd
import pytest

from src.data.features import (
    add_holiday_flag,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    build_full_grid,
)


def _demand(n_hours: int = 48, n_stations: int = 2) -> pd.DataFrame:
    hours = pd.date_range("2023-01-01", periods=n_hours, freq="h", tz="UTC")
    stations = [str(i) for i in range(1, n_stations + 1)]
    rows = [
        {"station_id": s, "hour_utc": h, "demand": float((i * n_hours + j) % 5)}
        for i, s in enumerate(stations)
        for j, h in enumerate(hours)
    ]
    return pd.DataFrame(rows)


def test_build_full_grid_completes_gaps():
    df = _demand(n_hours=10, n_stations=2)
    sparse = df.iloc[::2].copy()  # remove every other row
    grid = build_full_grid(sparse)
    counts = grid.groupby("station_id")["hour_utc"].count()
    assert counts.nunique() == 1, "all stations must have equal hour counts"
    assert (grid["demand"] >= 0).all()


def test_build_full_grid_zero_fill():
    df = _demand(n_hours=4, n_stations=1)
    sparse = df.iloc[[0, 3]].copy()  # only first and last row
    grid = build_full_grid(sparse)
    assert len(grid) == 4
    assert grid.iloc[1]["demand"] == 0.0
    assert grid.iloc[2]["demand"] == 0.0


def test_add_time_features_ranges():
    df = add_time_features(_demand())
    assert df["hour"].between(0, 23).all()
    assert df["dayofweek"].between(0, 6).all()
    assert df["month"].between(1, 12).all()
    assert df["is_weekend"].isin([0, 1]).all()


def test_add_holiday_flag_new_years():
    df = _demand(n_hours=48, n_stations=1)  # 2023-01-01 and 2023-01-02
    df = add_holiday_flag(df)
    assert "is_holiday" in df.columns
    # Hours >= 06:00 UTC on 2023-01-01 are New Year's Day in CST (UTC-6)
    jan1_afternoon = df[
        (df["hour_utc"].dt.date == pd.Timestamp("2023-01-01").date())
        & (df["hour_utc"].dt.hour >= 6)
    ]
    assert jan1_afternoon["is_holiday"].sum() > 0


def test_lag_features_correctness():
    df = build_full_grid(_demand(n_hours=200, n_stations=1))
    df = add_lag_features(df).sort_values("hour_utc").reset_index(drop=True)
    # lag_1h at row i should equal demand at row i-1
    assert df.loc[1, "lag_1h"] == df.loc[0, "demand"]
    # lag_168h at row 168 should equal demand at row 0
    assert df.loc[168, "lag_168h"] == df.loc[0, "demand"]


def test_lag_features_warmup_nan():
    df = build_full_grid(_demand(n_hours=50, n_stations=1))
    df = add_lag_features(df).sort_values("hour_utc").reset_index(drop=True)
    assert pd.isna(df.loc[0, "lag_1h"])
    assert pd.isna(df.loc[0, "lag_168h"])


def test_rolling_no_leakage():
    """rolling_mean_24h at t must not include demand at t."""
    df = build_full_grid(_demand(n_hours=50, n_stations=1))
    df = add_rolling_features(df).sort_values("hour_utc").reset_index(drop=True)
    assert "rolling_mean_24h" in df.columns
    assert "rolling_mean_168h" in df.columns
    # At t=0, the shift(1) is NaN so rolling mean uses nothing — should be NaN.
    assert pd.isna(df.loc[0, "rolling_mean_24h"])


def test_split_preserves_temporal_order(tmp_path, monkeypatch):
    from src.data import splits

    monkeypatch.setattr(splits, "_CONFIG_DIR", tmp_path)  # don't clobber the real config/splits.yaml
    df = build_full_grid(_demand(n_hours=100, n_stations=2))
    df = add_time_features(df)
    train, val, test, _ = splits.split_data(df, val_frac=0.15, test_frac=0.15)
    assert train["hour_utc"].max() < val["hour_utc"].min()
    assert val["hour_utc"].max() < test["hour_utc"].min()


def test_split_no_overlap(tmp_path, monkeypatch):
    from src.data import splits

    monkeypatch.setattr(splits, "_CONFIG_DIR", tmp_path)  # don't clobber the real config/splits.yaml
    df = build_full_grid(_demand(n_hours=100, n_stations=2))
    df = add_time_features(df)
    train, val, test, _ = splits.split_data(df)
    all_hours = set(df["hour_utc"].unique())
    train_h = set(train["hour_utc"].unique())
    val_h   = set(val["hour_utc"].unique())
    test_h  = set(test["hour_utc"].unique())
    assert not (train_h & val_h), "train and val must not share hours"
    assert not (val_h & test_h),  "val and test must not share hours"
    assert train_h | val_h | test_h == all_hours
