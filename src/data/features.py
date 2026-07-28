"""Feature engineering: grid expansion, weather join, lags, rolling means, calendars."""
import pandas as pd
import holidays


def build_full_grid(demand: pd.DataFrame) -> pd.DataFrame:
    """
    Expand sparse demand rows into a complete (station × hour) grid.
    Hours with no trips get demand=0 so lag/rolling windows have no gaps.
    """
    all_hours = pd.date_range(
        demand["hour_utc"].min(),
        demand["hour_utc"].max(),
        freq="h",
        tz="UTC",
    )
    all_stations = demand["station_id"].unique()
    grid = (
        pd.MultiIndex.from_product(
            [all_stations, all_hours], names=["station_id", "hour_utc"]
        )
        .to_frame(index=False)
    )
    return grid.merge(demand, on=["station_id", "hour_utc"], how="left").fillna(
        {"demand": 0}
    )


def add_weather(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Left-join daily weather onto hourly demand by Austin local date."""
    df = demand.copy()
    # Extract the local calendar date for the join key (Austin = America/Chicago).
    df["_date"] = (
        df["hour_utc"]
        .dt.tz_convert("America/Chicago")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    w = weather.copy()
    w["_date"] = w["date"].dt.normalize()
    df = df.merge(w.drop(columns="date"), on="_date", how="left").drop(columns="_date")
    # Forward-fill rare missing temperature readings; missing precip = dry day.
    df["temp_f"] = df["temp_f"].ffill()
    df["precip_in"] = df["precip_in"].fillna(0.0)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    local = df["hour_utc"].dt.tz_convert("America/Chicago")
    df["hour"]       = local.dt.hour
    df["dayofweek"]  = local.dt.dayofweek   # Monday=0, Sunday=6
    df["month"]      = local.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    local_dates = df["hour_utc"].dt.tz_convert("America/Chicago").dt.date
    years = sorted({d.year for d in local_dates.unique()})
    tx_holidays = holidays.country_holidays("US", subdiv="TX", years=years)
    df["is_holiday"] = local_dates.map(lambda d: int(d in tx_holidays))
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag-1h, 24h, 168h within each station. NaN for the warm-up period."""
    df = df.copy().sort_values(["station_id", "hour_utc"])
    grp = df.groupby("station_id", sort=False)["demand"]
    for lag_h in [1, 24, 168]:
        df[f"lag_{lag_h}h"] = grp.shift(lag_h)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling means over the preceding 24h and 168h (shift-1 avoids target leakage).
    min_periods=1 so early rows produce a value instead of NaN.
    """
    df = df.copy().sort_values(["station_id", "hour_utc"])
    grp = df.groupby("station_id", sort=False)["demand"]
    for window_h in [24, 168]:
        df[f"rolling_mean_{window_h}h"] = grp.transform(
            lambda s: s.shift(1).rolling(window_h, min_periods=1).mean()
        )
    return df
