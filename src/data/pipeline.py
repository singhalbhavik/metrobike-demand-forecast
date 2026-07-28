"""
Stage 1 data pipeline: fetch → expand → join weather → engineer features → split → save.
Entry point:  python -m src.data.pipeline
              make data
"""
from pathlib import Path

from src.data.fetch import fetch_hourly_demand, fetch_schema_stats, fetch_weather
from src.data.features import (
    add_holiday_flag,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    add_weather,
    build_full_grid,
)
from src.data.splits import split_data

_OUT = Path("data/processed")


def main() -> None:
    # ── 1. Schema exploration ───────────────────────────────────────────────
    print("=" * 60)
    print("Schema exploration")
    print("=" * 60)
    stats = fetch_schema_stats()
    print(stats.to_string(index=False))
    print()

    # ── 2. Hourly demand ───────────────────────────────────────────────────
    print("Fetching hourly demand from BigQuery...")
    demand = fetch_hourly_demand()
    print(f"  Non-zero rows : {len(demand):>10,}")
    print(f"  Stations      : {demand['station_id'].nunique():>10,}")
    print(f"  Date range    : {demand['hour_utc'].min()} → {demand['hour_utc'].max()}")
    print()

    start_year = demand["hour_utc"].dt.year.min()
    end_year   = demand["hour_utc"].dt.year.max()

    # ── 3. Weather ─────────────────────────────────────────────────────────
    print(f"Fetching NOAA GSOD weather ({start_year}–{end_year + 1})...")
    weather = fetch_weather(start_year, end_year + 1)
    print(f"  Days fetched  : {len(weather):>10,}")
    print(f"  Date range    : {weather['date'].min().date()} → {weather['date'].max().date()}")
    print()

    # ── 4. Full (station × hour) grid ──────────────────────────────────────
    print("Expanding to complete hourly grid (filling zero-demand hours)...")
    df = build_full_grid(demand)
    print(f"  Grid rows     : {len(df):>10,}")
    print()

    # ── 5. Join weather ────────────────────────────────────────────────────
    print("Joining daily weather onto hourly grid...")
    df = add_weather(df, weather)

    # ── 6. Calendar + holiday features ────────────────────────────────────
    print("Adding time and holiday features...")
    df = add_time_features(df)
    df = add_holiday_flag(df)

    # ── 7. Lag features ────────────────────────────────────────────────────
    print("Adding lag features (1h, 24h, 168h)...")
    df = add_lag_features(df)

    # ── 8. Rolling means ───────────────────────────────────────────────────
    print("Adding rolling means (24h, 168h)...")
    df = add_rolling_features(df)

    print()
    print(f"Feature matrix : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Columns        : {list(df.columns)}")
    print()

    # ── 9. Time-based splits ───────────────────────────────────────────────
    print("Splitting 70 / 15 / 15 (by unique hours, strict temporal order)...")
    train, val, test, bounds = split_data(df)
    for name, split in [("Train", train), ("Val  ", val), ("Test ", test)]:
        key = name.strip().lower()
        print(
            f"  {name}: {len(split):>10,} rows  "
            f"[{bounds[key]['start']} → {bounds[key]['end']}]"
        )
    print("  Split boundaries saved to config/splits.yaml")
    print()

    # ── 10. Save to parquet ────────────────────────────────────────────────
    _OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT / "features.parquet", index=False)
    train.to_parquet(_OUT / "train.parquet", index=False)
    val.to_parquet(_OUT / "val.parquet",     index=False)
    test.to_parquet(_OUT / "test.parquet",   index=False)
    print(f"Saved parquet files to {_OUT}/")
    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
