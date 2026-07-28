"""BigQuery fetch layer — each function returns a clean DataFrame."""
import os
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

_SQL_DIR = Path(__file__).parent.parent.parent / "sql"


def _client() -> bigquery.Client:
    # Picks up GOOGLE_CLOUD_PROJECT env var when set; falls back to ADC default.
    return bigquery.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))


def _sql(name: str) -> str:
    return (_SQL_DIR / name).read_text()


def fetch_schema_stats() -> pd.DataFrame:
    """Single-row summary: date range, station count, total trips."""
    return _client().query(_sql("01_explore.sql")).to_dataframe()


def fetch_hourly_demand() -> pd.DataFrame:
    """Hourly trip-start counts per station. Returns (station_id, hour_utc, demand)."""
    df = _client().query(_sql("02_hourly_demand.sql")).to_dataframe()
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True)
    df["station_id"] = df["station_id"].astype(str)
    return df


def fetch_weather(start_year: int, end_year: int) -> pd.DataFrame:
    """Daily Austin weather (temp °F, precip inches) from NOAA GSOD."""
    sql = _sql("03_weather.sql").format(start_year=start_year, end_year=end_year)
    df = _client().query(sql).to_dataframe()
    df["date"] = pd.to_datetime(df["date"])
    return df
