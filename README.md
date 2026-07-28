# MetroBike Demand Forecast

Hourly station-level trip-demand forecasting for Austin's MetroBike bikeshare system,
built on BigQuery public data and a PyTorch LSTM.

---

## Repo layout

```
sql/          BigQuery SQL — one file per logical step
src/
  data/       fetch, feature engineering, splits
  models/     LSTM and baseline models (Stage 2)
  copilot/    LangChain Q&A copilot (Stage 3)
notebooks/    exploratory notebooks
tests/        unit tests (no BigQuery required)
config/       split date boundaries (generated at runtime)
data/         local parquet files (gitignored)
```

---

## Stage 1 — Data Foundation

### Prerequisites

```bash
# Authenticate with Google Cloud (application-default credentials)
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID   # only if not already set

pip install -r requirements.txt
```

### Run the pipeline

```bash
make data          # or:  python -m src.data.pipeline
```

The pipeline executes in ten steps:

| Step | What happens |
|------|--------------|
| 1 | Print schema stats (date range, station count, total trips) |
| 2 | Fetch hourly trip-start counts per station from BigQuery |
| 3 | Fetch daily NOAA GSOD weather for Austin (temp, precipitation) |
| 4 | Expand sparse demand rows to a complete `(station × hour)` grid, filling gaps with 0 |
| 5 | Left-join daily weather onto the hourly grid by Austin local date |
| 6 | Add calendar features: `hour`, `dayofweek`, `month`, `is_weekend` |
| 7 | Add `is_holiday` flag (US federal + Texas state holidays) |
| 8 | Add lagged demand: `lag_1h`, `lag_24h`, `lag_168h` |
| 9 | Add rolling means (no-leakage shift-then-roll): `rolling_mean_24h`, `rolling_mean_168h` |
| 10 | Strict 70 / 15 / 15 time-based split; save boundaries to `config/splits.yaml` |

Output files written to `data/processed/`:

```
features.parquet   full feature matrix (all splits combined)
train.parquet
val.parquet
test.parquet
```

### Data sources

| Source | Table | Columns used |
|--------|-------|-------------|
| Austin MetroBike | `bigquery-public-data.austin_bikeshare.bikeshare_trips` | `start_station_id`, `start_time` |
| NOAA GSOD | `bigquery-public-data.noaa_gsod.gsod*` (USAF 722530) | `temp`, `prcp` |

### Run tests

```bash
make test          # or:  pytest tests/ -v
```

All tests exercise feature engineering logic offline — no BigQuery calls.

---

## Cost notes

- `01_explore.sql` — full table scan, runs once.
- `02_hourly_demand.sql` — reads only two columns (`start_station_id`, `start_time`).
- `03_weather.sql` — scoped to a single station and bounded by `_TABLE_SUFFIX` wildcard.
