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

---

## Stage 2 — LSTM Forecasting Model

A global PyTorch LSTM predicting next-hour demand per station, benchmarked against two
zero-cost naive baselines already implicit in Stage 1's engineered features.

### Design

- **Hybrid sequence design**: the LSTM sees a 24h lookback window (captures daily rhythm:
  demand, weather, hour/day-of-week cyclic encodings). Weekly-scale signal (`lag_168h`,
  `rolling_mean_168h`) and the *target* hour's own calendar attributes (hour/day-of-week —
  known in advance, since you always know what hour you're forecasting) are fed in as static
  features at the FC head instead of extending the sequence to a full 7 days — LSTMs aren't
  great at remembering 168 steps back, so the model gets that signal directly.
- **No leakage**: windows are built from the full continuous per-station timeline (same
  precedent as Stage 1's lag/rolling features) and assigned to train/val/test by their
  *target* hour against `config/splits.yaml`'s boundaries. The scaler (mean/std, log1p first
  for zero-heavy count columns) is fit on the train split only.
- **Baselines**: naive persistence (`lag_1h`) and seasonal-naive (`lag_168h`), evaluated
  directly from Stage 1's columns — no training needed.
- **Memory**: windows are sliced lazily from compact per-station arrays at `__getitem__` time
  rather than materialized upfront — with ~5M overlapping 24h windows, pre-stacking them would
  duplicate the data ~24x, which doesn't fit comfortably on an 8GB laptop.

### Run it

```bash
make train         # or: python -m src.models.pipeline
```

Runs entirely against the local Stage 1 parquet files — no BigQuery calls. Trains on Apple
Silicon MPS if available, else CPU. Produces:

```
models/lstm_best.pt        best checkpoint (by val RMSE, gitignored)
config/preprocessing.yaml  fitted scaler + station-id -> index map
config/metrics.yaml        final val/test metrics for LSTM + both baselines, and per-epoch history
```

### Results

From the committed `config/metrics.yaml` (2013-12 → 2024-06 data, 106 stations):

| Model                       | Val RMSE | Val MAE | Test RMSE | Test MAE |
|------------------------------|---------:|--------:|----------:|---------:|
| **LSTM**                     |   0.7638 |  0.2943 |    0.7202 |   0.2782 |
| Naive persistence (`lag_1h`) |   0.9938 |  0.3664 |    0.9352 |   0.3454 |
| Seasonal naive (`lag_168h`)  |   1.0308 |  0.3765 |    0.9823 |   0.3555 |

The LSTM beats naive persistence by ~23% RMSE / ~19% MAE on test, and seasonal-naive by ~27%
RMSE — trained on Apple Silicon MPS, early-stopped at epoch 5 (best weights from epoch 2).
