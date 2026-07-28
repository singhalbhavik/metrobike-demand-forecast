# MetroBike Demand Forecasting

Forecast hourly station-level demand for Austin's MetroBike system.

## Stack
Python 3.11, BigQuery (bigquery-public-data.austin_bikeshare + NOAA GSOD weather),
PyTorch (LSTM), LangChain copilot, Docker. Venv at .venv, deps pinned in requirements.txt.

## Rules
- Always activate .venv; add new deps to requirements.txt
- BigQuery auth is application-default credentials; never create key files
- Strict time-based train/val/test splits — no shuffling time series, no future leakage
- Every stage ends with: code runs end to end, results printed, committed to git
- Never fabricate metrics; all numbers must come from actual runs
- Keep queries cheap: use table partitioning filters, LIMIT while developing1