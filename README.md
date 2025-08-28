# Time Series Forecasting (Daily Regression)

A lightweight Streamlit app that benchmarks several fast regressors and a few classic time‑series baselines on daily data, and visualizes the forecast versus actuals.

## Overview

- **Models (scikit‑learn)**: Ridge, Lasso, ElasticNet, KNN, GradientBoosting, RandomForest, SVR
- **Baselines (optional)**: SARIMA, Prophet (yearly+weekly). Optional libs auto‑disable if not installed
- **Features**: Lags, moving averages, day‑of‑week, global trend, Fourier seasonality (weekly, monthly, quarterly, yearly, biannual)
- **Architecture**: Unified training and retraining system with consistent feature engineering
- **Metric**: RMSE on the last 20% of each dataset (holdout)
- **Datasets**: CSVs with time (first), optional features (middle), target (last)
- **UI**: One‑page layout with a results table and a simple forecast plot

### Project layout recommendation

- All helper modules live at the project root for quick navigation: `config.py`, `data_utils.py`, `plot_utils.py`.
- There is no `utils/` package anymore. You can open any module directly without drilling into subfolders.

## How to run

1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

2) Start the app (recommended: bind to localhost and suppress first‑run email prompt)

- Windows (PowerShell):

Option A — use the venv's Python directly (no activation required):

```powershell
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"; .\venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Option B — if you prefer to activate the venv first:

```powershell
& .\venv\Scripts\Activate.ps1
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"; python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

- macOS/Linux (bash/zsh):

```bash
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false python -m streamlit run streamlit_app.py --server.address localhost --server.port 8501 --server.headless true
```

Then open `http://localhost:8501` in your browser. Stop with Ctrl+C.

Notes:
- If you prefer the default behavior, you can still run: `streamlit run streamlit_app.py`.
- If `streamlit` is not on PATH, use `python -m streamlit` as shown above.

You can upload your own CSV (time column first, target last), or just run with the bundled `sample.csv`.

### Quick run in Cursor (Windows)

Open the Cursor terminal at the project root and run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"; .\venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Then open `http://localhost:8501`.

### Troubleshooting

- If `http://localhost:8501` doesn’t load, try changing the port:

```powershell
.\n+\venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true
```

- On Windows PowerShell, avoid piping Streamlit output to `| cat` — it can cause errors. Run the command directly as shown above.

## Data format and workflow

- Provide history with known target values.
- Append future dates at the end with target left blank and all feature columns filled (if features are used).
- The app trains on known rows and predicts targets for the trailing future rows you provided. There is no auto 20% extension mode.

Example with features:

```csv
date,price_tier,web_traffic_k,is_weekend,product_line,target
2024-03-29,1,11.5,No,Beta,113.8
2024-03-30,2,89.2,Yes,Gamma,147.1
2024-03-31,1,98.3,Yes,Alpha,
2024-04-01,2,19.8,No,Beta,
```

Example without features:

```csv
Date,Sold
2019-11-29,837
2019-11-30,842
2019-12-01,
2019-12-02,
```

## What you’ll see

- A table with per‑dataset RMSE (computed on demand)
- Pick a dataset and a Matplotlib plot of actuals and forecast for the test window

## Dependencies

- streamlit (app)
- pandas, numpy (data)
- matplotlib (plot)
- scikit-learn (regression)

See `requirements.txt` for exact versions.
