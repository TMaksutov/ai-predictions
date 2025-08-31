## Daily Forecasting (Streamlit)

Lightweight app to benchmark fast scikit‑learn regressors on daily data and forecast missing tail values. Includes a compact plot and results table.

### Features
- Fast regressors (+ optional linear/quadratic trend): Ridge, Lasso, ElasticNet, KNN, GradientBoosting, RandomForest, SVR, DecisionTree, BayesianRidge, Huber, PassiveAggressive, SGD
- Feature engineering: lags, moving averages, day‑of‑week, auto‑detected Fourier seasonality
- Metric: RMSE on the last 20% split

### Quickstart
1) Install
```bash
python -m pip install -r requirements.txt
```
2) Run
```bash
streamlit run streamlit_app.py
```
If `streamlit` isn't on PATH, use:
```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

### Data
- CSV or Excel. First column = date, last column = target. Optional feature columns in between
- Dates must be consecutive daily; use ISO (YYYY‑MM‑DD) or day‑first formats (not month‑first numeric)
- To forecast, append future dates with target left blank
- Limits: file ≤ 1 MB, ≤ 10,000 rows

Sample file: `sample.csv`

See `requirements.txt` for versions.
