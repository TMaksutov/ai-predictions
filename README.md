# Time Series Forecasting (Daily Regression)

A lightweight Streamlit app that benchmarks a simple multiple linear regression on daily time series and visualizes the forecast versus actuals.

## Overview

- **Model**: Multiple linear regression with lag (1, 7) and day-of-week features
- **Metric**: nRMSE on the last 20% of each dataset (holdout)
- **Datasets**: CSVs with two columns: time (first), target (last), daily frequency
- **UI**: One-page layout with a results table and a simple forecast plot

## How to run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What you’ll see

- A table with per‑dataset nRMSE (computed on demand)
- Pick a dataset and a Matplotlib plot of actuals and forecast for the test window

## Dependencies

- streamlit (app)
- pandas, numpy (data)
- matplotlib (plot)
- scikit-learn (regression)

See `requirements.txt` for exact versions.

## Notes

- Assumes daily frequency with regular 1‑day intervals (no gaps). If not, please resample/fill before use.
- Uses Streamlit session state to keep selection across edits.