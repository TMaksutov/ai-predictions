# Time Series Forecasting Benchmark (AutoTS)

A lightweight Streamlit app that benchmarks a minimal AutoTS setup on 10 synthetic time series and visualizes the forecast versus actuals.

## Overview

- **Models**: AutoTS with a small model list
- **Metric**: NRMSE on the last 20% of each dataset (holdout)
- **Datasets**: 10 diverse synthetic series
- **UI**: One-page layout with a results table and a simple forecast plot

## How to run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What you’ll see

- A table with per‑dataset NRMSE (computed inline without caching)
- A dropdown to choose a dataset and a Matplotlib plot of actuals and forecast for the test window

## Dependencies

- streamlit (app)
- pandas, numpy (data)
- matplotlib (plot)
- AutoTS (forecasting)

See `requirements.txt` for exact versions.

## Notes

- No Streamlit caching or session state is used.
- No spinners, confidence bands, or extra UI chrome.
- External TSF/Prophet content was removed for clarity.