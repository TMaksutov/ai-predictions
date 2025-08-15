# Time Series Forecasting Benchmark

A Streamlit web application for benchmarking Prophet time series forecasting on sample datasets using an optimized one-page layout.

## Overview

This application provides:
- **Benchmark results** for Prophet model only
- **RMSE evaluation** on the last 20% of each dataset (holdout test set)
- **Interactive visualization** of forecasts for selected datasets
- **10 diverse sample datasets** covering different time series patterns
- **Optimized layout** with table on left, graph on right, all content fitting on one page
- **Click-to-select**: Pick a dataset by clicking its row in the Benchmark Results table

## Dependencies

- `streamlit`: Web application framework (>=1.29.0)
- `pandas`: Data manipulation and analysis (>=2.1.0)
- `numpy`: Numerical computing (>=1.24.0)
- `matplotlib`: Plotting and visualization (>=3.7.0)
- `prophet`: Forecasting tool (>=1.1.4)
- `plotly`: Additional plotting capabilities (>=5.15.0)

### Updating Dependencies

To update dependencies in the future:

1. **Check for updates**:
   ```bash
   pip list --outdated
   ```

2. **Update specific packages**:
   ```bash
   pip install --upgrade package_name
   ```

3. **Update requirements.txt**:
   ```bash
   pip freeze > requirements.txt
   ```

## Related Work

This benchmarking application is inspired by the TSForecasting repository. For more comprehensive forecasting research and datasets, see:
- **TSForecasting Repository**: [TSForecasting_README.md](TSForecasting_README.md)
- **AI Agent Instructions**: [agents.md](agents.md)
- **Monash Time Series Forecasting Archive**: https://forecastingdata.org/