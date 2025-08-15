# Simple Time-Series Predictor (Streamlit)

A clean, simple Streamlit app for time-series forecasting. Upload CSV/XLS/XLSX files, select your date and target columns, and get baseline forecasts using linear regression with automatic fallback.

## Features

- **Simple Interface**: Clean, intuitive UI with automatic column detection
- **Robust Forecasting**: Linear regression with fallback to last value if modeling fails
- **File Support**: CSV, Excel (.xlsx, .xls) files up to 10 MB
- **Auto-detection**: Automatically identifies date columns and numeric targets
- **Interval Detection**: Detects time series frequency (daily, hourly, etc.)
- **Sample Data**: Includes example datasets for testing

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## How It Works

1. **Upload Data**: Upload your CSV or Excel file
2. **Auto-detect**: The app automatically detects date and target columns
3. **Select Columns**: Manually adjust column selections if needed
4. **Generate Forecast**: Forecasts are generated automatically; small datasets fall back to a simple model
5. **Download Results**: Export your forecasts as CSV

## Technical Details

- **Forecasting**: Uses scikit-learn LinearRegression on time indices
- **Fallback**: If modeling fails, falls back to naive forecast (last value)
- **Data Cleaning**: Automatic handling of missing values and duplicates
- **Safety Checks**: Built-in validation for file size and data quality

## Sample Data

The `test_files/` directory contains example datasets covering different time intervals:
- `daily.csv` - Daily data
- `hourly.csv` - Hourly data  
- `weekly_multi.csv` - Weekly data with multiple features
- `minute_multi.csv` - Minute-level data

## Dependencies

- **Core**: streamlit, pandas, numpy, scikit-learn
- **File Support**: openpyxl, xlrd
- **Python**: 3.8+

## Development

- All changes should be pushed directly to `main` (no branches/PRs).

Run tests locally before committing:
```bash
pytest
```

