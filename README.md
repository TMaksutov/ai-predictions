# Time Series Forecasting Benchmark - Monash Archive

A Streamlit application that benchmarks time series forecasting performance on the first 10 datasets from the Monash Time Series Forecasting Archive. The app computes RMSE on a 20% holdout test set using Linear Regression with polynomial features and provides interactive visualization of forecasts.

## Features

- **Automated Benchmarking**: Runs forecasting on first 10 Monash datasets automatically
- **RMSE Evaluation**: Computes Root Mean Square Error on 20% holdout test data
- **Interactive Visualization**: Select any dataset to view the forecast plot
- **Robust Data Handling**: Automatically processes diverse time series formats
- **Reliable Deployment**: Uses scikit-learn based forecasting that works consistently across environments
- **Clean Interface**: Simple Streamlit UI with clear results presentation

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the benchmark app
streamlit run streamlit_app.py
```

## How It Works

1. **Dataset Loading**: Automatically loads first 10 datasets from Monash Time Series Forecasting Archive via tsdl
2. **Data Preparation**: Normalizes data to consistent format (ds, y columns) and handles multiple series
3. **Feature Engineering**: Creates time-based features including cyclical patterns (day of week, month)
4. **Train/Test Split**: Uses 80% for training, 20% for testing
5. **Forecasting**: Fits Linear Regression with polynomial features on time-based features
6. **RMSE Calculation**: Computes RMSE between predictions and actual test values
7. **Interactive Plots**: Click any dataset to see actual vs predicted values for test period

## Technical Details

- **Forecasting Model**: Linear Regression with polynomial features and time-based features
- **Features**: Time index, cyclical encodings (day of week, month), polynomial combinations
- **Evaluation Metric**: RMSE on 20% holdout test set
- **Data Source**: Monash Time Series Forecasting Archive (via tsdl library)
- **Data Processing**: Automatic datetime parsing, deduplication, and missing value handling
- **Visualization**: Matplotlib plots showing actual data, forecast, and train/test split
- **Fallback**: Simple linear regression if polynomial features fail

## Dependencies

- **Core**: streamlit, pandas, numpy, matplotlib
- **Forecasting**: scikit-learn, scipy
- **Data**: tsdl (Monash Time Series Data Library)
- **Python**: 3.8+

## Benchmark Results

The application displays:
- **Summary Table**: All 10 datasets with their forecasting RMSE scores
- **Detailed View**: Select any dataset to see the forecast visualization
- **Test Period Focus**: Plots highlight the 20% test period where RMSE is calculated

## Why This Approach?

This implementation uses Linear Regression with polynomial features instead of more complex models like Prophet because:
- **Reliability**: Works consistently across all deployment environments
- **Speed**: Fast training and prediction
- **Transparency**: Clear, interpretable model behavior
- **Robustness**: Handles various time series patterns through feature engineering

## Development

All changes should be pushed directly to `main` branch.

