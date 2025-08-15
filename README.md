# Time Series Forecasting Benchmark

A self-contained Streamlit application that benchmarks time series forecasting performance on 10 diverse sample datasets. The app computes RMSE on a 20% holdout test set using Linear Regression with polynomial features and provides interactive visualization of forecasts.

## Features

- **Automated Benchmarking**: Runs forecasting on 10 diverse sample datasets automatically
- **RMSE Evaluation**: Computes Root Mean Square Error on 20% holdout test data
- **Interactive Visualization**: Select any dataset to view the forecast plot with confidence intervals
- **Diverse Test Cases**: 10 different time series patterns (trends, seasonality, cycles, etc.)
- **Self-Contained**: No external dependencies - generates sample data internally
- **Reliable Deployment**: Uses scikit-learn based forecasting that works consistently across environments
- **Clean Interface**: Modern Streamlit UI with metrics and clear visualizations

## Sample Datasets

The benchmark includes 10 synthetically generated datasets with different characteristics:

1. **Linear_Trend** - Simple linear trend with noise
2. **Weekly_Seasonal** - Weekly seasonal pattern
3. **Exponential_Growth** - Exponential growth pattern
4. **Hourly_Multi_Seasonal** - Multiple seasonality (daily + weekly)
5. **Step_Change** - Step change in the middle of series
6. **Annual_Cycle** - Annual cyclical pattern
7. **Random_Walk** - Random walk process
8. **Polynomial_Trend** - Polynomial trend pattern
9. **Damped_Oscillation** - Damped oscillating pattern
10. **Mixed_Patterns** - Combination of trend, seasonal, and weekly patterns

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

1. **Dataset Generation**: Creates 10 diverse synthetic time series with different patterns
2. **Data Preparation**: Ensures consistent format (ds, y columns) for all datasets
3. **Feature Engineering**: Creates comprehensive time-based features:
   - Normalized time index for trend capture
   - Cyclical encodings (day of week, month, hour, day of year)
   - Sine/cosine transformations for seasonal patterns
4. **Train/Test Split**: Uses 80% for training, 20% for testing
5. **Forecasting**: Fits Linear Regression with polynomial features on engineered features
6. **RMSE Calculation**: Computes RMSE between predictions and actual test values
7. **Interactive Analysis**: Detailed visualization with confidence intervals and metrics

## Technical Details

- **Forecasting Model**: Linear Regression with polynomial features (degree 2)
- **Feature Engineering**: Comprehensive time-based features with cyclical encodings
- **Evaluation Metric**: RMSE on 20% holdout test set
- **Data Generation**: Reproducible synthetic datasets (seed=42)
- **Fallback**: Simple linear regression if polynomial features fail
- **Visualization**: Matplotlib plots with confidence intervals and split markers

## Dependencies

- **Core**: streamlit, pandas, numpy, matplotlib
- **Machine Learning**: scikit-learn, scipy
- **Python**: 3.8+

## Why This Approach?

This implementation provides several advantages:

- **Self-Contained**: No external data dependencies or API calls
- **Reliability**: Works consistently across all deployment environments
- **Speed**: Fast training and prediction on all datasets
- **Transparency**: Clear, interpretable model behavior
- **Diversity**: Tests various time series patterns and edge cases
- **Robustness**: Handles different frequencies, trends, and seasonal patterns

## Benchmark Results

The application displays:
- **Summary Table**: All 10 datasets with their RMSE scores
- **Interactive Visualization**: Select any dataset to see detailed forecast plots
- **Performance Metrics**: RMSE, data points, train/test sizes
- **Confidence Intervals**: Visual uncertainty quantification
- **Pattern Analysis**: See how the model handles different time series patterns

## Development

All changes should be pushed directly to `main` branch.

