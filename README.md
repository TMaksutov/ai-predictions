# Time Series Forecasting Benchmark

A Streamlit web application for benchmarking time series forecasting models on sample datasets. This application compares Linear Regression with Polynomial Features and Prophet forecasting models.

## Overview

This application provides:
- **Benchmark comparison** of Linear Regression and Prophet models
- **RMSE evaluation** on the last 20% of each dataset (holdout test set)
- **Interactive visualization** of forecasts for selected datasets
- **10 diverse sample datasets** covering different time series patterns

## Features

### Benchmark Models
1. **Linear Regression + Polynomial Features**: Uses time-based features (day of week, month, hour, day of year) with polynomial transformations
2. **Prophet**: Facebook's Prophet model with automatic seasonality detection

### Sample Datasets
The application includes 10 synthetic datasets with various characteristics:
- Linear trends with noise
- Seasonal patterns (weekly, monthly, yearly)
- Exponential growth and decay
- Cyclical patterns
- Step changes and volatility patterns

### Evaluation Methodology
- **Training**: Uses the first 80% of each dataset
- **Testing**: Evaluates on the last 20% of each dataset
- **Metric**: Root Mean Square Error (RMSE)
- **Comparison**: Side-by-side RMSE results for both models

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The application will:
1. Display a benchmark table comparing RMSE results for both models across all 10 datasets
2. Allow you to select a specific dataset to view its forecast visualization
3. Show the forecast plot with confidence intervals

## Dependencies

- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization
- `scikit-learn`: Machine learning models and metrics
- `scipy`: Scientific computing
- `prophet`: Facebook's forecasting tool

## Dataset Information

All datasets are synthetically generated with known patterns for benchmarking purposes. Each dataset contains:
- **Date column** (`ds`): Daily timestamps
- **Value column** (`y`): Time series values
- **Length**: 150-500 data points per dataset

## Model Details

### Linear Regression Model
- Uses temporal features: time index, day of week, month, hour, day of year (sin/cos encoded)
- Applies polynomial feature transformation (degree 2)
- Fallback to simple linear regression if polynomial fitting fails

### Prophet Model
- Automatic seasonality detection (daily, weekly, yearly)
- Configurable parameters for changepoint and seasonality prior scales
- Built-in holiday effects and trend analysis

## Performance Notes

- Results are cached for faster subsequent runs
- Prophet may take longer to train but often provides better accuracy
- The application processes all 10 datasets for benchmarking

## Related Work

This benchmarking application is inspired by the TSForecasting repository. For more comprehensive forecasting research and datasets, see:
- **TSForecasting Repository**: [TSForecasting_README.md](TSForecasting_README.md)
- **Monash Time Series Forecasting Archive**: https://forecastingdata.org/

## Contributing

Feel free to contribute by:
- Adding new forecasting models
- Including additional evaluation metrics
- Expanding the dataset collection
- Improving the user interface

## License

This project is open source and available under the MIT License.