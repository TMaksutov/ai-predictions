# Prophet Benchmark - Monash Time Series Forecasting Archive

A Streamlit application that benchmarks Prophet forecasting performance on the first 10 datasets from the Monash Time Series Forecasting Archive. The app computes RMSE on a 20% holdout test set and provides interactive visualization of forecasts.

## Features

- **Automated Benchmarking**: Runs Prophet on first 10 Monash datasets automatically
- **RMSE Evaluation**: Computes Root Mean Square Error on 20% holdout test data
- **Interactive Visualization**: Select any dataset to view the forecast plot
- **Robust Data Handling**: Automatically processes diverse time series formats
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
2. **Data Preparation**: Normalizes data to Prophet format (ds, y columns) and handles multiple series
3. **Train/Test Split**: Uses 80% for training, 20% for testing
4. **Prophet Forecasting**: Fits Prophet model on training data and forecasts on full timeline
5. **RMSE Calculation**: Computes RMSE between Prophet predictions and actual test values
6. **Interactive Plots**: Click any dataset to see actual vs predicted values for test period

## Technical Details

- **Forecasting Model**: Facebook Prophet with default parameters
- **Evaluation Metric**: RMSE on 20% holdout test set
- **Data Source**: Monash Time Series Forecasting Archive (via tsdl library)
- **Data Processing**: Automatic datetime parsing, deduplication, and missing value handling
- **Visualization**: Matplotlib plots showing actual data, forecast, and train/test split

## Dependencies

- **Core**: streamlit, pandas, numpy, matplotlib
- **Forecasting**: prophet, pystan, cmdstanpy
- **Data**: tsdl (Monash Time Series Data Library)
- **Python**: 3.8+

## Benchmark Results

The application displays:
- **Summary Table**: All 10 datasets with their Prophet RMSE scores
- **Detailed View**: Select any dataset to see the forecast visualization
- **Test Period Focus**: Plots highlight the 20% test period where RMSE is calculated

## Development

All changes should be pushed directly to `main` branch.

