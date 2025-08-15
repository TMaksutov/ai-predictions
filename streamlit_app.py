import math
from typing import List, Tuple, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from prophet import Prophet


st.set_page_config(page_title="Time Series Forecasting Benchmark", layout="wide")
st.title("Time Series Forecasting Benchmark — 10 Sample Datasets")
st.caption("RMSE on last 20% holdout using Linear Regression + Polynomial Features. Select a dataset to view its forecast plot.")


def _generate_sample_datasets() -> Dict[str, pd.DataFrame]:
    """Generate 10 diverse sample time series datasets for benchmarking."""
    datasets = {}
    np.random.seed(42)  # For reproducible results
    
    # Dataset 1: Linear trend with noise
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    trend = np.linspace(100, 200, 200)
    noise = np.random.normal(0, 5, 200)
    datasets['Linear_Trend'] = pd.DataFrame({
        'ds': dates,
        'y': trend + noise
    })
    
    # Dataset 2: Seasonal pattern (weekly)
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    seasonal = 50 + 20 * np.sin(2 * np.pi * np.arange(300) / 7)  # Weekly pattern
    noise = np.random.normal(0, 3, 300)
    datasets['Weekly_Seasonal'] = pd.DataFrame({
        'ds': dates,
        'y': seasonal + noise
    })
    
    # Dataset 3: Exponential growth
    dates = pd.date_range('2020-01-01', periods=150, freq='D')
    growth = 10 * np.exp(0.01 * np.arange(150))
    noise = np.random.normal(0, growth * 0.05)  # Proportional noise
    datasets['Exponential_Growth'] = pd.DataFrame({
        'ds': dates,
        'y': growth + noise
    })
    
    # Dataset 4: Multiple seasonality (daily + weekly)
    dates = pd.date_range('2020-01-01', periods=400, freq='H')
    daily = 10 * np.sin(2 * np.pi * np.arange(400) / 24)  # Daily pattern
    weekly = 5 * np.sin(2 * np.pi * np.arange(400) / (24*7))  # Weekly pattern
    base = 100
    noise = np.random.normal(0, 2, 400)
    datasets['Hourly_Multi_Seasonal'] = pd.DataFrame({
        'ds': dates,
        'y': base + daily + weekly + noise
    })
    
    # Dataset 5: Step change
    dates = pd.date_range('2020-01-01', periods=250, freq='D')
    values = np.ones(250) * 50
    values[125:] += 30  # Step change halfway
    noise = np.random.normal(0, 4, 250)
    datasets['Step_Change'] = pd.DataFrame({
        'ds': dates,
        'y': values + noise
    })
    
    # Dataset 6: Cyclical pattern (annual-like)
    dates = pd.date_range('2020-01-01', periods=365*2, freq='D')
    annual = 75 + 25 * np.sin(2 * np.pi * np.arange(365*2) / 365)
    noise = np.random.normal(0, 3, 365*2)
    datasets['Annual_Cycle'] = pd.DataFrame({
        'ds': dates,
        'y': annual + noise
    })
    
    # Dataset 7: Random walk
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    random_walk = np.cumsum(np.random.normal(0, 2, 200)) + 100
    datasets['Random_Walk'] = pd.DataFrame({
        'ds': dates,
        'y': random_walk
    })
    
    # Dataset 8: Polynomial trend
    dates = pd.date_range('2020-01-01', periods=180, freq='D')
    x = np.arange(180) / 180
    polynomial = 50 + 30*x + 20*x**2 - 10*x**3
    noise = np.random.normal(0, 4, 180)
    datasets['Polynomial_Trend'] = pd.DataFrame({
        'ds': dates,
        'y': polynomial + noise
    })
    
    # Dataset 9: Damped oscillation
    dates = pd.date_range('2020-01-01', periods=220, freq='D')
    t = np.arange(220)
    damped = 100 + 50 * np.exp(-t/100) * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 3, 220)
    datasets['Damped_Oscillation'] = pd.DataFrame({
        'ds': dates,
        'y': damped + noise
    })
    
    # Dataset 10: Mixed patterns
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    trend = 0.1 * np.arange(300)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(300) / 50)  # Custom period
    weekly = 5 * np.sin(2 * np.pi * np.arange(300) / 7)
    base = 80
    noise = np.random.normal(0, 4, 300)
    datasets['Mixed_Patterns'] = pd.DataFrame({
        'ds': dates,
        'y': base + trend + seasonal + weekly + noise
    })
    
    return datasets


def _get_dataset_names() -> List[str]:
    """Get list of available dataset names."""
    return [
        'Linear_Trend', 'Weekly_Seasonal', 'Exponential_Growth', 
        'Hourly_Multi_Seasonal', 'Step_Change', 'Annual_Cycle',
        'Random_Walk', 'Polynomial_Trend', 'Damped_Oscillation', 'Mixed_Patterns'
    ]


@st.cache_data(show_spinner=False)
def _load_dataset(name: str) -> Tuple[pd.DataFrame, dict]:
    """Load a specific dataset by name."""
    datasets = _generate_sample_datasets()
    if name not in datasets:
        raise ValueError(f"Dataset {name} not found")
    
    data = datasets[name].copy()
    metadata = {"name": name, "description": f"Generated sample dataset: {name}"}
    return data, metadata


def _prepare_single_series(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset for forecasting."""
    # Already in correct format (ds, y)
    sub = df[["ds", "y"]].copy()
    
    # Ensure proper types
    sub["ds"] = pd.to_datetime(sub["ds"])
    sub["y"] = pd.to_numeric(sub["y"], errors="coerce")
    sub = sub.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    
    # Deduplicate by timestamp (shouldn't be needed for generated data)
    sub = sub.groupby("ds", as_index=False).agg({"y": "mean"})
    
    return sub


def _compute_forecast_and_rmse(series_df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Compute forecasts using Linear Regression with polynomial features and time-based features.
    This is a reliable forecasting method that works well in deployment environments.
    """
    n = len(series_df)
    if n < 10:
        raise ValueError("Series too short for 20% holdout evaluation.")

    test_size = max(1, int(math.ceil(n * test_fraction)))
    train_df = series_df.iloc[:-test_size].copy()
    test_df = series_df.iloc[-test_size:].copy()

    # Create time-based features
    def create_features(df):
        features_df = df.copy()
        features_df['timestamp'] = features_df['ds'].astype('int64') // 10**9  # Unix timestamp
        
        # Normalize timestamp
        min_ts = features_df['timestamp'].min()
        features_df['time_idx'] = features_df['timestamp'] - min_ts
        if features_df['time_idx'].std() > 0:
            features_df['time_idx'] = features_df['time_idx'] / features_df['time_idx'].std()
        
        # Add cyclical features
        features_df['day_of_week'] = features_df['ds'].dt.dayofweek
        features_df['month'] = features_df['ds'].dt.month
        features_df['hour'] = features_df['ds'].dt.hour
        features_df['day_of_year'] = features_df['ds'].dt.dayofyear
        
        # Sine/cosine encoding for cyclical features
        features_df['dow_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['dow_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['doy_sin'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
        features_df['doy_cos'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
        
        return features_df
    
    # Prepare features
    train_features = create_features(train_df)
    
    # Select feature columns
    feature_cols = ['time_idx', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 
                   'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos']
    
    X_train = train_features[feature_cols].values
    y_train = train_features['y'].values
    
    # Create polynomial features and fit model
    try:
        # Try polynomial features with linear regression
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ])
        model.fit(X_train, y_train)
    except:
        # Fallback to simple linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
    
    # Create features for full series to get forecast
    full_features = create_features(series_df)
    X_full = full_features[feature_cols].values
    
    # Predict
    y_pred = model.predict(X_full)
    
    # Create forecast dataframe
    forecast = pd.DataFrame({
        'ds': series_df['ds'],
        'yhat': y_pred,
        'yhat_lower': y_pred * 0.95,  # Simple confidence interval
        'yhat_upper': y_pred * 1.05
    })
    
    # Calculate RMSE on test set
    yhat_test = y_pred[-test_size:]
    y_true = test_df['y'].to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, yhat_test)))

    return rmse, forecast, test_df


def _compute_prophet_forecast_and_rmse(series_df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Compute Prophet forecast and RMSE on test set.
    
    Args:
        series_df: DataFrame with 'ds' and 'y' columns
        test_fraction: Fraction of data to use for testing
        
    Returns:
        tuple: (rmse, forecast_df, test_df)
    """
    # Split data
    test_size = int(len(series_df) * test_fraction)
    train_df = series_df.iloc[:-test_size].copy()
    test_df = series_df.iloc[-test_size:].copy()
    
    # Train Prophet model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # Suppress Prophet output
    import logging
    logging.getLogger('prophet').setLevel(logging.WARNING)
    
    model.fit(train_df)
    
    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=test_size)
    forecast = model.predict(future)
    
    # Calculate RMSE on test set
    yhat_test = forecast['yhat'].iloc[-test_size:].values
    y_true = test_df['y'].to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, yhat_test)))
    
    return rmse, forecast, test_df


@st.cache_data(show_spinner=False)
def _compute_benchmark(dataset_names: Tuple[str, ...]) -> pd.DataFrame:
    """Compute benchmark results for all datasets."""
    results = []
    for name in dataset_names:
        try:
            raw_df, _ = _load_dataset(name)
            series_df = _prepare_single_series(raw_df)
            
            # Compute Linear Regression RMSE
            lr_rmse, _, _ = _compute_forecast_and_rmse(series_df, test_fraction=0.2)
            
            # Compute Prophet RMSE
            prophet_rmse, _, _ = _compute_prophet_forecast_and_rmse(series_df, test_fraction=0.2)
            
            results.append({
                "Dataset": name, 
                "Linear Regression RMSE": f"{lr_rmse:.4f}",
                "Prophet RMSE": f"{prophet_rmse:.4f}"
            })
        except Exception as e:
            results.append({
                "Dataset": name, 
                "Linear Regression RMSE": "Error", 
                "Prophet RMSE": "Error",
                "Error": str(e)[:50]
            })
    return pd.DataFrame(results)


# Load dataset names
dataset_names = _get_dataset_names()

# Benchmark table
st.subheader("Benchmark Results (Linear Regression and Prophet RMSE on last 20%)")
with st.spinner("Computing benchmark for 10 sample datasets..."):
    bench_df = _compute_benchmark(tuple(dataset_names))

st.dataframe(bench_df, use_container_width=True)

# Selection and plot
st.subheader("Forecast plot for selected dataset")
selected = st.selectbox("Select dataset to visualize", dataset_names, index=0)

if selected:
    with st.spinner(f"Loading and forecasting: {selected}"):
        raw_df, metadata = _load_dataset(selected)
        series_df = _prepare_single_series(raw_df)
        rmse, forecast, test_df = _compute_forecast_and_rmse(series_df, test_fraction=0.2)

    st.caption(f"**Dataset**: {metadata['description']}")
    st.caption(f"**RMSE**: {rmse:.4f} | **Data Points**: {len(series_df)} | **Test Size**: {len(test_df)}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot full actual data
    ax.plot(series_df["ds"], series_df["y"], color="#333333", label="Actual", linewidth=2, alpha=0.8)

    # Plot forecast for test period only
    test_pred = forecast.tail(len(test_df))
    ax.plot(test_pred["ds"], test_pred["yhat"], color="#1f77b4", linestyle="--", linewidth=2, label="Forecast (test period)")
    
    # Add confidence interval for test period
    ax.fill_between(test_pred["ds"], test_pred["yhat_lower"], test_pred["yhat_upper"], 
                   color="#1f77b4", alpha=0.2, label="Confidence interval")

    # Split marker
    if len(test_df) > 0:
        ax.axvline(test_df["ds"].iloc[0], color="#888888", linestyle=":", linewidth=1, label="Train/Test split")

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"Time Series Forecast: {selected}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    st.pyplot(fig)
    
    # Display some stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{rmse:.3f}")
    with col2:
        st.metric("Train Size", f"{len(series_df) - len(test_df)}")
    with col3:
        st.metric("Test Size", f"{len(test_df)}")
    with col4:
        st.metric("Mean Value", f"{series_df['y'].mean():.1f}")

st.info("💡 This benchmark uses 10 synthetically generated time series datasets with different patterns (trends, seasonality, cycles, etc.) to test forecasting performance.")
